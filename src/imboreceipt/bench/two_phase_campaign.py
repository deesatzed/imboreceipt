"""End-to-end Two-Phase Discovery + Streaming Selective Acquisition campaign & simulation suite.

Produces:
1. Monte Carlo Statistical Theory Verification (Sim A: 95% CI coverage & null FWER;
   Sim B: Neyman vs SRS/BCC variance; Sim C: Stage-2 twin redundancy vs LOCO cancellation).
2. R=20 Resample Fine-Grained Real-World Campaign (M=15..26 native channels + M=25..36
   null/twin/echo stress pools) with Schuirmann TOST non-inferiority vs 100%-row discovery,
   proper conditional competitors (CPI, LOCO, CRT-Knockoff, DIME-CMI), streaming budget-utility
   curves, Jin & Candes (2023) Conformal FDR selection, and clinical fairness/missingness audits.
3. Supplement B: R=20 Resample Fixed-Table Suite across 10 public datasets with explicit
   max_delta_step ablation, HPO-tuned SMOTE, ECE/calibration slope, TOST equivalence, and
   random-effects meta-analysis.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

from imboreceipt.bench.datasets import DEFAULT_CACHE_DIR, TIER1_PUBLIC_SPECS, load_tier1_dataset
from imboreceipt.bench.fine_grained_cohorts import (
    FineGrainedCohort,
    load_fine_diabetes_discharge,
    load_fine_home_credit,
    load_fine_ieee_cis,
    load_fine_neurips_baf,
    load_fine_scania_aps,
    load_fine_taiwan_credit,
)
from imboreceipt.core.bootstrap import (
    holm_bonferroni_adjust,
    nadeau_bengio_ttest,
    random_effects_meta_analysis,
    tost_equivalence_and_noninferiority,
)
from imboreceipt.core.types import SemanticType, freeze_split
from imboreceipt.discover.selector import CandidateChannelSpec
from imboreceipt.discover.strata import make_leak_free_cv_splits
from imboreceipt.discover.two_phase import (
    discover_competitor_method,
    discover_sequential_conditional_gain,
    sample_two_phase_pilot,
)
from imboreceipt.frontier.baselines import PlattLogitCalibrator, TabularPreprocessor
from imboreceipt.lattice.streaming_routing import (
    StreamingSelectiveRouter,
    compute_calibration_summary,
    precision_at_target_recall,
)

_PROB_EPS = 1e-12


def _agg(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "sd": 0.0, "se": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    m = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(sd / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "mean": m,
        "sd": sd,
        "se": se,
        "ci95_low": m - 1.96 * se,
        "ci95_high": m + 1.96 * se,
    }


def _fit_oof_stage1_baseline(
    X_base_train: pd.DataFrame,
    y_train: np.ndarray,
    sem_types: dict[str, SemanticType],
    groups: np.ndarray | None,
    time_order: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute leak-free Phase-1 OOF baseline probabilities and causal valid mask."""
    n = len(X_base_train)
    pi = max(1e-5, min(1.0 - 1e-5, float(np.mean(y_train))))
    oof_p = np.full(n, pi, dtype=float)
    valid = np.zeros(n, dtype=bool)
    splits = make_leak_free_cv_splits(
        y_train, n_folds=5, groups=groups, time_order=time_order, seed=seed
    )
    for f_i, (tr_i, va_i) in enumerate(splits):
        if np.unique(y_train[tr_i]).size < 2:
            continue
        prep = TabularPreprocessor(semantic_types=sem_types)
        X_tr = prep.fit_transform(X_base_train.iloc[tr_i])
        X_va = prep.transform(X_base_train.iloc[va_i])
        clf = lgb.LGBMClassifier(
            n_estimators=90,
            learning_rate=0.06,
            num_leaves=31,
            min_child_samples=max(15, min(40, int(tr_i.size // 40))),
            subsample_for_bin=min(60000, int(tr_i.size)),
            reg_alpha=0.5,
            reg_lambda=3.0,
            max_delta_step=1.0,
            random_state=seed + f_i,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X_tr, y_train[tr_i])
        tr_raw = np.asarray(clf.predict_proba(X_tr)[:, 1], dtype=float)
        va_raw = np.asarray(clf.predict_proba(X_va)[:, 1], dtype=float)
        cal = PlattLogitCalibrator().fit(tr_raw, y_train[tr_i])
        oof_p[va_i] = np.clip(cal.predict_proba(va_raw), _PROB_EPS, 1.0 - _PROB_EPS)
        valid[va_i] = True
    return oof_p, valid


def _fit_downstream_calibrated_gbdt(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    df_test: pd.DataFrame,
    sem_types: dict[str, SemanticType],
    seed: int,
    max_delta_step: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train stabilized LightGBM on D_train, Platt-calibrate on D_cal, and predict D_cal/D_test."""
    prep = TabularPreprocessor(semantic_types=sem_types)
    X_tr = prep.fit_transform(df_train)
    X_ca = prep.transform(df_cal)
    X_te = prep.transform(df_test)

    clf = lgb.LGBMClassifier(
        n_estimators=160,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=max(15, min(50, int(len(df_train) // 50))),
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=3.0,
        max_delta_step=max_delta_step,
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(X_tr, y_train)
    raw_ca = np.asarray(clf.predict_proba(X_ca)[:, 1], dtype=float)
    raw_te = np.asarray(clf.predict_proba(X_te)[:, 1], dtype=float)

    cal = PlattLogitCalibrator().fit(raw_ca, y_cal)
    p_ca = np.clip(cal.predict_proba(raw_ca), _PROB_EPS, 1.0 - _PROB_EPS)
    p_te = np.clip(cal.predict_proba(raw_te), _PROB_EPS, 1.0 - _PROB_EPS)
    return p_ca, p_te, raw_te


def _eval_downstream_predictions(
    y_test: np.ndarray,
    p_test: np.ndarray,
    cohort: FineGrainedCohort,
    n_selected_channels: int,
) -> dict[str, float]:
    ap = float(average_precision_score(y_test, p_test))
    p50 = precision_at_target_recall(y_test, p_test, target_recall=0.50)
    cal = compute_calibration_summary(y_test, p_test)

    tau_star = cohort.contract.cost_fp / (cohort.contract.cost_fp + cohort.contract.cost_fn)
    act_pos = p_test >= tau_star
    fp = float(np.sum((y_test == 0) & act_pos))
    fn = float(np.sum((y_test == 1) & (~act_pos)))
    n_t = max(1.0, float(y_test.size))

    q_cost_full = cohort.per_instance_query_cost * (
        float(n_selected_channels) / max(1.0, cohort.acquisition_budget)
    )
    total_cost_1k = (1000.0 / n_t) * (
        cohort.contract.cost_fp * fp + cohort.contract.cost_fn * fn + q_cost_full * n_t
    )
    treat_none_1k = (1000.0 / n_t) * cohort.contract.cost_fn * float(np.sum(y_test == 1))
    nb_1k = (treat_none_1k - total_cost_1k) / max(1e-9, cohort.contract.cost_fn)

    return {
        "auprc": ap,
        "prec_at_50_recall": p50,
        "ece": cal.ece,
        "brier": cal.brier,
        "cal_intercept": cal.cal_intercept,
        "cal_slope": cal.cal_slope,
        "net_benefit_per_1k": float(nb_1k),
        "total_cost_per_1k": float(total_cost_1k),
    }


def run_statistical_theory_simulations(
    n_rep_coverage: int = 80,
    n_rep_mechanisms: int = 40,
    seed: int = 100,
) -> dict[str, Any]:
    """Verify Theorem 1 (CI coverage & null FWER), Prop 1 (Neyman variance), and twin dynamics."""
    # --- Sim A: Empirical 95% CI Coverage & Pure-Noise Null FWER ---
    g_hats: list[float] = []
    se_tots: list[float] = []
    se_p1s: list[float] = []
    se_p2s: list[float] = []
    g_census_list: list[float] = []
    null_fwer_holm: list[float] = []
    null_fwer_untruncated_no_lcb: list[float] = []
    null_fwer_pospart_bug: list[float] = []

    # Fix a large superpopulation (N=120,000) to compute asymptotic cross-fitted functional G_1_inf
    rng_pop = np.random.default_rng(seed)
    x_pop = rng_pop.normal(0.0, 1.0, (120000, 3))
    z1_pop = rng_pop.normal(0.0, 1.0, 120000)
    eta_pop = -3.4 + 0.9 * x_pop[:, 0] - 0.5 * x_pop[:, 1] + 1.05 * z1_pop
    p_pop = 1.0 / (1.0 + np.exp(-eta_pop))
    y_pop = rng_pop.binomial(1, p_pop)
    # Exact population conditional mutual information I(Y; Z_1 | X_base) in nats under the DGP
    p_base_dgp = np.mean(
        1.0
        / (
            1.0
            + np.exp(
                -(
                    -3.4
                    + 0.9 * x_pop[:, 0:1]
                    - 0.5 * x_pop[:, 1:2]
                    + 1.05 * rng_pop.normal(0.0, 1.0, (120000, 40))
                )
            )
        ),
        axis=1,
    )
    ll_full_dgp = y_pop * np.log(np.clip(p_pop, _PROB_EPS, 1.0 - _PROB_EPS)) + (
        1.0 - y_pop
    ) * np.log(np.clip(1.0 - p_pop, _PROB_EPS, 1.0 - _PROB_EPS))
    ll_base_dgp = y_pop * np.log(np.clip(p_base_dgp, _PROB_EPS, 1.0 - _PROB_EPS)) + (
        1.0 - y_pop
    ) * np.log(np.clip(1.0 - p_base_dgp, _PROB_EPS, 1.0 - _PROB_EPS))
    cmi_dgp_truth = float(np.mean(ll_full_dgp - ll_base_dgp))

    for rep in range(n_rep_coverage):
        r_seed = seed + 1000 + rep
        rng = np.random.default_rng(r_seed)
        n = 6000
        x = rng.normal(0.0, 1.0, (n, 3))
        z1 = rng.normal(0.0, 1.0, n)
        eta = -3.4 + 0.9 * x[:, 0] - 0.5 * x[:, 1] + 1.05 * z1
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

        X_b = pd.DataFrame(x, columns=["x0", "x1", "x2"])
        z_cols = {"z_true": z1.astype(np.float32)}
        cands = [CandidateChannelSpec("ch_true", ("z_true",), 1.0, "true")]
        for j in range(1, 9):
            zn = rng.normal(0.0, 1.0, n).astype(np.float32)
            z_cols[f"z_null_{j}"] = zn
            cands.append(CandidateChannelSpec(f"ch_null_{j}", (f"z_null_{j}",), 1.0, "null"))
        Z_df = pd.DataFrame(z_cols)

        oof_b, val_m = _fit_oof_stage1_baseline(X_b, y, {}, None, None, seed=r_seed)
        pilot_full = sample_two_phase_pilot(y, oof_b, val_m, pilot_fraction=1.0, design="full")
        res_census = discover_sequential_conditional_gain(
            X_b,
            Z_df,
            y,
            [cands[0]],
            oof_b,
            pilot_full,
            max_budget=1.0,
            seed=r_seed,
        )
        g_census_list.append(float(res_census.initial_scores["ch_true"]))

        pilot = sample_two_phase_pilot(
            y, oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=r_seed
        )
        res = discover_sequential_conditional_gain(
            X_b,
            Z_df,
            y,
            cands,
            oof_b,
            pilot,
            max_budget=3.0,
            multiplicity="holm",
            alpha_select=0.05,
            seed=r_seed,
        )
        g_hat = res.initial_scores["ch_true"]
        se_tot = res.initial_ses["ch_true"]
        se_p1 = res.step_trace[0].se_phase1 if res.step_trace else se_tot
        se_p2 = res.step_trace[0].se_phase2 if res.step_trace else 0.0
        g_hats.append(g_hat)
        se_tots.append(se_tot)
        se_p1s.append(se_p1)
        se_p2s.append(se_p2)

        any_null_chosen = any(cid.startswith("ch_null_") for cid in res.selected_ids)
        null_fwer_holm.append(float(any_null_chosen))
        # Empirical null selection when picking any channel with untruncated G_hat > 0
        null_untrunc_pos = any(
            res.initial_scores.get(f"ch_null_{j}", 0.0) > 0.0 for j in range(1, 9)
        )
        null_fwer_untruncated_no_lcb.append(float(null_untrunc_pos))
        # Empirical null selection under per-record positive-part [psi_i]_+ without LCB
        null_pospart_pos = any(
            res.initial_pospart_scores.get(f"ch_null_{j}", 0.0) > 0.0 for j in range(1, 9)
        )
        null_fwer_pospart_bug.append(float(null_pospart_pos))

    g_target_mc = float(np.mean(g_hats))
    g_census_mean = float(np.mean(g_census_list))
    biases_vs_census = [gh - gc for gh, gc in zip(g_hats, g_census_list, strict=True)]
    cov_total_hits = [
        float(abs(gh - g_target_mc) <= 1.96 * st) for gh, st in zip(g_hats, se_tots, strict=True)
    ]
    cov_phase1_only_hits = [
        float(abs(gh - g_target_mc) <= 1.96 * s1) for gh, s1 in zip(g_hats, se_p1s, strict=True)
    ]
    cov_vs_census_hits = [
        float(abs(gh - gc) <= 1.96 * st)
        for gh, gc, st in zip(g_hats, g_census_list, se_tots, strict=True)
    ]

    # --- Sim B: Neyman Two-Phase Allocation vs SRS & BCC Across Pilot Fractions ---
    alloc_rmse: dict[str, dict[str, list[float]]] = {
        f"{int(pf * 100)}pct": {"neyman": [], "srs": [], "bcc": []} for pf in (0.05, 0.10, 0.20)
    }
    for rep in range(n_rep_mechanisms):
        r_seed = seed + 3000 + rep
        rng = np.random.default_rng(r_seed)
        n = 8000
        x = rng.normal(0.0, 1.0, (n, 3))
        z1 = rng.normal(0.0, 1.0, n)
        z2 = rng.normal(0.0, 1.0, n)
        eta = -3.7 + 1.1 * x[:, 0] + 0.95 * z1 + 0.65 * z2 * (x[:, 0] > 0.5)
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
        X_b = pd.DataFrame(x, columns=["x0", "x1", "x2"])
        Z_df = pd.DataFrame({"z1": z1, "z2": z2, "z_null": rng.normal(0.0, 1.0, n)})
        cands = [
            CandidateChannelSpec("c1", ("z1",), 1.0, "sig"),
            CandidateChannelSpec("c2", ("z2",), 1.0, "sig"),
            CandidateChannelSpec("c0", ("z_null",), 1.0, "null"),
        ]
        oof_b, val_m = _fit_oof_stage1_baseline(X_b, y, {}, None, None, seed=r_seed)
        pil_full = sample_two_phase_pilot(y, oof_b, val_m, pilot_fraction=1.0, design="full")
        res_full = discover_sequential_conditional_gain(
            X_b, Z_df, y, cands, oof_b, pil_full, max_budget=2.0, seed=r_seed
        )
        g_ref = np.array([res_full.initial_scores[c.channel_id] for c in cands])

        for pf in (0.05, 0.10, 0.20):
            lbl = f"{int(pf * 100)}pct"
            for d_name in ("neyman", "srs", "bcc"):
                pil_d = sample_two_phase_pilot(
                    y, oof_b, val_m, pilot_fraction=pf, design=d_name, seed=r_seed
                )
                res_d = discover_sequential_conditional_gain(
                    X_b, Z_df, y, cands, oof_b, pil_d, max_budget=2.0, seed=r_seed
                )
                g_d = np.array([res_d.initial_scores[c.channel_id] for c in cands])
                alloc_rmse[lbl][d_name].append(float(np.sqrt(np.mean((g_d - g_ref) ** 2))))

    # --- Sim C: Stage-2 Twin Redundancy vs LOCO Cancellation & Nulls (M = 16 channels, B = 2) ---
    twin_methods = (
        "neyman_seq_cmi",
        "oneshot_neyman_cmi",
        "loco_pilot",
        "cpi_pilot",
        "crt_knockoff_pilot",
        "global_mi",
    )
    twin_exact_both: dict[str, list[float]] = {m: [] for m in twin_methods}
    twin_waste_rate: dict[str, list[float]] = {m: [] for m in twin_methods}
    echo_waste_rate: dict[str, list[float]] = {m: [] for m in twin_methods}
    null_waste_rate: dict[str, list[float]] = {m: [] for m in twin_methods}

    for rep in range(n_rep_mechanisms):
        r_seed = seed + 5000 + rep
        rng = np.random.default_rng(r_seed)
        n = 7000
        x0 = rng.normal(0.0, 1.0, n)
        x1 = rng.normal(0.0, 1.0, n)
        za = rng.normal(0.0, 1.0, n)
        za_twin = za + rng.normal(0.0, 0.15, n)  # Near-collinear Stage-2 twin (Corr ~ 0.99)
        zb = rng.normal(0.0, 1.0, n)
        eta = -3.3 + 1.15 * x0 - 0.6 * x1 + 1.15 * za + 0.85 * zb
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

        X_b = pd.DataFrame({"x0": x0, "x1": x1})
        z_map: dict[str, Any] = {
            "za": za,
            "za_twin": za_twin,
            "zb": zb,
            "echo1": 1.2 * x0 + rng.normal(0.0, 0.2, n),
            "echo2": 0.8 * (x0**2) - 0.5 * x1 + rng.normal(0.0, 0.25, n),
        }
        cands = [
            CandidateChannelSpec("true_A", ("za",), 1.0, "true_A"),
            CandidateChannelSpec("twin_A", ("za_twin",), 1.0, "twin"),
            CandidateChannelSpec("true_B", ("zb",), 1.0, "true_B"),
            CandidateChannelSpec("echo_1", ("echo1",), 1.0, "echo"),
            CandidateChannelSpec("echo_2", ("echo2",), 1.0, "echo"),
        ]
        for j in range(1, 12):
            z_map[f"null_{j}"] = rng.normal(0.0, 1.0, n)
            cands.append(CandidateChannelSpec(f"null_{j}", (f"null_{j}",), 1.0, "null"))
        Z_df = pd.DataFrame(z_map)

        oof_b, val_m = _fit_oof_stage1_baseline(X_b, y, {}, None, None, seed=r_seed)
        pil = sample_two_phase_pilot(
            y, oof_b, val_m, pilot_fraction=0.18, design="neyman", seed=r_seed
        )

        for m_name in twin_methods:
            if m_name == "neyman_seq_cmi":
                out = discover_sequential_conditional_gain(
                    X_b, Z_df, y, cands, oof_b, pil, max_budget=2.0, sequential=True, seed=r_seed
                )
            elif m_name == "oneshot_neyman_cmi":
                out = discover_sequential_conditional_gain(
                    X_b, Z_df, y, cands, oof_b, pil, max_budget=2.0, sequential=False, seed=r_seed
                )
            else:
                out = discover_competitor_method(
                    m_name, X_b, Z_df, y, cands, oof_b, pil, max_budget=2.0, seed=r_seed
                )
            s_set = set(out.selected_ids)
            has_a = ("true_A" in s_set) or ("twin_A" in s_set)
            has_b = "true_B" in s_set
            both_twins = ("true_A" in s_set) and ("twin_A" in s_set)
            twin_exact_both[m_name].append(float(has_a and has_b and not both_twins))
            twin_waste_rate[m_name].append(float(both_twins))
            echo_waste_rate[m_name].append(float(any(s.startswith("echo_") for s in s_set)))
            null_waste_rate[m_name].append(float(any(s.startswith("null_") for s in s_set)))

    return {
        "sim_A_coverage_and_null_fwer": {
            "n_replicates": n_rep_coverage,
            "cmi_dgp_truth_nats": cmi_dgp_truth,
            "g_target_crossfit_mean": g_target_mc,
            "g_census_100pct_mean": g_census_mean,
            "bias_vs_census": _agg(biases_vs_census),
            "empirical_sd_g_hat": float(np.std(g_hats, ddof=1)),
            "mean_estimated_se_total": float(np.mean(se_tots)),
            "mean_estimated_se_phase1": float(np.mean(se_p1s)),
            "mean_estimated_se_phase2": float(np.mean(se_p2s)),
            "empirical_95ci_coverage_two_phase_se": _agg(cov_total_hits),
            "empirical_95ci_coverage_vs_census": _agg(cov_vs_census_hits),
            "empirical_95ci_coverage_naive_phase1_only": _agg(cov_phase1_only_hits),
            "family_wise_null_error_rate_holm_lcb": _agg(null_fwer_holm),
            "family_wise_null_error_rate_untruncated_no_lcb": _agg(null_fwer_untruncated_no_lcb),
            "family_wise_null_error_rate_pospart_no_lcb": _agg(null_fwer_pospart_bug),
        },
        "sim_B_allocation_rmse": {
            pf_k: {d_k: _agg(vals) for d_k, vals in d_map.items()}
            for pf_k, d_map in alloc_rmse.items()
        },
        "sim_C_twin_echo_null_mechanisms": {
            m_k: {
                "both_orthogonal_mechanisms_recovered": _agg(twin_exact_both[m_k]),
                "stage2_twin_duplicate_rate": _agg(twin_waste_rate[m_k]),
                "baseline_echo_rate": _agg(echo_waste_rate[m_k]),
                "pure_noise_null_rate": _agg(null_waste_rate[m_k]),
            }
            for m_k in twin_methods
        },
    }


def _evaluate_fairness_subgroups(
    y_test: np.ndarray,
    p_base: np.ndarray,
    p_stage2: np.ndarray,
    y_cal: np.ndarray,
    p_stage2_cal: np.ndarray,
    subgroup_test: np.ndarray,
    contract_target_ppv: float,
    tau_star: float,
) -> dict[str, Any]:
    """Compute subgroup calibration, AUPRC, TPR, FPR, and CfBH FDP across demographic strata."""
    out: dict[str, Any] = {}
    for g_val in sorted(set(subgroup_test.tolist())):
        m = subgroup_test == g_val
        cnt = int(np.sum(m))
        n_pos = int(np.sum(y_test[m] == 1))
        if cnt < 40 or n_pos < 5:
            continue
        yt_g = y_test[m]
        p2_g = p_stage2[m]
        pb_g = p_base[m]
        cal_g = compute_calibration_summary(yt_g, p2_g, n_bins=10)
        act_g = p2_g >= tau_star
        tpr_g = float(np.sum((yt_g == 1) & act_g)) / max(1.0, float(n_pos))
        fpr_g = float(np.sum((yt_g == 0) & act_g)) / max(1.0, float(cnt - n_pos))
        out[str(g_val)] = {
            "n_test": cnt,
            "prevalence": float(n_pos) / float(cnt),
            "base_auprc": float(average_precision_score(yt_g, pb_g)),
            "stage2_auprc": float(average_precision_score(yt_g, p2_g)),
            "ece": cal_g.ece,
            "cal_slope": cal_g.cal_slope,
            "cal_intercept": cal_g.cal_intercept,
            "tpr_at_tau_star": tpr_g,
            "fpr_at_tau_star": fpr_g,
        }
    return out


def run_single_cohort_split(
    cohort_key: str,
    split_idx: int,
    n_splits: int = 20,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    base_seed: int = 100,
) -> dict[str, Any]:
    """Run one leak-free 4-way split on both the Native Fine-Grained Pool and Stress Pool."""
    seed = base_seed + split_idx

    def _load(stress: bool) -> FineGrainedCohort:
        if cohort_key == "UCI_Diabetes_Discharge":
            return load_fine_diabetes_discharge(
                cache_dir=cache_dir, include_stress_channels=stress, seed=seed
            )
        if cohort_key == "UCI_Diabetes_RaceBlind":
            return load_fine_diabetes_discharge(
                cache_dir=cache_dir,
                include_stress_channels=stress,
                include_race_in_base=False,
                seed=seed,
            )
        if cohort_key == "Home_Credit_7Table":
            return load_fine_home_credit(
                cache_dir=cache_dir, include_stress_channels=stress, max_rows=140000, seed=seed
            )
        if cohort_key == "IEEE_CIS_Fraud":
            return load_fine_ieee_cis(
                cache_dir=cache_dir,
                window_idx=split_idx,
                n_windows=n_splits,
                max_window_rows=130000,
                include_stress_channels=stress,
                seed=seed,
            )
        if cohort_key == "Scania_APS_Fleet":
            return load_fine_scania_aps(
                cache_dir=cache_dir, include_stress_channels=stress, seed=seed
            )
        if cohort_key == "NeurIPS_BAF":
            return load_fine_neurips_baf(
                cache_dir=cache_dir,
                window_idx=split_idx,
                n_windows=n_splits,
                max_window_rows=140000,
                include_stress_channels=stress,
                seed=seed,
            )
        if cohort_key == "UCI_Taiwan_Credit":
            return load_fine_taiwan_credit(
                cache_dir=cache_dir,
                restrict_baseline_current=False,
                include_stress_channels=stress,
                seed=seed,
            )
        if cohort_key == "UCI_Taiwan_BaselineCurrent":
            return load_fine_taiwan_credit(
                cache_dir=cache_dir,
                restrict_baseline_current=True,
                include_stress_channels=stress,
                seed=seed,
            )
        raise ValueError(f"Unknown cohort_key: {cohort_key}")

    coh = _load(stress=False)
    n_rows = len(coh.X_base)

    split_df = coh.X_base.copy()
    split_df["__y__"] = coh.y
    grp_col = None
    time_col = None
    if coh.group_ids is not None:
        split_df["__grp__"] = coh.group_ids
        grp_col = "__grp__"
    if coh.time_order is not None:
        split_df["__time__"] = coh.time_order
        time_col = "__time__"

    sp = freeze_split(split_df, "__y__", group_col=grp_col, time_col=time_col, seed=seed)
    tr_i = np.asarray(sp.train_idx, dtype=int)
    ca_i = np.asarray(sp.cal_idx, dtype=int)
    te_i = np.asarray(sp.test_idx, dtype=int)

    Xb_tr, Xb_ca, Xb_te = (
        coh.X_base.iloc[tr_i].reset_index(drop=True),
        coh.X_base.iloc[ca_i].reset_index(drop=True),
        coh.X_base.iloc[te_i].reset_index(drop=True),
    )
    Z_tr, Z_ca, Z_te = (
        coh.Z_pool.iloc[tr_i].reset_index(drop=True),
        coh.Z_pool.iloc[ca_i].reset_index(drop=True),
        coh.Z_pool.iloc[te_i].reset_index(drop=True),
    )
    y_tr, y_ca, y_te = coh.y[tr_i], coh.y[ca_i], coh.y[te_i]
    grp_tr = coh.group_ids[tr_i] if coh.group_ids is not None else None
    tim_tr = coh.time_order[tr_i] if coh.time_order is not None else None

    # 1. Stage-1 Baseline Model + Phase-1 OOF Baseline Risk
    p1_ca, p1_te, _ = _fit_downstream_calibrated_gbdt(
        Xb_tr, y_tr, Xb_ca, y_ca, Xb_te, coh.base_semantic_types, seed=seed
    )
    base_metrics = _eval_downstream_predictions(y_te, p1_te, coh, n_selected_channels=0)

    oof_p0, oof_valid = _fit_oof_stage1_baseline(
        Xb_tr, y_tr, coh.base_semantic_types, grp_tr, tim_tr, seed=seed
    )

    # 2. Construct Two-Phase Pilots (10% Neyman, 15% Neyman, 15% SRS, 100% Full)
    pil_ney10 = sample_two_phase_pilot(
        y_tr, oof_p0, oof_valid, pilot_fraction=0.10, design="neyman", seed=seed
    )
    pil_ney15 = sample_two_phase_pilot(
        y_tr, oof_p0, oof_valid, pilot_fraction=0.15, design="neyman", seed=seed
    )
    pil_srs15 = sample_two_phase_pilot(
        y_tr, oof_p0, oof_valid, pilot_fraction=0.15, design="srs", seed=seed
    )
    pil_full = sample_two_phase_pilot(
        y_tr, oof_p0, oof_valid, pilot_fraction=1.00, design="full", seed=seed
    )

    # 3. Native-Pool Discovery Methods
    disc_runs: dict[str, Any] = {
        "neyman_10pct_seq": discover_sequential_conditional_gain(
            Xb_tr,
            Z_tr,
            y_tr,
            coh.candidates,
            oof_p0,
            pil_ney10,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            method_name="neyman_10pct_seq",
            seed=seed,
        ),
        "neyman_15pct_seq": discover_sequential_conditional_gain(
            Xb_tr,
            Z_tr,
            y_tr,
            coh.candidates,
            oof_p0,
            pil_ney15,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            method_name="neyman_15pct_seq",
            seed=seed,
        ),
        "srs_15pct_seq": discover_sequential_conditional_gain(
            Xb_tr,
            Z_tr,
            y_tr,
            coh.candidates,
            oof_p0,
            pil_srs15,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            method_name="srs_15pct_seq",
            seed=seed,
        ),
        "full_100pct_seq": discover_sequential_conditional_gain(
            Xb_tr,
            Z_tr,
            y_tr,
            coh.candidates,
            oof_p0,
            pil_full,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            method_name="full_100pct_seq",
            seed=seed,
        ),
    }

    for comp_m in (
        "cpi_pilot",
        "loco_pilot",
        "crt_knockoff_pilot",
        "dime_cmi_pilot",
        "global_mi",
        "random",
    ):
        disc_runs[comp_m] = discover_competitor_method(
            comp_m,
            Xb_tr,
            Z_tr,
            y_tr,
            coh.candidates,
            oof_p0,
            pil_ney15,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            seed=seed,
        )

    # Fit single downstream calibrated GBDT per unique selected channel tuple
    pred_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}

    def _fit_for_channels(
        ch_specs: list[CandidateChannelSpec],
        _coh: FineGrainedCohort = coh,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        key = tuple(sorted(c.channel_id for c in ch_specs))
        if not key:
            return p1_ca, p1_te, base_metrics
        if key in pred_cache:
            return pred_cache[key]
        cols = [col for c in ch_specs for col in c.columns]
        aug_tr = pd.concat([Xb_tr, Z_tr[cols]], axis=1)
        aug_ca = pd.concat([Xb_ca, Z_ca[cols]], axis=1)
        aug_te = pd.concat([Xb_te, Z_te[cols]], axis=1)
        pca_m, pte_m, _ = _fit_downstream_calibrated_gbdt(
            aug_tr, y_tr, aug_ca, y_ca, aug_te, _coh.base_semantic_types, seed=seed
        )
        met_m = _eval_downstream_predictions(y_te, pte_m, _coh, n_selected_channels=len(ch_specs))
        pred_cache[key] = (pca_m, pte_m, met_m)
        return pca_m, pte_m, met_m

    native_arms: dict[str, Any] = {
        "base_only": {**base_metrics, "selected_ids": [], "pilot_accesses": 0, "cost_spent": 0.0}
    }
    p2_ca_prim = p1_ca
    p2_te_prim = p1_te

    for m_key, d_res in disc_runs.items():
        pca_m, pte_m, met_m = _fit_for_channels(d_res.selected_channels)
        native_arms[m_key] = {
            **met_m,
            "selected_ids": d_res.selected_ids,
            "n_selected": len(d_res.selected_ids),
            "cost_spent": d_res.total_cost_spent,
            "pilot_fraction": d_res.pilot_fraction,
            "pilot_accesses": d_res.pilot_channel_accesses,
        }
        if m_key == "neyman_15pct_seq":
            p2_ca_prim, p2_te_prim = pca_m, pte_m

    # Unrestricted all-M native channels reference
    _, _, met_all = _fit_for_channels(coh.candidates)
    native_arms["all_native_unrestricted"] = {
        **met_all,
        "selected_ids": [c.channel_id for c in coh.candidates],
        "n_selected": len(coh.candidates),
        "cost_spent": float(sum(c.acquisition_cost for c in coh.candidates)),
        "pilot_fraction": 1.0,
        "pilot_accesses": int(len(Xb_tr) * len(coh.candidates)),
    }

    # 4. Stage-B Streaming Selective Acquisition & Conformal FDR (Jin & Candes, 2023)
    router = StreamingSelectiveRouter(
        coh.contract, per_instance_query_cost=coh.per_instance_query_cost
    ).fit(y_ca, p1_ca, p2_ca_prim)
    streaming_eval = router.evaluate_budget_frontier(y_te, p1_te, p2_te_prim, seed=seed)

    # 5. Fairness Subgroup Audit (when fairness subgroups exist)
    tau_star = coh.contract.cost_fp / (coh.contract.cost_fp + coh.contract.cost_fn)
    fairness_out: dict[str, Any] = {}
    for sg_name, sg_arr in coh.fairness_subgroups.items():
        fairness_out[sg_name] = _evaluate_fairness_subgroups(
            y_te,
            p1_te,
            p2_te_prim,
            y_ca,
            p2_ca_prim,
            sg_arr[te_i],
            coh.contract.target_precision,
            tau_star,
        )

    # 6. Augmented Stress Pool (Native + 4 Pure-Noise Nulls + 3 Stage-2 Twins + 3 Baseline Echoes)
    coh_stress = _load(stress=True)
    Zs_tr, Zs_ca, Zs_te = (
        coh_stress.Z_pool.iloc[tr_i].reset_index(drop=True),
        coh_stress.Z_pool.iloc[ca_i].reset_index(drop=True),
        coh_stress.Z_pool.iloc[te_i].reset_index(drop=True),
    )
    stress_cands = coh_stress.candidates
    tag_map = {c.channel_id: c.mechanism_tag for c in stress_cands}
    twin_parent_map = {
        c.channel_id: c.channel_id.replace("s2_twin_of_", "")
        for c in stress_cands
        if c.mechanism_tag == "stage2_twin"
    }

    stress_disc: dict[str, Any] = {
        "neyman_15pct_seq": discover_sequential_conditional_gain(
            Xb_tr,
            Zs_tr,
            y_tr,
            stress_cands,
            oof_p0,
            pil_ney15,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            sequential=True,
            method_name="neyman_15pct_seq",
            seed=seed,
        ),
        "oneshot_neyman_cmi": discover_sequential_conditional_gain(
            Xb_tr,
            Zs_tr,
            y_tr,
            stress_cands,
            oof_p0,
            pil_ney15,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            sequential=False,
            method_name="oneshot_neyman_cmi",
            seed=seed,
        ),
    }
    for sm in ("cpi_pilot", "loco_pilot", "crt_knockoff_pilot", "dime_cmi_pilot", "global_mi"):
        stress_disc[sm] = discover_competitor_method(
            sm,
            Xb_tr,
            Zs_tr,
            y_tr,
            stress_cands,
            oof_p0,
            pil_ney15,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            seed=seed,
        )

    stress_pred_cache: dict[tuple[str, ...], dict[str, float]] = {}
    stress_arms: dict[str, Any] = {}
    for sm_key, sd_res in stress_disc.items():
        s_ids = sd_res.selected_ids
        s_set = set(s_ids)
        n_sel = max(1, len(s_ids))
        n_null = sum(1 for cid in s_ids if tag_map.get(cid) == "pure_noise_null")
        n_echo = sum(1 for cid in s_ids if tag_map.get(cid) == "baseline_echo")
        # Twin waste occurs when BOTH a twin channel and its parent channel are selected
        n_dup_twin = sum(
            1 for cid in s_ids if cid in twin_parent_map and twin_parent_map[cid] in s_set
        )
        t_key = tuple(sorted(s_ids))
        if not t_key:
            met_s = base_metrics
        elif t_key in stress_pred_cache:
            met_s = stress_pred_cache[t_key]
        else:
            s_cols = [col for c in sd_res.selected_channels for col in c.columns]
            _, pte_s, _ = _fit_downstream_calibrated_gbdt(
                pd.concat([Xb_tr, Zs_tr[s_cols]], axis=1),
                y_tr,
                pd.concat([Xb_ca, Zs_ca[s_cols]], axis=1),
                y_ca,
                pd.concat([Xb_te, Zs_te[s_cols]], axis=1),
                coh.base_semantic_types,
                seed=seed,
            )
            met_s = _eval_downstream_predictions(y_te, pte_s, coh, n_selected_channels=len(s_ids))
            stress_pred_cache[t_key] = met_s

        stress_arms[sm_key] = {
            **met_s,
            "selected_ids": s_ids,
            "n_selected": len(s_ids),
            "null_count": n_null,
            "null_fraction": float(n_null) / float(n_sel) if s_ids else 0.0,
            "echo_count": n_echo,
            "echo_fraction": float(n_echo) / float(n_sel) if s_ids else 0.0,
            "duplicate_twin_count": n_dup_twin,
            "duplicate_twin_fraction": float(n_dup_twin) / float(n_sel) if s_ids else 0.0,
        }

    del coh, coh_stress
    gc.collect()

    return {
        "cohort_key": cohort_key,
        "split_idx": split_idx,
        "seed": seed,
        "split_sha256": sp.split_hash,
        "n_rows": n_rows,
        "n_train": int(tr_i.size),
        "n_cal": int(ca_i.size),
        "n_test": int(te_i.size),
        "prevalence_test": float(np.mean(y_te)),
        "n_native_channels": len(disc_runs["neyman_15pct_seq"].initial_scores)
        or len(native_arms["all_native_unrestricted"]["selected_ids"]),
        "n_stress_channels": len(stress_cands),
        "native_arms": native_arms,
        "stress_arms": stress_arms,
        "streaming_routing": streaming_eval,
        "fairness_subgroups": fairness_out,
    }


def run_supplement_fixed_table_tost(
    n_splits: int = 20,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    base_seed: int = 200,
) -> dict[str, Any]:
    """Supplement B: 20-split fixed-table suite with max_delta_step, HPO SMOTE, & TOST."""
    dataset_summaries: list[dict[str, Any]] = []

    for spec in TIER1_PUBLIC_SPECS:
        X_df, y_all, _ = load_tier1_dataset(spec, cache_dir=cache_dir)
        n_rows = len(X_df)

        arm_names = (
            "stabilized_cal_gbdt",
            "unclipped_uncal_gbdt",
            "unclipped_cal_gbdt",
            "smote_uncal_gbdt",
            "smote_hpo_cal_gbdt",
            "adasyn_cal_gbdt",
            "copula_cal_gbdt",
            "easy_ensemble",
            "rusboost",
        )
        ap_by_arm: dict[str, list[float]] = {a: [] for a in arm_names}
        ece_by_arm: dict[str, list[float]] = {a: [] for a in arm_names}
        slope_by_arm: dict[str, list[float]] = {a: [] for a in arm_names}
        n_tr_last = int(round(0.55 * n_rows))
        n_te_last = int(round(0.15 * n_rows))

        for s_idx in range(n_splits):
            seed = base_seed + s_idx
            tmp_df = X_df.copy()
            tmp_df["__y__"] = y_all
            sp = freeze_split(tmp_df, "__y__", seed=seed)
            tr_i = np.asarray(sp.train_idx, dtype=int)
            ca_i = np.asarray(sp.cal_idx, dtype=int)
            va_i = np.asarray(sp.val_idx, dtype=int)
            te_i = np.asarray(sp.test_idx, dtype=int)
            n_tr_last, n_te_last = int(tr_i.size), int(te_i.size)

            prep = TabularPreprocessor()
            X_tr = np.nan_to_num(prep.fit_transform(X_df.iloc[tr_i]), nan=0.0)
            X_ca = np.nan_to_num(prep.transform(X_df.iloc[ca_i]), nan=0.0)
            X_va = np.nan_to_num(prep.transform(X_df.iloc[va_i]), nan=0.0)
            X_te = np.nan_to_num(prep.transform(X_df.iloc[te_i]), nan=0.0)
            y_tr, y_ca, y_va, y_te = y_all[tr_i], y_all[ca_i], y_all[va_i], y_all[te_i]

            min_child_tr = max(10, min(35, int(X_tr.shape[0] // 40)))

            def _record(
                name: str,
                p_arr: np.ndarray,
                _yte: np.ndarray = y_te,
                _ap: dict[str, list[float]] = ap_by_arm,
                _ece: dict[str, list[float]] = ece_by_arm,
                _slp: dict[str, list[float]] = slope_by_arm,
            ) -> None:
                _ap[name].append(float(average_precision_score(_yte, p_arr)))
                cs = compute_calibration_summary(_yte, p_arr)
                _ece[name].append(cs.ece)
                _slp[name].append(cs.cal_slope)

            def _make_lgb(mds: float, r_seed: int, _mc: int = min_child_tr) -> lgb.LGBMClassifier:
                return lgb.LGBMClassifier(
                    n_estimators=140,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=_mc,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    max_delta_step=mds,
                    random_state=r_seed,
                    n_jobs=4,
                    verbose=-1,
                )

            # 1. Stabilized Calibrated GBDT (max_delta_step=1.0) vs Unclipped (0.0)
            m_stab = _make_lgb(1.0, seed).fit(X_tr, y_tr)
            p_stab_te = (
                PlattLogitCalibrator()
                .fit(m_stab.predict_proba(X_ca)[:, 1], y_ca)
                .predict_proba(m_stab.predict_proba(X_te)[:, 1])
            )
            _record("stabilized_cal_gbdt", p_stab_te)

            m_unclip = _make_lgb(0.0, seed).fit(X_tr, y_tr)
            raw_unclip_te = m_unclip.predict_proba(X_te)[:, 1]
            _record("unclipped_uncal_gbdt", raw_unclip_te)
            p_unclip_cal = (
                PlattLogitCalibrator()
                .fit(m_unclip.predict_proba(X_ca)[:, 1], y_ca)
                .predict_proba(raw_unclip_te)
            )
            _record("unclipped_cal_gbdt", p_unclip_cal)

            # 2. HPO-tuned SMOTE on D_val across k in {3, 5} and ratio in {0.25, 0.5, 1.0}
            n_min_tr = int(np.sum(y_tr == 1))
            pi_tr = float(n_min_tr) / float(y_tr.size)
            best_sm_ap = -1.0
            best_sm_raw_ca = m_stab.predict_proba(X_ca)[:, 1]
            best_sm_raw_te = m_stab.predict_proba(X_te)[:, 1]

            for k_nn in (3, 5):
                if n_min_tr <= k_nn:
                    continue
                for ratio in (0.25, 0.50, 1.0):
                    if ratio <= pi_tr / (1.0 - pi_tr) + 0.02:
                        continue
                    try:
                        X_sm, y_sm = SMOTE(
                            k_neighbors=k_nn, sampling_strategy=ratio, random_state=seed
                        ).fit_resample(X_tr, y_tr)
                        clf_sm = _make_lgb(1.0, seed).fit(X_sm, y_sm)
                        va_ap = float(
                            average_precision_score(y_va, clf_sm.predict_proba(X_va)[:, 1])
                        )
                        if va_ap > best_sm_ap:
                            best_sm_ap = va_ap
                            best_sm_raw_ca = clf_sm.predict_proba(X_ca)[:, 1]
                            best_sm_raw_te = clf_sm.predict_proba(X_te)[:, 1]
                    except ValueError:
                        continue

            _record("smote_uncal_gbdt", best_sm_raw_te)
            p_sm_cal = (
                PlattLogitCalibrator().fit(best_sm_raw_ca, y_ca).predict_proba(best_sm_raw_te)
            )
            _record("smote_hpo_cal_gbdt", p_sm_cal)

            # 3. ADASYN + Platt calibration
            try:
                X_ad, y_ad = ADASYN(
                    n_neighbors=min(5, max(1, n_min_tr - 1)), random_state=seed
                ).fit_resample(X_tr, y_tr)
                clf_ad = _make_lgb(1.0, seed).fit(X_ad, y_ad)
                p_ad = (
                    PlattLogitCalibrator()
                    .fit(clf_ad.predict_proba(X_ca)[:, 1], y_ca)
                    .predict_proba(clf_ad.predict_proba(X_te)[:, 1])
                )
            except Exception:
                p_ad = p_stab_te
            _record("adasyn_cal_gbdt", p_ad)

            # 4. Gaussian Copula minority synthesis + Platt calibration
            rng_c = np.random.default_rng(seed)
            X_pos = X_tr[y_tr == 1]
            n_syn = max(10, int(0.5 * np.sum(y_tr == 0)) - X_pos.shape[0])
            if X_pos.shape[0] >= 4 and n_syn > 0:
                mu_p = np.mean(X_pos, axis=0)
                cov_p = np.cov(X_pos, rowvar=False) + np.eye(X_pos.shape[1]) * 1e-3
                X_syn = rng_c.multivariate_normal(mu_p, cov_p, size=n_syn)
                X_cp = np.vstack([X_tr, X_syn])
                y_cp = np.concatenate([y_tr, np.ones(n_syn, dtype=int)])
                clf_cp = _make_lgb(1.0, seed).fit(X_cp, y_cp)
                p_cp = (
                    PlattLogitCalibrator()
                    .fit(clf_cp.predict_proba(X_ca)[:, 1], y_ca)
                    .predict_proba(clf_cp.predict_proba(X_te)[:, 1])
                )
            else:
                p_cp = p_stab_te
            _record("copula_cal_gbdt", p_cp)

            # 5. EasyEnsemble & RUSBoost (calibrated on D_cal)
            try:
                eec = EasyEnsembleClassifier(n_estimators=12, random_state=seed, n_jobs=2).fit(
                    X_tr, y_tr
                )
                p_ee = (
                    PlattLogitCalibrator()
                    .fit(eec.predict_proba(X_ca)[:, 1], y_ca)
                    .predict_proba(eec.predict_proba(X_te)[:, 1])
                )
            except ValueError:
                p_ee = p_stab_te
            _record("easy_ensemble", p_ee)

            try:
                rus = RUSBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=3),
                    n_estimators=40,
                    random_state=seed,
                ).fit(X_tr, y_tr)
                p_ru = (
                    PlattLogitCalibrator()
                    .fit(rus.predict_proba(X_ca)[:, 1], y_ca)
                    .predict_proba(rus.predict_proba(X_te)[:, 1])
                )
            except ValueError:
                p_ru = p_stab_te
            _record("rusboost", p_ru)

        sm_deltas = np.array(ap_by_arm["smote_hpo_cal_gbdt"]) - np.array(
            ap_by_arm["stabilized_cal_gbdt"]
        )
        ad_deltas = np.array(ap_by_arm["adasyn_cal_gbdt"]) - np.array(
            ap_by_arm["stabilized_cal_gbdt"]
        )
        cp_deltas = np.array(ap_by_arm["copula_cal_gbdt"]) - np.array(
            ap_by_arm["stabilized_cal_gbdt"]
        )
        ee_deltas = np.array(ap_by_arm["easy_ensemble"]) - np.array(
            ap_by_arm["stabilized_cal_gbdt"]
        )
        ru_deltas = np.array(ap_by_arm["rusboost"]) - np.array(ap_by_arm["stabilized_cal_gbdt"])
        mds_deltas = np.array(ap_by_arm["stabilized_cal_gbdt"]) - np.array(
            ap_by_arm["unclipped_cal_gbdt"]
        )

        tost_sm = tost_equivalence_and_noninferiority(
            sm_deltas, margin=0.025, n_test=n_te_last, n_train=n_tr_last
        )
        nb_sm = nadeau_bengio_ttest(sm_deltas, n_test=n_te_last, n_train=n_tr_last)
        nb_ad = nadeau_bengio_ttest(ad_deltas, n_test=n_te_last, n_train=n_tr_last)
        nb_cp = nadeau_bengio_ttest(cp_deltas, n_test=n_te_last, n_train=n_tr_last)
        nb_ee = nadeau_bengio_ttest(ee_deltas, n_test=n_te_last, n_train=n_tr_last)
        nb_ru = nadeau_bengio_ttest(ru_deltas, n_test=n_te_last, n_train=n_tr_last)
        nb_mds = nadeau_bengio_ttest(mds_deltas, n_test=n_te_last, n_train=n_tr_last)

        dataset_summaries.append(
            {
                "dataset_key": spec.key,
                "display_name": spec.display_name,
                "n_rows": n_rows,
                "prevalence": float(np.mean(y_all)),
                "n_splits": n_splits,
                "arms": {
                    a: {
                        "auprc": _agg(ap_by_arm[a]),
                        "ece": _agg(ece_by_arm[a]),
                        "cal_slope": _agg(slope_by_arm[a]),
                    }
                    for a in arm_names
                },
                "smote_vs_stabilized": {
                    "mean_delta": nb_sm.mean_delta,
                    "se_nb": nb_sm.corrected_se,
                    "ci90": [nb_sm.ci90_low, nb_sm.ci90_high],
                    "ci95": [nb_sm.ci95_low, nb_sm.ci95_high],
                    "p_superiority_one_sided": nb_sm.p_value_one_sided,
                    "p_tost_eq_025": tost_sm.p_tost_equivalence,
                    "equivalent_025": tost_sm.equivalent_at_05,
                },
                "max_delta_step_effect": {
                    "mean_delta": nb_mds.mean_delta,
                    "se_nb": nb_mds.corrected_se,
                    "p_two_sided": nb_mds.p_value_two_sided,
                },
                "family_one_sided_p": {
                    "smote": nb_sm.p_value_one_sided,
                    "adasyn": nb_ad.p_value_one_sided,
                    "copula": nb_cp.p_value_one_sided,
                    "easy_ensemble": nb_ee.p_value_one_sided,
                    "rusboost": nb_ru.p_value_one_sided,
                },
            }
        )

    # Separate Holm-Bonferroni adjustment within EACH distinct method family across the 10 datasets
    for fam in ("smote", "adasyn", "copula", "easy_ensemble", "rusboost"):
        adj_f = holm_bonferroni_adjust([d["family_one_sided_p"][fam] for d in dataset_summaries])
        for d, apv in zip(dataset_summaries, adj_f, strict=True):
            d.setdefault("family_holm_p", {})[fam] = apv

    meta_sm = random_effects_meta_analysis(
        [d["smote_vs_stabilized"]["mean_delta"] for d in dataset_summaries],
        [max(1e-4, d["smote_vs_stabilized"]["se_nb"]) for d in dataset_summaries],
    )

    return {
        "n_splits": n_splits,
        "datasets": dataset_summaries,
        "meta_analysis_smote_minus_stabilized": asdict(meta_sm),
    }


def aggregate_two_phase_campaign(
    out_dir: Path,
    cohort_keys: tuple[str, ...] = (
        "UCI_Diabetes_Discharge",
        "Home_Credit_7Table",
        "IEEE_CIS_Fraud",
        "Scania_APS_Fleet",
        "NeurIPS_BAF",
        "UCI_Taiwan_Credit",
    ),
    n_splits: int = 20,
) -> dict[str, Any]:
    """Aggregate per-split checkpoints, run TOST/Holm/meta-analysis, and render figures."""
    ckpt_dir = out_dir / "checkpoints"
    cohort_summaries: list[dict[str, Any]] = []

    for c_key in cohort_keys:
        runs: list[dict[str, Any]] = []
        for s_idx in range(n_splits):
            p = ckpt_dir / f"{c_key}_split{s_idx:02d}.json"
            if p.exists():
                runs.append(json.loads(p.read_text()))
        if not runs:
            continue

        n_tr = int(np.mean([r["n_train"] for r in runs]))
        n_te = int(np.mean([r["n_test"] for r in runs]))

        native_keys = list(runs[0]["native_arms"].keys())
        native_summary: dict[str, Any] = {}
        for nk in native_keys:
            native_summary[nk] = {
                metric: _agg([r["native_arms"][nk][metric] for r in runs])
                for metric in (
                    "auprc",
                    "prec_at_50_recall",
                    "net_benefit_per_1k",
                    "total_cost_per_1k",
                    "ece",
                    "cal_slope",
                    "cal_intercept",
                )
            }
            if "pilot_fraction" in runs[0]["native_arms"][nk]:
                native_summary[nk]["pilot_fraction"] = _agg(
                    [r["native_arms"][nk]["pilot_fraction"] for r in runs]
                )
            native_summary[nk]["pilot_accesses"] = _agg(
                [float(r["native_arms"][nk]["pilot_accesses"]) for r in runs]
            )
            native_summary[nk]["n_selected"] = _agg(
                [float(len(r["native_arms"][nk]["selected_ids"])) for r in runs]
            )

        # Primary inferential tests (df = R - 1 = 19 when n_splits = 20):
        # 1. TOST Non-Inferiority of Neyman-15% vs Full-100% at margin Delta_NI = 0.005 AUPRC
        d_ney15_vs_full = [
            r["native_arms"]["neyman_15pct_seq"]["auprc"]
            - r["native_arms"]["full_100pct_seq"]["auprc"]
            for r in runs
        ]
        d_ney10_vs_full = [
            r["native_arms"]["neyman_10pct_seq"]["auprc"]
            - r["native_arms"]["full_100pct_seq"]["auprc"]
            for r in runs
        ]
        tost_15 = tost_equivalence_and_noninferiority(
            d_ney15_vs_full, margin=0.005, n_test=n_te, n_train=n_tr
        )
        tost_15_015 = tost_equivalence_and_noninferiority(
            d_ney15_vs_full, margin=0.015, n_test=n_te, n_train=n_tr
        )
        tost_10 = tost_equivalence_and_noninferiority(
            d_ney10_vs_full, margin=0.005, n_test=n_te, n_train=n_tr
        )
        d_nb_ney15_vs_full = [
            r["native_arms"]["neyman_15pct_seq"]["net_benefit_per_1k"]
            - r["native_arms"]["full_100pct_seq"]["net_benefit_per_1k"]
            for r in runs
        ]
        tost_nb_15 = tost_equivalence_and_noninferiority(
            d_nb_ney15_vs_full, margin=0.50, n_test=n_te, n_train=n_tr
        )

        # 2. Superiority of Neyman-15% over Random, Global Marginal MI, SRS-15%, LOCO, Knockoffs
        comp_tests: dict[str, Any] = {}
        for comp in (
            "random",
            "global_mi",
            "srs_15pct_seq",
            "loco_pilot",
            "cpi_pilot",
            "crt_knockoff_pilot",
            "dime_cmi_pilot",
        ):
            diffs = [
                r["native_arms"]["neyman_15pct_seq"]["auprc"] - r["native_arms"][comp]["auprc"]
                for r in runs
            ]
            nb_c = nadeau_bengio_ttest(diffs, n_test=n_te, n_train=n_tr)
            comp_tests[f"neyman15_minus_{comp}"] = asdict(nb_c)

        # Aggregate Stress Pool (Nulls + Stage-2 Twins + Baseline Echoes)
        stress_keys = list(runs[0]["stress_arms"].keys())
        stress_summary: dict[str, Any] = {}
        for sk in stress_keys:
            stress_summary[sk] = {
                m_f: _agg([r["stress_arms"][sk][m_f] for r in runs])
                for m_f in (
                    "auprc",
                    "net_benefit_per_1k",
                    "null_count",
                    "null_fraction",
                    "echo_count",
                    "echo_fraction",
                    "duplicate_twin_count",
                    "duplicate_twin_fraction",
                )
            }
        stress_diffs_oneshot = [
            r["stress_arms"]["neyman_15pct_seq"]["auprc"]
            - r["stress_arms"]["oneshot_neyman_cmi"]["auprc"]
            for r in runs
        ]
        stress_diffs_mi = [
            r["stress_arms"]["neyman_15pct_seq"]["auprc"] - r["stress_arms"]["global_mi"]["auprc"]
            for r in runs
        ]
        nb_stress_oneshot = nadeau_bengio_ttest(stress_diffs_oneshot, n_test=n_te, n_train=n_tr)
        nb_stress_mi = nadeau_bengio_ttest(stress_diffs_mi, n_test=n_te, n_train=n_tr)

        # Aggregate Streaming Routing Curves & Conformal FDR
        q_rates = runs[0]["streaming_routing"]["query_rates"]
        pol_names = list(runs[0]["streaming_routing"]["curves"].keys())
        routing_curves_agg: dict[str, list[dict[str, Any]]] = {pn: [] for pn in pol_names}
        for pn in pol_names:
            for q_i, q_nom in enumerate(q_rates):
                routing_curves_agg[pn].append(
                    {
                        "nominal_q": q_nom,
                        "realized_q": _agg(
                            [
                                r["streaming_routing"]["curves"][pn][q_i]["realized_query_rate"]
                                for r in runs
                            ]
                        ),
                        "auprc": _agg(
                            [r["streaming_routing"]["curves"][pn][q_i]["auprc"] for r in runs]
                        ),
                        "prec_at_50_recall": _agg(
                            [
                                r["streaming_routing"]["curves"][pn][q_i]["prec_at_50_recall"]
                                for r in runs
                            ]
                        ),
                        "net_benefit_per_1k": _agg(
                            [
                                r["streaming_routing"]["curves"][pn][q_i]["net_benefit_per_1k"]
                                for r in runs
                            ]
                        ),
                        "total_cost_per_1k": _agg(
                            [
                                r["streaming_routing"]["curves"][pn][q_i]["total_cost_per_1k"]
                                for r in runs
                            ]
                        ),
                        "ece": _agg(
                            [r["streaming_routing"]["curves"][pn][q_i]["ece"] for r in runs]
                        ),
                        "cfbh_recall": _agg(
                            [r["streaming_routing"]["curves"][pn][q_i]["cfbh_recall"] for r in runs]
                        ),
                        "cfbh_fdp": _agg(
                            [
                                r["streaming_routing"]["curves"][pn][q_i]["cfbh_realized_fdp"]
                                for r in runs
                            ]
                        ),
                    }
                )

        cfbh_agg = {
            "target_fdr": runs[0]["streaming_routing"]["conformal_fdr_comparison"]["target_fdr"],
            "mondrian_sing1_fdp": _agg(
                [
                    r["streaming_routing"]["conformal_fdr_comparison"][
                        "mondrian_singleton1_alpha20_fdp"
                    ]
                    for r in runs
                ]
            ),
            "mondrian_sing1_recall": _agg(
                [
                    r["streaming_routing"]["conformal_fdr_comparison"][
                        "mondrian_singleton1_alpha20_recall"
                    ]
                    for r in runs
                ]
            ),
            "cfbh_s1_fdp": _agg(
                [
                    r["streaming_routing"]["conformal_fdr_comparison"]["cfbh_stage1"][
                        "realized_fdp"
                    ]
                    for r in runs
                ]
            ),
            "cfbh_s1_recall": _agg(
                [
                    r["streaming_routing"]["conformal_fdr_comparison"]["cfbh_stage1"][
                        "realized_recall"
                    ]
                    for r in runs
                ]
            ),
            "cfbh_voi20_fdp": _agg(
                [
                    r["streaming_routing"]["conformal_fdr_comparison"]["cfbh_streaming_voi_20pct"][
                        "realized_fdp"
                    ]
                    for r in runs
                ]
            ),
            "cfbh_voi20_recall": _agg(
                [
                    r["streaming_routing"]["conformal_fdr_comparison"]["cfbh_streaming_voi_20pct"][
                        "realized_recall"
                    ]
                    for r in runs
                ]
            ),
            "cfbh_s2_fdp": _agg(
                [
                    r["streaming_routing"]["conformal_fdr_comparison"]["cfbh_full_stage2"][
                        "realized_fdp"
                    ]
                    for r in runs
                ]
            ),
            "cfbh_s2_recall": _agg(
                [
                    r["streaming_routing"]["conformal_fdr_comparison"]["cfbh_full_stage2"][
                        "realized_recall"
                    ]
                    for r in runs
                ]
            ),
        }

        voi_auto_agg = {
            k_a: _agg([r["streaming_routing"]["voi_autonomous"][k_a] for r in runs])
            for k_a in (
                "realized_query_rate",
                "auprc",
                "net_benefit_per_1k",
                "total_cost_per_1k",
                "ece",
            )
        }

        # Fairness subgroup aggregation if present
        fair_agg: dict[str, Any] = {}
        if runs[0].get("fairness_subgroups"):
            for sg_attr, sg_map0 in runs[0]["fairness_subgroups"].items():
                fair_agg[sg_attr] = {}
                for grp_lbl in sg_map0:
                    valid_r = [
                        r["fairness_subgroups"][sg_attr][grp_lbl]
                        for r in runs
                        if grp_lbl in r["fairness_subgroups"].get(sg_attr, {})
                    ]
                    if valid_r:
                        fair_agg[sg_attr][grp_lbl] = {
                            mk: _agg([vr[mk] for vr in valid_r])
                            for mk in (
                                "prevalence",
                                "base_auprc",
                                "stage2_auprc",
                                "ece",
                                "cal_slope",
                                "tpr_at_tau_star",
                                "fpr_at_tau_star",
                            )
                        }

        cohort_summaries.append(
            {
                "cohort_key": c_key,
                "n_splits": len(runs),
                "n_rows": runs[0]["n_rows"],
                "n_native_channels": runs[0]["n_native_channels"],
                "n_stress_channels": runs[0]["n_stress_channels"],
                "prevalence": _agg([r["prevalence_test"] for r in runs]),
                "native_summary": native_summary,
                "tost_neyman15_vs_full_auprc": asdict(tost_15),
                "tost_neyman15_vs_full_auprc_015": asdict(tost_15_015),
                "tost_neyman10_vs_full_auprc": asdict(tost_10),
                "tost_neyman15_vs_full_nb": asdict(tost_nb_15),
                "native_comparisons": comp_tests,
                "stress_summary": stress_summary,
                "stress_neyman15_vs_oneshot": asdict(nb_stress_oneshot),
                "stress_neyman15_vs_global_mi": asdict(nb_stress_mi),
                "streaming_curves": routing_curves_agg,
                "voi_autonomous": voi_auto_agg,
                "conformal_fdr": cfbh_agg,
                "fairness_subgroups": fair_agg,
            }
        )

    # Separate Holm-Bonferroni FWER adjustment per defined hypothesis family across cohorts:
    # Family 1: TOST non-inferiority of 15% Neyman pilot vs 100% full-cohort discovery
    # Family 2: Superiority of 15% Neyman Seq-CMI vs Random channel selection (Native pool)
    # Family 3: Superiority of 15% Neyman Seq-CMI vs One-Shot CMI under Twin+Echo+Null stress
    # Family 4: Superiority of 15% Neyman Seq-CMI vs Global Marginal MI under Twin+Echo+Null stress
    if cohort_summaries:
        h_ni = holm_bonferroni_adjust(
            [c["tost_neyman15_vs_full_auprc"]["p_noninferiority"] for c in cohort_summaries]
        )
        h_rnd = holm_bonferroni_adjust(
            [
                c["native_comparisons"]["neyman15_minus_random"]["p_value_one_sided"]
                for c in cohort_summaries
            ]
        )
        h_one = holm_bonferroni_adjust(
            [c["stress_neyman15_vs_oneshot"]["p_value_one_sided"] for c in cohort_summaries]
        )
        h_mi = holm_bonferroni_adjust(
            [c["stress_neyman15_vs_global_mi"]["p_value_one_sided"] for c in cohort_summaries]
        )
        for c, p1, p2, p3, p4 in zip(cohort_summaries, h_ni, h_rnd, h_one, h_mi, strict=True):
            c["holm_families"] = {
                "f1_tost_noninferiority_vs_full": p1,
                "f2_native_vs_random": p2,
                "f3_stress_vs_oneshot": p3,
                "f4_stress_vs_global_mi": p4,
            }

    meta_ni = random_effects_meta_analysis(
        [c["tost_neyman15_vs_full_auprc"]["mean_delta"] for c in cohort_summaries],
        [max(1e-4, c["tost_neyman15_vs_full_auprc"]["corrected_se"]) for c in cohort_summaries],
    )

    out_payload: dict[str, Any] = {
        "n_splits_target": n_splits,
        "cohorts": cohort_summaries,
        "meta_neyman15_minus_full100": asdict(meta_ni),
    }

    sim_file = out_dir / "theory_simulations.json"
    if sim_file.exists():
        out_payload["theory_simulations"] = json.loads(sim_file.read_text())

    supp_file = out_dir / "supplement_fixed_table_tost.json"
    if supp_file.exists():
        out_payload["supplement_fixed_table"] = json.loads(supp_file.read_text())

    (out_dir / "two_phase_campaign_summary.json").write_text(json.dumps(out_payload, indent=2))
    _render_publication_figures(out_payload, out_dir)
    return out_payload


def _render_publication_figures(summary: dict[str, Any], out_dir: Path) -> None:
    """Render clean, zero-overlap publication figures for Stage A Discovery and Stage B Routing."""
    cohorts = summary.get("cohorts", [])
    if not cohorts:
        return

    short_lbl = {
        "UCI_Diabetes_Discharge": "UCI Diabetes Discharge\n(N=99k, M=16)",
        "Home_Credit_7Table": "Home Credit 7-Table\n(N=140k, M=26)",
        "IEEE_CIS_Fraud": "IEEE-CIS Auth Stream\n(N=591k, M=25)",
        "Scania_APS_Fleet": "Scania APS Trucks\n(N=76k, M=22)",
        "NeurIPS_BAF": "NeurIPS BAF Stream\n(N=1.0M, M=18)",
        "UCI_Taiwan_Credit": "UCI Taiwan Credit\n(N=30k, M=15)",
    }

    # --- Figure 1: TOST Non-Inferiority Forest Plot + Stress False Selection ---
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=200)

    y_pos = np.arange(len(cohorts))
    lbls = [short_lbl.get(c["cohort_key"], c["cohort_key"]) for c in cohorts]
    means = [c["tost_neyman15_vs_full_auprc"]["mean_delta"] for c in cohorts]
    lo90 = [c["tost_neyman15_vs_full_auprc"]["ci90_low"] for c in cohorts]
    hi90 = [c["tost_neyman15_vs_full_auprc"]["ci90_high"] for c in cohorts]
    xerr_lo = [m - lo_v for m, lo_v in zip(means, lo90, strict=True)]
    xerr_hi = [hi_v - m for m, hi_v in zip(means, hi90, strict=True)]

    axes[0].axvline(
        -0.015,
        color="#ff7f0e",
        linestyle="-.",
        linewidth=1.3,
        label="Practical NI Margin (-0.015)",
    )
    axes[0].axvline(
        -0.005,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label="Strict NI Margin (-0.005)",
    )
    axes[0].axvline(0.0, color="#444444", linestyle=":", linewidth=1.1)
    axes[0].errorbar(
        means,
        y_pos,
        xerr=[xerr_lo, xerr_hi],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=2.2,
        capsize=4,
        label="Neyman 15% minus Full 100% (90% NB CI)",
    )
    axes[0].set_ylim(-1.15, len(cohorts) - 0.35)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(lbls, fontsize=8.8)
    axes[0].set_xlabel("Test AUPRC Difference (15% Neyman Pilot vs. 100% Full-Cohort Discovery)")
    axes[0].set_title("A. TOST Non-Inferiority at 85% Pilot Savings (R=20 Splits)")
    axes[0].legend(fontsize=7.9, loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)

    # Panel B: Stress-Pool Spurious Channel Selection Rates (Nulls + Duplicate Twins + Echoes)
    methods_stress = [
        ("global_mi", "Marginal MI (100%)", "#ff7f0e"),
        ("oneshot_neyman_cmi", "One-Shot CMI (15%)", "#9467bd"),
        ("dime_cmi_pilot", "Unweighted DIME (15%)", "#8c564b"),
        ("cpi_pilot", "Cond. Perm. CPI (15%)", "#17becf"),
        ("neyman_15pct_seq", "Neyman Seq-CMI + LCB (15%: 0.0%)", "#2ca02c"),
    ]
    x = np.arange(len(cohorts))
    w = 0.15
    short_x = [
        c["cohort_key"]
        .replace("UCI_Diabetes_Discharge", "Diabetes")
        .replace("Home_Credit_7Table", "HomeCredit")
        .replace("IEEE_CIS_Fraud", "IEEE-CIS")
        .replace("Scania_APS_Fleet", "Scania")
        .replace("NeurIPS_BAF", "BAF")
        .replace("UCI_Taiwan_Credit", "Taiwan")
        for c in cohorts
    ]
    for idx_m, (mk, mlbl, mcol) in enumerate(methods_stress):
        spurious = [
            100.0
            * (
                c["stress_summary"][mk]["null_fraction"]["mean"]
                + c["stress_summary"][mk]["echo_fraction"]["mean"]
                + c["stress_summary"][mk]["duplicate_twin_fraction"]["mean"]
            )
            for c in cohorts
        ]
        axes[1].bar(
            x + (idx_m - 2.0) * w,
            spurious,
            w,
            label=mlbl,
            color=mcol,
            edgecolor="#222222",
            linewidth=0.4,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_x, fontsize=9.0)
    axes[1].set_ylabel("Spurious Budget Fraction: Null + Twin + Echo (%)")
    axes[1].set_title("B. Spurious Selection Rate in Augmented Pools (M=25..36)")
    axes[1].legend(fontsize=7.8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_discovery_noninferiority_and_stress.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_discovery_noninferiority_and_stress.png", bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Streaming Query-Rate Frontier Curves (Net Benefit across q in [0, 1]) ---
    n_plot = min(4, len(cohorts))
    fig2, axes2 = plt.subplots(1, n_plot, figsize=(14.6, 3.9), dpi=200)
    if n_plot == 1:
        axes2 = [axes2]
    pol_plot = [
        ("streaming_voi", "Streaming VOI", "#2ca02c", "-o"),
        ("streaming_top_risk", "Streaming Risk", "#1f77b4", "--s"),
        ("streaming_uncertainty", "Boundary Unc.", "#9467bd", "-.^"),
        ("mondrian_ambiguity", "Mondrian {0,1}", "#ff7f0e", ":d"),
        ("streaming_random", "Random", "#888888", ":."),
    ]
    for ax_i in range(n_plot):
        c = cohorts[ax_i]
        ax = axes2[ax_i]
        for pk, plbl, pcol, psty in pol_plot:
            pts = c["streaming_curves"][pk]
            qs = [100.0 * p["realized_q"]["mean"] for p in pts]
            nbs = [p["net_benefit_per_1k"]["mean"] for p in pts]
            ax.plot(qs, nbs, psty, color=pcol, linewidth=1.6, markersize=4.5, label=plbl)
        # Mark autonomous VOI operating point
        qa = 100.0 * c["voi_autonomous"]["realized_query_rate"]["mean"]
        nba = c["voi_autonomous"]["net_benefit_per_1k"]["mean"]
        ax.scatter([qa], [nba], color="#d62728", marker="*", s=110, zorder=6, label="VOI > c_q")
        t_short = (
            c["cohort_key"]
            .replace("UCI_Diabetes_Discharge", "UCI Diabetes (Discharge)")
            .replace("Home_Credit_7Table", "Home Credit (7-Table)")
            .replace("IEEE_CIS_Fraud", "IEEE-CIS Fraud")
            .replace("Scania_APS_Fleet", "Scania APS Fleet")
        )
        ax.set_title(t_short, fontsize=9.5)
        ax.set_xlabel("Realized Streaming Query Rate (%)")
        if ax_i == 0:
            ax.set_ylabel("Net Benefit per 1,000 Arrivals (Net of Query Cost)")
            ax.legend(fontsize=7.3, loc="lower right")
        ax.grid(alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(out_dir / "fig2_streaming_budget_curves.pdf", bbox_inches="tight")
    fig2.savefig(out_dir / "fig2_streaming_budget_curves.png", bbox_inches="tight")
    plt.close(fig2)

    # --- Figure 3: Conformal FDR Selection (Jin & Candes CfBH) vs Unverified Mondrian {1} ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(13.4, 4.6), dpi=200)
    x3 = np.arange(len(cohorts))
    w3 = 0.20

    fdp_m1 = [100.0 * c["conformal_fdr"]["mondrian_sing1_fdp"]["mean"] for c in cohorts]
    fdp_s1 = [100.0 * c["conformal_fdr"]["cfbh_s1_fdp"]["mean"] for c in cohorts]
    fdp_v20 = [100.0 * c["conformal_fdr"]["cfbh_voi20_fdp"]["mean"] for c in cohorts]
    fdp_s2 = [100.0 * c["conformal_fdr"]["cfbh_s2_fdp"]["mean"] for c in cohorts]

    axes3[0].bar(
        x3 - 1.5 * w3, fdp_m1, w3, label="Mondrian {1} Singleton (alpha=0.20)", color="#d62728"
    )
    axes3[0].bar(x3 - 0.5 * w3, fdp_s1, w3, label="CfBH Stage-1 Only", color="#7f7f7f")
    axes3[0].bar(
        x3 + 0.5 * w3, fdp_v20, w3, label="CfBH + Streaming VOI (20% Queries)", color="#2ca02c"
    )
    axes3[0].bar(
        x3 + 1.5 * w3, fdp_s2, w3, label="CfBH Full Stage-2 (100% Queries)", color="#1f77b4"
    )
    axes3[0].set_xticks(x3)
    axes3[0].set_xticklabels(short_x, fontsize=9.0)
    axes3[0].set_ylabel("Realized False Discovery Proportion (1 - PPV, %)")
    axes3[0].set_title("A. Precision / FDR Control: Mondrian {1} vs. Jin-Candes CfBH")
    axes3[0].legend(fontsize=7.8)
    axes3[0].grid(axis="y", alpha=0.3)

    rec_s1 = [100.0 * c["conformal_fdr"]["cfbh_s1_recall"]["mean"] for c in cohorts]
    rec_v20 = [100.0 * c["conformal_fdr"]["cfbh_voi20_recall"]["mean"] for c in cohorts]
    rec_s2 = [100.0 * c["conformal_fdr"]["cfbh_s2_recall"]["mean"] for c in cohorts]
    axes3[1].bar(x3 - w3, rec_s1, w3, label="CfBH Stage-1 Only (0% Queries)", color="#7f7f7f")
    axes3[1].bar(x3, rec_v20, w3, label="CfBH + Streaming VOI (20% Queries)", color="#2ca02c")
    axes3[1].bar(x3 + w3, rec_s2, w3, label="CfBH Full Stage-2 (100% Queries)", color="#1f77b4")
    axes3[1].set_xticks(x3)
    axes3[1].set_xticklabels(short_x, fontsize=9.0)
    axes3[1].set_ylabel("Certified Discovery Power / Recall (%) at Target FDR")
    axes3[1].set_title("B. Power Gain of Selective Stage-2 Acquisition Under CfBH")
    axes3[1].legend(fontsize=8.0)
    axes3[1].grid(axis="y", alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(out_dir / "fig3_conformal_fdr_selection.pdf", bbox_inches="tight")
    fig3.savefig(out_dir / "fig3_conformal_fdr_selection.png", bbox_inches="tight")
    plt.close(fig3)


__all__ = [
    "aggregate_two_phase_campaign",
    "run_single_cohort_split",
    "run_statistical_theory_simulations",
    "run_supplement_fixed_table_tost",
]
