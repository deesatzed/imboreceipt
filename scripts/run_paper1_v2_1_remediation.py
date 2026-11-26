"""Comprehensive Paper 1 (imboreceipt v2.1) Scientific Remediation & Verification Suite.

Closes all 8 external-critic audit items (P1-1 through P1-8) plus code-vs-theory checks:
  Block 1 (Monte Carlo Theory Suite with Clopper-Pearson 95% CIs):
    - Sim A (1,000 replicates, N=6,000, rho=15%): Empirical verification of Assumption 1
      (capacity-matched cross-fitted null non-positivity E[psi_null] <= 0 across 8,000 Step-1
      and Step-2 null evaluations), sequential multi-step FWER & pure-null FWER for 4-fold CV,
      50/50 Split-Pilot, and Breslow-Cain Offset Pilot GBDTs.
    - Sim B (200 replicates, N=8,000): Gain RMSE across rho in {5%, 10%, 20%} for SRS, BCC,
      Constrained Neyman (HT GBDT), and Constrained Neyman (Breslow-Cain Offset GBDT).
    - Sim C (200 replicates, N=7,000, M=16): Twin + Echo + Null mechanism recovery comparing
      Marginal MI, One-Shot CMI, LOCO, CRT/Knockoffs, Baseline-Stratified Perm., Chamma
      Residualized CPI (showing twin cancellation), and Sequential Forward Neyman + Holm LCB
      (with and without lazy Step-1 screen).
  Block 2 (Leak-Free Stress-Pool Verification, Clean 2x2 Ablations, & 120-Split Channel Ledger):
    - Leak-free train_idx-fitted stress pools across all 6 cohorts (verifying 0/0/0% spurious
      selection and lazy-screen invariance).
    - Clean 2x2 Stage-A ablation (HT Weights vs Unweighted vs Breslow-Cain Offset x Holm LCB
      vs Unstopped G > 0) on UCI Diabetes and Scania APS.
    - Selected channel frequency & TOST summary extracted across all 120 cached splits.
  Block 3 (Full Canonical Non-Overlapping Holdouts, B=10,000 Cluster/Block Bootstraps,
           Covert-AFA, Gauss-Hermite vs Empirical Quantile VOI, & CfBH Shift Audit):
    - Evaluates all 6 cohorts at full unsampled scale (UCI Diabetes 99,343 patient-disjoint;
      Home Credit full 307,511 borrowers; IEEE-CIS full 590,540 out-of-time + card1 cluster
      bootstrap + 4 strictly disjoint eras; Scania APS 76,000; NeurIPS BAF full 1,000,000
      months 0-4 -> 5 -> 6 -> 7 out-of-time + 4 strictly disjoint eras; UCI Taiwan 30,000).
    - Computes B=10,000 paired Davison-Hinkley bootstraps, sNB, MCIDs, AUPRC_full vs
      AUPRC_routed, CITL/Slope/Smooth-ICI, CovertGreedyCMIAFA, and temporal CfBH drift +
      local burn-in recalibration.
  Block 4 (Clinical Case Study Deep Audit on UCI Diabetes 130-US):
    - R=20 patient-grouped development-split Deployment Estimator Triad (Primary Single-Draw
      Binary Bagged VOI, Symmetric Split-Averaged Policy, and Per-Split Mean +- SE) on locked
      patient-disjoint test cohort (B=10,000 patient-cluster bootstraps).
    - Transitional-care referral alert-to-case exchange-rate table across thresholds.
    - Directional skipped-case (Q=0) action-disagreement bounds (Neg->Pos vs Pos->Neg).
    - Subgroup equity, calibration (CITL, slope, smooth LOESS ICI), FNR/FPR parity, Cochran Q
      heterogeneity, race-blind audit, and hyperparameter sensitivity grid.
"""

from __future__ import annotations

import os

# Enforce single-threaded BLAS/OpenMP inside worker processes before importing numpy/lightgbm
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import gc
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import chi2, ttest_1samp
from sklearn.metrics import average_precision_score

from imboreceipt.bench.fine_grained_cohorts import (
    FineGrainedCohort,
    load_fine_diabetes_discharge,
    load_fine_home_credit,
    load_fine_ieee_cis,
    load_fine_neurips_baf,
    load_fine_scania_aps,
    load_fine_taiwan_credit,
    materialize_leak_free_stress_pool,
)
from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.types import SemanticType, freeze_split
from imboreceipt.discover.selector import CandidateChannelSpec
from imboreceipt.discover.strata import make_leak_free_cv_splits
from imboreceipt.discover.two_phase import (
    discover_competitor_method,
    discover_sequential_conditional_gain,
    evaluate_split_pilot_step1_inference,
    sample_two_phase_pilot,
)
from imboreceipt.frontier.baselines import PlattLogitCalibrator, TabularPreprocessor
from imboreceipt.lattice.covert_afa import (
    CovertGreedyCMIAFA,
    validate_covert_afa_synthetic_oracle,
)
from imboreceipt.lattice.streaming_routing import (
    StreamingSelectiveRouter,
    calibrated_randomized_threshold_masks,
    compute_calibration_summary,
    conformal_fdr_selection,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "two_phase"
OUT_JSON = OUT_DIR / "paper1_v2_1_remediation_results.json"
PARTIAL_DIR = OUT_DIR / "v2_1_blocks"

COHORT_MCIDS: dict[str, float] = {
    "UCI_Diabetes_Discharge": 0.50,
    "Home_Credit_7Table": 0.50,
    "IEEE_CIS_Fraud": 0.25,
    "Scania_APS_Fleet": 0.25,
    "NeurIPS_BAF": 0.25,
    "UCI_Taiwan_Credit": 0.50,
}


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson binomial confidence interval."""
    if n <= 0:
        return (0.0, 1.0)
    lo = 0.0 if k <= 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def clopper_pearson_upper_1sided(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided (1-alpha) Clopper-Pearson binomial upper bound."""
    if n <= 0 or k >= n:
        return 1.0
    if k <= 0:
        return float(1.0 - (alpha ** (1.0 / float(n))))
    return float(beta_dist.ppf(1.0 - alpha, k + 1, n - k))


def prop_summary(flags: list[int] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(flags, dtype=int).ravel()
    k = int(np.sum(arr))
    n = int(arr.size)
    lo, hi = clopper_pearson_ci(k, n)
    return {
        "k": k,
        "n": n,
        "rate": float(k) / max(1, n),
        "cp95_low": lo,
        "cp95_high": hi,
    }


def paired_cluster_bootstrap_nb(
    y: np.ndarray,
    p_num: np.ndarray,
    q_num: np.ndarray,
    p_ref: np.ndarray,
    q_ref: np.ndarray,
    contract: DecisionContract,
    c_query: float,
    groups: np.ndarray | None = None,
    n_boot: int = 10000,
    seed: int = 2026,
) -> dict[str, float]:
    """Vectorized B=10,000 paired cluster/instance ratio-estimator bootstrap for Delta NB."""
    yt = np.asarray(y, dtype=float).ravel()
    pn = np.asarray(p_num, dtype=float).ravel()
    qn = np.asarray(q_num, dtype=float).ravel()
    pr = np.asarray(p_ref, dtype=float).ravel()
    qr = np.asarray(q_ref, dtype=float).ravel()

    cfp = float(contract.cost_fp)
    cfn = max(1e-9, float(contract.cost_fn))
    tau = cfp / (cfp + cfn)
    w_fp = cfp / cfn
    w_q = float(c_query) / cfn

    u_num = 1000.0 * (yt * (pn >= tau) - w_fp * (1.0 - yt) * (pn >= tau) - w_q * qn)
    u_ref = 1000.0 * (yt * (pr >= tau) - w_fp * (1.0 - yt) * (pr >= tau) - w_q * qr)
    diff_i = u_num - u_ref

    nb_num = float(np.mean(u_num))
    nb_ref = float(np.mean(u_ref))
    delta_obs = nb_num - nb_ref

    rng = np.random.default_rng(seed)
    n_rows = yt.size

    if groups is None:
        # Fast multinomial / Poisson-Dirichlet exact cluster=row bootstrap in chunks
        draws = np.empty(n_boot, dtype=float)
        chunk = 500
        for start in range(0, n_boot, chunk):
            b_cur = min(chunk, n_boot - start)
            cnts = rng.multinomial(n_rows, np.full(n_rows, 1.0 / n_rows), size=b_cur)
            draws[start : start + b_cur] = (cnts @ diff_i) / float(n_rows)
    else:
        _, inv = np.unique(groups, return_inverse=True)
        g_cnt = int(np.max(inv)) + 1
        cl_sum = np.bincount(inv, weights=diff_i, minlength=g_cnt)
        cl_n = np.bincount(inv, minlength=g_cnt).astype(float)
        draws = np.empty(n_boot, dtype=float)
        chunk = 500
        p_unif = np.full(g_cnt, 1.0 / g_cnt)
        for start in range(0, n_boot, chunk):
            b_cur = min(chunk, n_boot - start)
            cnts = rng.multinomial(g_cnt, p_unif, size=b_cur)
            num_b = cnts @ cl_sum
            den_b = np.maximum(1.0, cnts @ cl_n)
            draws[start : start + b_cur] = num_b / den_b

    ci_lo, ci_hi = np.quantile(draws, [0.025, 0.975])
    k_neg = int(np.sum(draws <= 0.0))
    k_pos = int(np.sum(draws >= 0.0))
    p_dh = min(1.0, 2.0 * (min(k_neg, k_pos) + 1.0) / float(n_boot + 1))

    return {
        "nb_num": nb_num,
        "nb_ref": nb_ref,
        "delta_nb": delta_obs,
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "p_dh": float(p_dh),
    }


def _fit_oof_stage1_worker(
    X_base: pd.DataFrame,
    y: np.ndarray,
    sem_types: dict[str, SemanticType],
    groups: np.ndarray | None,
    time_order: np.ndarray | None,
    seed: int,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X_base)
    pi = max(1e-5, min(1.0 - 1e-5, float(np.mean(y))))
    oof_p = np.full(n, pi, dtype=float)
    valid = np.zeros(n, dtype=bool)
    splits = make_leak_free_cv_splits(y, n_folds=5, groups=groups, time_order=time_order, seed=seed)
    for f_i, (tr_i, va_i) in enumerate(splits):
        if np.unique(y[tr_i]).size < 2:
            continue
        prep = TabularPreprocessor(semantic_types=sem_types)
        X_tr = prep.fit_transform(X_base.iloc[tr_i])
        X_va = prep.transform(X_base.iloc[va_i])
        clf = lgb.LGBMClassifier(
            n_estimators=80,
            learning_rate=0.06,
            num_leaves=31,
            min_child_samples=max(15, min(40, int(tr_i.size // 40))),
            subsample_for_bin=min(60000, int(tr_i.size)),
            reg_alpha=0.5,
            reg_lambda=3.0,
            max_delta_step=1.0,
            random_state=seed + f_i,
            n_jobs=n_jobs,
            verbose=-1,
        )
        clf.fit(X_tr, y[tr_i])
        tr_raw = np.asarray(clf.predict_proba(X_tr)[:, 1], dtype=float)
        va_raw = np.asarray(clf.predict_proba(X_va)[:, 1], dtype=float)
        cal = PlattLogitCalibrator().fit(tr_raw, y[tr_i])
        oof_p[va_i] = np.clip(cal.predict_proba(va_raw), 1e-9, 1.0 - 1e-9)
        valid[va_i] = True
    return oof_p, valid


def _fit_downstream_gbdt_worker(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    df_test: pd.DataFrame,
    sem_types: dict[str, SemanticType],
    seed: int,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
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
        max_delta_step=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )
    clf.fit(X_tr, y_train)
    raw_ca = np.asarray(clf.predict_proba(X_ca)[:, 1], dtype=float)
    raw_te = np.asarray(clf.predict_proba(X_te)[:, 1], dtype=float)
    cal = PlattLogitCalibrator().fit(raw_ca, y_cal)
    p_ca = np.clip(cal.predict_proba(raw_ca), 1e-9, 1.0 - 1e-9)
    p_te = np.clip(cal.predict_proba(raw_te), 1e-9, 1.0 - 1e-9)
    return p_ca, p_te


# ===========================================================================
# BLOCK 1: Monte Carlo Simulations (1,000-rep Sim A, 200-rep Sim B, 200-rep Sim C)
# ===========================================================================


def _sim_a_one_rep(rep: int) -> dict[str, Any]:
    r_seed = 1100 + rep
    rng = np.random.default_rng(r_seed)
    n = 6000
    x = rng.normal(0.0, 1.0, (n, 3))
    z1 = rng.normal(0.0, 1.0, n)
    eta = -3.4 + 0.9 * x[:, 0] - 0.5 * x[:, 1] + 1.05 * z1
    y = rng.binomial(1, expit(eta))

    X_b = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    z_cols: dict[str, np.ndarray] = {"z_true": z1.astype(np.float32)}
    cands = [CandidateChannelSpec("ch_true", ("z_true",), 1.0, "true")]
    null_cands: list[CandidateChannelSpec] = []
    for j in range(1, 9):
        zn = rng.normal(0.0, 1.0, n).astype(np.float32)
        z_cols[f"z_null_{j}"] = zn
        nc = CandidateChannelSpec(f"ch_null_{j}", (f"z_null_{j}",), 1.0, "null")
        cands.append(nc)
        null_cands.append(nc)
    Z_df = pd.DataFrame(z_cols)

    oof_b, val_m = _fit_oof_stage1_worker(X_b, y, {}, None, None, seed=r_seed, n_jobs=1)

    # Full census reference on first 250 reps (deterministic target mean)
    cen_g = np.nan
    if rep < 250:
        pilot_full = sample_two_phase_pilot(y, oof_b, val_m, pilot_fraction=1.0, design="full")
        res_cen = discover_sequential_conditional_gain(
            X_b, Z_df, y, [cands[0]], oof_b, pilot_full, max_budget=1.0, seed=r_seed
        )
        cen_g = float(res_cen.initial_scores["ch_true"])

    pilot = sample_two_phase_pilot(
        y, oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=r_seed
    )

    # 1. Proposed 4-fold CV + Two-Phase SE + Holm LCB (lazy screen disabled to record Step-2 nulls)
    res_cv = discover_sequential_conditional_gain(
        X_b,
        Z_df,
        y,
        cands,
        oof_b,
        pilot,
        max_budget=3.0,
        allow_lazy_submodular_screen=False,
        multiplicity="holm",
        alpha_select=0.05,
        seed=r_seed,
    )
    step1_null_gains = [float(res_cv.initial_scores[f"ch_null_{j}"]) for j in range(1, 9)]
    # Pure all-null 8-channel pool FWER (exact Step-1 Holm over 8 nulls)
    null_raw_pvals = [float(res_cv.initial_pvals[f"ch_null_{j}"]) for j in range(1, 9)]
    min_null_p = min(null_raw_pvals)
    pure_null_fwer_cv = int(
        (8.0 * min_null_p < 0.05) and (step1_null_gains[int(np.argmin(null_raw_pvals))] > 0.0)
    )

    # 2. Breslow-Cain Stratum Offset variant on identical pilot
    res_bc = discover_sequential_conditional_gain(
        X_b,
        Z_df,
        y,
        cands,
        oof_b,
        pilot,
        max_budget=3.0,
        use_breslow_cain_offset=True,
        allow_lazy_submodular_screen=False,
        multiplicity="holm",
        alpha_select=0.05,
        seed=r_seed,
    )

    # 3. Disjoint 50/50 split-pilot on identical 15% Neyman pilot
    res_sp = evaluate_split_pilot_step1_inference(
        X_b, Z_df, y, cands, oof_b, pilot, alpha_select=0.05, seed=r_seed
    )
    g_s = float(res_sp["scores"]["ch_true"])
    se_s = float(res_sp["ses_total"]["ch_true"])
    targ_s = float(res_sp["census_targets_conditional"]["ch_true"])
    sp_null_gains = [float(res_sp["scores"][f"ch_null_{j}"]) for j in range(1, 9)]

    return {
        "cen_g": cen_g,
        "cv_g": float(res_cv.initial_scores["ch_true"]),
        "cv_se_tot": float(res_cv.initial_ses["ch_true"]),
        "cv_se_p1": float(res_cv.step_trace[0].se_phase1) if res_cv.step_trace else 0.0,
        "cv_null_fwer": int(any(cid.startswith("ch_null_") for cid in res_cv.selected_ids)),
        "cv_pure_null_fwer": pure_null_fwer_cv,
        "cv_power": int("ch_true" in res_cv.selected_ids),
        "bc_g": float(res_bc.initial_scores["ch_true"]),
        "bc_null_fwer": int(any(cid.startswith("ch_null_") for cid in res_bc.selected_ids)),
        "bc_power": int("ch_true" in res_bc.selected_ids),
        "untrunc_no_lcb_fwer": int(any(g > 0.0 for g in step1_null_gains)),
        "pospart_no_lcb_fwer": int(
            any(res_cv.initial_pospart_scores.get(f"ch_null_{j}", 0.0) > 0.0 for j in range(1, 9))
        ),
        "step1_null_gains": step1_null_gains,
        "step2_best_null_gain": (
            float(res_cv.step_trace[1].gain_nats) if len(res_cv.step_trace) >= 2 else np.nan
        ),
        "sp_g": g_s,
        "sp_se_tot": se_s,
        "sp_target": targ_s,
        "sp_cov_cond": int(abs(g_s - targ_s) <= 1.96 * se_s),
        "sp_null_fwer": int(any(cid.startswith("ch_null_") for cid in res_sp["selected_ids"])),
        "sp_power": int("ch_true" in res_sp["selected_ids"]),
        "sp_null_gains": sp_null_gains,
    }


def _sim_b_one_rep(rep: int) -> dict[str, Any]:
    r_seed = 4000 + rep
    rng = np.random.default_rng(r_seed)
    n = 8000
    x = rng.normal(0.0, 1.0, (n, 3))
    z1 = rng.normal(0.0, 1.0, n)
    z2 = rng.normal(0.0, 1.0, n)
    eta = -3.7 + 1.0 * x[:, 0] - 0.6 * x[:, 1] + 1.15 * z1 + 0.75 * z2
    y = rng.binomial(1, expit(eta))

    X_b = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    Z_df = pd.DataFrame({"z1": z1.astype(np.float32), "z2": z2.astype(np.float32)})
    cands = [
        CandidateChannelSpec("ch_1", ("z1",), 1.0, "m1"),
        CandidateChannelSpec("ch_2", ("z2",), 1.0, "m2"),
    ]
    oof_b, val_m = _fit_oof_stage1_worker(X_b, y, {}, None, None, seed=r_seed, n_jobs=1)
    p_full = sample_two_phase_pilot(y, oof_b, val_m, pilot_fraction=1.0, design="full")
    r_full = discover_sequential_conditional_gain(
        X_b, Z_df, y, cands, oof_b, p_full, max_budget=2.0, seed=r_seed
    )
    targ = np.array([r_full.initial_scores["ch_1"], r_full.initial_scores["ch_2"]], dtype=float)

    sq_err: dict[str, float] = {}
    for rho in (0.05, 0.10, 0.20):
        r_lbl = f"{int(round(rho * 100)):02d}"
        for des in ("srs", "bcc", "neyman"):
            pil = sample_two_phase_pilot(
                y, oof_b, val_m, pilot_fraction=rho, design=des, seed=r_seed
            )
            res = discover_sequential_conditional_gain(
                X_b, Z_df, y, cands, oof_b, pil, max_budget=1.0, seed=r_seed
            )
            est = np.array([res.initial_scores["ch_1"], res.initial_scores["ch_2"]], dtype=float)
            sq_err[f"{des}_{r_lbl}"] = float(np.mean((est - targ) ** 2))
            if des == "neyman":
                res_bc = discover_sequential_conditional_gain(
                    X_b,
                    Z_df,
                    y,
                    cands,
                    oof_b,
                    pil,
                    max_budget=1.0,
                    use_breslow_cain_offset=True,
                    seed=r_seed,
                )
                est_bc = np.array(
                    [res_bc.initial_scores["ch_1"], res_bc.initial_scores["ch_2"]], dtype=float
                )
                sq_err[f"neyman_bc_{r_lbl}"] = float(np.mean((est_bc - targ) ** 2))
    return sq_err


def _sim_c_one_rep(rep: int) -> dict[str, Any]:
    r_seed = 6000 + rep
    rng = np.random.default_rng(r_seed)
    n = 7000
    x = rng.normal(0.0, 1.0, (n, 3))
    za = rng.normal(0.0, 1.0, n)
    za_twin = za + rng.normal(0.0, 0.18, n)
    zb = rng.normal(0.0, 1.0, n)
    eta = -2.8 + 0.9 * x[:, 0] - 0.5 * x[:, 1] + 1.25 * za + 0.80 * zb
    y = rng.binomial(1, expit(eta))

    X_b = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    z_dict: dict[str, np.ndarray] = {
        "za": za.astype(np.float32),
        "za_twin": za_twin.astype(np.float32),
        "zb": zb.astype(np.float32),
        "echo_1": (0.85 * x[:, 0] + rng.normal(0.0, 0.25, n)).astype(np.float32),
        "echo_2": (0.65 * (x[:, 0] ** 2) - 0.5 * x[:, 1] + rng.normal(0.0, 0.30, n)).astype(
            np.float32
        ),
    }
    cands = [
        CandidateChannelSpec("ch_A", ("za",), 1.0, "mech_A"),
        CandidateChannelSpec("ch_A_twin", ("za_twin",), 1.0, "twin"),
        CandidateChannelSpec("ch_B", ("zb",), 1.0, "mech_B"),
        CandidateChannelSpec("ch_echo_1", ("echo_1",), 1.0, "echo"),
        CandidateChannelSpec("ch_echo_2", ("echo_2",), 1.0, "echo"),
    ]
    for k in range(1, 12):
        z_dict[f"null_{k}"] = rng.normal(0.0, 1.0, n).astype(np.float32)
        cands.append(CandidateChannelSpec(f"ch_null_{k}", (f"null_{k}",), 1.0, "null"))
    Z_df = pd.DataFrame(z_dict)

    oof_b, val_m = _fit_oof_stage1_worker(X_b, y, {}, None, None, seed=r_seed, n_jobs=1)
    pilot = sample_two_phase_pilot(
        y, oof_b, val_m, pilot_fraction=0.18, design="neyman", seed=r_seed
    )

    def _grade(sel_ids: list[str]) -> dict[str, int]:
        s = set(sel_ids)
        has_a = ("ch_A" in s) or ("ch_A_twin" in s)
        has_b = "ch_B" in s
        dup_twin = int(("ch_A" in s) and ("ch_A_twin" in s))
        has_echo = int(("ch_echo_1" in s) or ("ch_echo_2" in s))
        has_null = int(any(cid.startswith("ch_null_") for cid in s))
        return {
            "both_mech": int(has_a and has_b and not dup_twin and not has_echo and not has_null),
            "dup_twin": dup_twin,
            "echo": has_echo,
            "null": has_null,
        }

    r_seq = discover_sequential_conditional_gain(
        X_b, Z_df, y, cands, oof_b, pilot, max_budget=2.0, seed=r_seed
    )
    r_seq_nolazy = discover_sequential_conditional_gain(
        X_b,
        Z_df,
        y,
        cands,
        oof_b,
        pilot,
        max_budget=2.0,
        allow_lazy_submodular_screen=False,
        seed=r_seed,
    )
    r_oneshot = discover_sequential_conditional_gain(
        X_b, Z_df, y, cands, oof_b, pilot, max_budget=2.0, sequential=False, seed=r_seed
    )
    r_marg = discover_competitor_method(
        "global_mi", X_b, Z_df, y, cands, oof_b, pilot, max_budget=2.0, seed=r_seed
    )
    r_loco = discover_competitor_method(
        "loco_pilot", X_b, Z_df, y, cands, oof_b, pilot, max_budget=2.0, seed=r_seed
    )
    r_ko = discover_competitor_method(
        "crt_knockoff_pilot", X_b, Z_df, y, cands, oof_b, pilot, max_budget=2.0, seed=r_seed
    )
    r_bperm = discover_competitor_method(
        "cpi_pilot", X_b, Z_df, y, cands, oof_b, pilot, max_budget=2.0, seed=r_seed
    )
    r_chamma = discover_competitor_method(
        "chamma_residual_cpi_pilot", X_b, Z_df, y, cands, oof_b, pilot, max_budget=2.0, seed=r_seed
    )

    return {
        "seq_default": _grade(r_seq.selected_ids),
        "seq_no_lazy": _grade(r_seq_nolazy.selected_ids),
        "lazy_exact_match": int(r_seq.selected_ids == r_seq_nolazy.selected_ids),
        "oneshot_cmi": _grade(r_oneshot.selected_ids),
        "marginal_mi": _grade(r_marg.selected_ids),
        "loco": _grade(r_loco.selected_ids),
        "crt_knockoff": _grade(r_ko.selected_ids),
        "baseline_strat_perm": _grade(r_bperm.selected_ids),
        "chamma_residual_cpi": _grade(r_chamma.selected_ids),
    }


def run_block_1(max_workers: int = 8) -> dict[str, Any]:
    out_file = PARTIAL_DIR / "block1_monte_carlo.json"
    if out_file.exists():
        print("Loaded cached Block 1:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    print("[Block 1] Running 1,000 Sim-A replicates + 200 Sim-B + 200 Sim-C...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        sim_a_rows = list(ex.map(_sim_a_one_rep, range(1000), chunksize=8))
        sim_b_rows = list(ex.map(_sim_b_one_rep, range(200), chunksize=5))
        sim_c_rows = list(ex.map(_sim_c_one_rep, range(200), chunksize=5))

    cen_vals = [r["cen_g"] for r in sim_a_rows if not np.isnan(r["cen_g"])]
    cen_mean = float(np.mean(cen_vals))
    cv_g = np.array([r["cv_g"] for r in sim_a_rows])
    cv_se = np.array([r["cv_se_tot"] for r in sim_a_rows])
    cv_se_p1 = np.array([r["cv_se_p1"] for r in sim_a_rows])
    sp_g = np.array([r["sp_g"] for r in sim_a_rows])
    sp_se = np.array([r["sp_se_tot"] for r in sim_a_rows])

    all_s1_null = np.concatenate([np.asarray(r["step1_null_gains"]) for r in sim_a_rows])
    s2_null = np.array(
        [r["step2_best_null_gain"] for r in sim_a_rows if not np.isnan(r["step2_best_null_gain"])]
    )
    all_sp_null = np.concatenate([np.asarray(r["sp_null_gains"]) for r in sim_a_rows])
    t_s1, _ = ttest_1samp(all_s1_null, 0.0)
    t_s2, _ = ttest_1samp(s2_null, 0.0)
    t_sp, _ = ttest_1samp(all_sp_null, 0.0)

    cv_mean = float(np.mean(cv_g))
    sp_mean = float(np.mean(sp_g))

    sim_a_summary = {
        "n_replicates": 1000,
        "census_mean_gain": cen_mean,
        "assumption_1_empirical_null_audit": {
            "step1_n_evals": int(all_s1_null.size),
            "step1_mean_psi_null": float(np.mean(all_s1_null)),
            "step1_ci95_low": float(
                np.mean(all_s1_null)
                - 1.96 * np.std(all_s1_null, ddof=1) / np.sqrt(all_s1_null.size)
            ),
            "step1_ci95_high": float(
                np.mean(all_s1_null)
                + 1.96 * np.std(all_s1_null, ddof=1) / np.sqrt(all_s1_null.size)
            ),
            "step1_t_stat": float(t_s1),
            "step2_post_signal_n_evals": int(s2_null.size),
            "step2_best_null_mean": float(np.mean(s2_null)),
            "step2_t_stat": float(t_s2),
            "split_pilot_n_evals": int(all_sp_null.size),
            "split_pilot_mean_psi_null": float(np.mean(all_sp_null)),
            "split_pilot_t_stat": float(t_sp),
        },
        "pospart_no_lcb_null_fwer": prop_summary([r["pospart_no_lcb_fwer"] for r in sim_a_rows]),
        "untruncated_no_lcb_null_fwer": prop_summary(
            [r["untrunc_no_lcb_fwer"] for r in sim_a_rows]
        ),
        "cv_4fold_ht_holm": {
            "mean_gain": cv_mean,
            "empirical_sd": float(np.std(cv_g, ddof=1)),
            "mean_se_total": float(np.mean(cv_se)),
            "ci95_cov_unconditional": prop_summary(
                (np.abs(cv_g - cv_mean) <= 1.96 * cv_se).astype(int)
            ),
            "ci95_cov_phase1_only": prop_summary(
                (np.abs(cv_g - cv_mean) <= 1.96 * cv_se_p1).astype(int)
            ),
            "ci95_cov_vs_census": prop_summary(
                (np.abs(cv_g - cen_mean) <= 1.96 * cv_se).astype(int)
            ),
            "sequential_null_fwer": prop_summary([r["cv_null_fwer"] for r in sim_a_rows]),
            "pure_all_null_fwer": prop_summary([r["cv_pure_null_fwer"] for r in sim_a_rows]),
            "true_channel_power": prop_summary([r["cv_power"] for r in sim_a_rows]),
        },
        "cv_4fold_breslow_cain_holm": {
            "mean_gain": float(np.mean([r["bc_g"] for r in sim_a_rows])),
            "sequential_null_fwer": prop_summary([r["bc_null_fwer"] for r in sim_a_rows]),
            "true_channel_power": prop_summary([r["bc_power"] for r in sim_a_rows]),
        },
        "split_pilot_50_50_holm": {
            "mean_gain": sp_mean,
            "empirical_sd": float(np.std(sp_g, ddof=1)),
            "mean_se_total": float(np.mean(sp_se)),
            "ci95_cov_data_adaptive": prop_summary([r["sp_cov_cond"] for r in sim_a_rows]),
            "ci95_cov_unconditional": prop_summary(
                (np.abs(sp_g - sp_mean) <= 1.96 * sp_se).astype(int)
            ),
            "null_fwer": prop_summary([r["sp_null_fwer"] for r in sim_a_rows]),
            "true_channel_power": prop_summary([r["sp_power"] for r in sim_a_rows]),
        },
    }

    sim_b_summary: dict[str, Any] = {"n_replicates": 200, "rmse": {}}
    for k_name in sim_b_rows[0]:
        mse_arr = np.array([r[k_name] for r in sim_b_rows], dtype=float)
        rmse_val = float(np.sqrt(np.mean(mse_arr)))
        # Delta-method SE on RMSE
        se_mse = float(np.std(mse_arr, ddof=1) / np.sqrt(mse_arr.size))
        sim_b_summary["rmse"][k_name] = {
            "rmse": rmse_val,
            "se": se_mse / max(1e-8, 2.0 * rmse_val),
        }

    sim_c_methods = [
        "marginal_mi",
        "oneshot_cmi",
        "loco",
        "crt_knockoff",
        "baseline_strat_perm",
        "chamma_residual_cpi",
        "seq_default",
        "seq_no_lazy",
    ]
    sim_c_summary: dict[str, Any] = {
        "n_replicates": 200,
        "lazy_screen_exact_agreement": prop_summary([r["lazy_exact_match"] for r in sim_c_rows]),
        "methods": {},
    }
    for m_key in sim_c_methods:
        sim_c_summary["methods"][m_key] = {
            "both_mech_recovered": prop_summary([r[m_key]["both_mech"] for r in sim_c_rows]),
            "dup_twin_rate": prop_summary([r[m_key]["dup_twin"] for r in sim_c_rows]),
            "echo_rate": prop_summary([r[m_key]["echo"] for r in sim_c_rows]),
            "null_fwer": prop_summary([r[m_key]["null"] for r in sim_c_rows]),
        }

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "sim_a_1000rep": sim_a_summary,
        "sim_b_200rep": sim_b_summary,
        "sim_c_200rep": sim_c_summary,
    }
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 1] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


# ===========================================================================
# BLOCK 2: Leak-Free Train-Fit Stress Pool, Clean 2x2 Ablations, & 120-Split Ledger
# ===========================================================================


def _load_native_cohort(
    cohort_key: str, split_idx: int, n_splits: int, seed: int
) -> FineGrainedCohort:
    if cohort_key == "UCI_Diabetes_Discharge":
        return load_fine_diabetes_discharge(include_stress_channels=False, seed=seed)
    if cohort_key == "Home_Credit_7Table":
        return load_fine_home_credit(include_stress_channels=False, max_rows=140000, seed=seed)
    if cohort_key == "IEEE_CIS_Fraud":
        return load_fine_ieee_cis(
            window_idx=split_idx, n_windows=n_splits, max_window_rows=130000, seed=seed
        )
    if cohort_key == "Scania_APS_Fleet":
        return load_fine_scania_aps(include_stress_channels=False, seed=seed)
    if cohort_key == "NeurIPS_BAF":
        return load_fine_neurips_baf(
            window_idx=split_idx, n_windows=n_splits, max_window_rows=140000, seed=seed
        )
    if cohort_key == "UCI_Taiwan_Credit":
        return load_fine_taiwan_credit(restrict_baseline_current=False, seed=seed)
    raise ValueError(cohort_key)


def _spurious_rates(
    selected_ids: list[str], cands: list[CandidateChannelSpec]
) -> tuple[float, float, float]:
    if not selected_ids:
        return (0.0, 0.0, 0.0)
    tag_map = {c.channel_id: c.mechanism_tag for c in cands}
    n_s = float(len(selected_ids))
    n_null = sum(1 for cid in selected_ids if tag_map.get(cid) == "pure_noise_null")
    n_echo = sum(1 for cid in selected_ids if tag_map.get(cid) == "baseline_echo")
    # Duplicate Stage-2 twin waste: buying BOTH a native parent and its s2_twin_of_<parent>
    s_set = set(selected_ids)
    n_dup_twin = 0
    for cid in selected_ids:
        if cid.startswith("s2_twin_of_"):
            parent = cid[len("s2_twin_of_") :]
            if parent in s_set:
                n_dup_twin += 1
    return (100.0 * n_null / n_s, 100.0 * n_dup_twin / n_s, 100.0 * n_echo / n_s)


def _stress_and_ablation_task(args: tuple[str, int]) -> dict[str, Any]:
    cohort_key, split_idx = args
    seed = 100 + split_idx
    coh = _load_native_cohort(cohort_key, split_idx, 20, seed)
    split_df = coh.X_base.copy()
    split_df["__y__"] = coh.y
    grp_col = "__grp__" if coh.group_ids is not None else None
    tim_col = "__time__" if coh.time_order is not None else None
    if grp_col:
        split_df[grp_col] = coh.group_ids
    if tim_col:
        split_df[tim_col] = coh.time_order

    sp = freeze_split(split_df, "__y__", group_col=grp_col, time_col=tim_col, seed=seed)
    tr_i = np.asarray(sp.train_idx, dtype=int)
    ca_i = np.asarray(sp.cal_idx, dtype=int)
    te_i = np.asarray(sp.test_idx, dtype=int)

    # Fit stress twins & echoes STRICTLY on train_idx (zero test/cal/val statistics!)
    Z_stress, cands_stress = materialize_leak_free_stress_pool(coh, train_idx=tr_i, seed=seed)

    Xb_tr = coh.X_base.iloc[tr_i].reset_index(drop=True)
    Xb_ca = coh.X_base.iloc[ca_i].reset_index(drop=True)
    Xb_te = coh.X_base.iloc[te_i].reset_index(drop=True)
    Zs_tr = Z_stress.iloc[tr_i].reset_index(drop=True)
    Zs_ca = Z_stress.iloc[ca_i].reset_index(drop=True)
    Zs_te = Z_stress.iloc[te_i].reset_index(drop=True)
    y_tr, y_ca, y_te = coh.y[tr_i], coh.y[ca_i], coh.y[te_i]
    grp_tr = coh.group_ids[tr_i] if coh.group_ids is not None else None
    tim_tr = coh.time_order[tr_i] if coh.time_order is not None else None

    oof_b, val_m = _fit_oof_stage1_worker(
        Xb_tr, y_tr, coh.base_semantic_types, grp_tr, tim_tr, seed=seed, n_jobs=1
    )
    pilot = sample_two_phase_pilot(
        y_tr, oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=seed
    )

    def _eval_sel(sel_cands: list[CandidateChannelSpec]) -> dict[str, Any]:
        cols = [col for c in sel_cands for col in c.columns]
        df_tr = pd.concat([Xb_tr, Zs_tr[cols]], axis=1) if cols else Xb_tr
        df_ca = pd.concat([Xb_ca, Zs_ca[cols]], axis=1) if cols else Xb_ca
        df_te = pd.concat([Xb_te, Zs_te[cols]], axis=1) if cols else Xb_te
        _, p_te = _fit_downstream_gbdt_worker(
            df_tr, y_tr, df_ca, y_ca, df_te, coh.base_semantic_types, seed=seed, n_jobs=1
        )
        sel_ids = [c.channel_id for c in sel_cands]
        nr, tr, er = _spurious_rates(sel_ids, cands_stress)
        return {
            "auprc": float(average_precision_score(y_te, p_te)),
            "selected_ids": sel_ids,
            "null_pct": nr,
            "twin_pct": tr,
            "echo_pct": er,
        }

    # Proposed Neyman Seq-CMI + Holm LCB (with and without lazy screen)
    r_prop = discover_sequential_conditional_gain(
        Xb_tr,
        Zs_tr,
        y_tr,
        cands_stress,
        oof_b,
        pilot,
        max_budget=coh.acquisition_budget,
        base_semantic_types=coh.base_semantic_types,
        groups_train=grp_tr,
        time_order_train=tim_tr,
        seed=seed,
    )
    r_nolazy = discover_sequential_conditional_gain(
        Xb_tr,
        Zs_tr,
        y_tr,
        cands_stress,
        oof_b,
        pilot,
        max_budget=coh.acquisition_budget,
        base_semantic_types=coh.base_semantic_types,
        groups_train=grp_tr,
        time_order_train=tim_tr,
        allow_lazy_submodular_screen=False,
        seed=seed,
    )
    r_bc = discover_sequential_conditional_gain(
        Xb_tr,
        Zs_tr,
        y_tr,
        cands_stress,
        oof_b,
        pilot,
        max_budget=coh.acquisition_budget,
        base_semantic_types=coh.base_semantic_types,
        groups_train=grp_tr,
        time_order_train=tim_tr,
        use_breslow_cain_offset=True,
        seed=seed,
    )
    r_chamma = discover_competitor_method(
        "chamma_residual_cpi_pilot",
        Xb_tr,
        Zs_tr,
        y_tr,
        cands_stress,
        oof_b,
        pilot,
        max_budget=coh.acquisition_budget,
        base_semantic_types=coh.base_semantic_types,
        groups_train=grp_tr,
        time_order_train=tim_tr,
        seed=seed,
    )

    out_row: dict[str, Any] = {
        "cohort_key": cohort_key,
        "split_idx": split_idx,
        "lazy_exact_match": int(r_prop.selected_ids == r_nolazy.selected_ids),
        "proposed_ht_holm": _eval_sel(r_prop.selected_channels),
        "no_lazy_ht_holm": _eval_sel(r_nolazy.selected_channels),
        "breslow_cain_holm": _eval_sel(r_bc.selected_channels),
        "chamma_residual_cpi": _eval_sel(r_chamma.selected_channels),
    }

    # Clean 2x2 HT Weighting x Holm LCB ablation on UCI Diabetes and Scania APS
    if cohort_key in {"UCI_Diabetes_Discharge", "Scania_APS_Fleet"}:
        r_ht_nolcb = discover_sequential_conditional_gain(
            Xb_tr,
            Zs_tr,
            y_tr,
            cands_stress,
            oof_b,
            pilot,
            max_budget=coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            use_ht_weights=True,
            multiplicity="none",
            alpha_select=0.50,
            seed=seed,
        )
        r_unwt_holm = discover_sequential_conditional_gain(
            Xb_tr,
            Zs_tr,
            y_tr,
            cands_stress,
            oof_b,
            pilot,
            max_budget=coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            use_ht_weights=False,
            multiplicity="holm",
            alpha_select=0.05,
            seed=seed,
        )
        r_unwt_nolcb = discover_sequential_conditional_gain(
            Xb_tr,
            Zs_tr,
            y_tr,
            cands_stress,
            oof_b,
            pilot,
            max_budget=coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            use_ht_weights=False,
            multiplicity="none",
            alpha_select=0.50,
            seed=seed,
        )
        out_row["ablation_2x2"] = {
            "ht_no_lcb": _eval_sel(r_ht_nolcb.selected_channels),
            "unwt_holm": _eval_sel(r_unwt_holm.selected_channels),
            "unwt_no_lcb": _eval_sel(r_unwt_nolcb.selected_channels),
        }

    return out_row


def _summarize_120_cached_splits() -> dict[str, Any]:
    base_dir = OUT_DIR / "checkpoints"
    cohorts = list(COHORT_MCIDS.keys())
    ledger: dict[str, Any] = {}
    for c_key in cohorts:
        ch_counts_15: dict[str, int] = {}
        ch_counts_100: dict[str, int] = {}
        spent_15: list[float] = []
        spent_100: list[float] = []
        for s_i in range(20):
            ck = json.loads((base_dir / f"{c_key}_split{s_i:02d}.json").read_text(encoding="utf-8"))
            arm15 = ck["native_arms"]["neyman_15pct_seq"]
            arm100 = ck["native_arms"]["full_100pct_seq"]
            spent_15.append(float(arm15["cost_spent"]))
            spent_100.append(float(arm100["cost_spent"]))
            for cid in arm15["selected_ids"]:
                ch_counts_15[cid] = ch_counts_15.get(cid, 0) + 1
            for cid in arm100["selected_ids"]:
                ch_counts_100[cid] = ch_counts_100.get(cid, 0) + 1
        top15 = sorted(ch_counts_15.items(), key=lambda kv: (-kv[1], kv[0]))
        top100 = sorted(ch_counts_100.items(), key=lambda kv: (-kv[1], kv[0]))
        v2_sum = json.loads((OUT_DIR / "v2_campaign_summary.json").read_text(encoding="utf-8"))
        ledger[c_key] = {
            "mean_spent_15pct": float(np.mean(spent_15)),
            "mean_spent_100pct": float(np.mean(spent_100)),
            "top_selected_15pct_out_of_20": top15[:6],
            "top_selected_100pct_out_of_20": top100[:6],
            "pilot_frontier_summary": v2_sum["cohorts_v2"][c_key]["pilot_fraction_curve"],
        }
    return ledger


def run_block_2(max_workers: int = 8) -> dict[str, Any]:
    out_file = PARTIAL_DIR / "block2_stress_and_ablations.json"
    if out_file.exists():
        print("Loaded cached Block 2:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    print("[Block 2] Running leak-free train-fit stress-pool & 2x2 ablation suite...")
    tasks: list[tuple[str, int]] = []
    for c_key in COHORT_MCIDS:
        n_sp = 10 if c_key in {"UCI_Diabetes_Discharge", "Scania_APS_Fleet"} else 5
        for s_i in range(n_sp):
            tasks.append((c_key, s_i))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_stress_and_ablation_task, tasks, chunksize=1))

    by_cohort: dict[str, Any] = {}
    for c_key in COHORT_MCIDS:
        sub = [r for r in rows if r["cohort_key"] == c_key]
        entry: dict[str, Any] = {
            "n_splits": len(sub),
            "lazy_exact_match_rate": float(np.mean([r["lazy_exact_match"] for r in sub])),
        }
        for arm in (
            "proposed_ht_holm",
            "no_lazy_ht_holm",
            "breslow_cain_holm",
            "chamma_residual_cpi",
        ):
            entry[arm] = {
                "auprc_mean": float(np.mean([r[arm]["auprc"] for r in sub])),
                "null_pct": float(np.mean([r[arm]["null_pct"] for r in sub])),
                "twin_pct": float(np.mean([r[arm]["twin_pct"] for r in sub])),
                "echo_pct": float(np.mean([r[arm]["echo_pct"] for r in sub])),
            }
        if "ablation_2x2" in sub[0]:
            entry["ablation_2x2"] = {}
            for ab_k in ("ht_no_lcb", "unwt_holm", "unwt_no_lcb"):
                entry["ablation_2x2"][ab_k] = {
                    "auprc_mean": float(np.mean([r["ablation_2x2"][ab_k]["auprc"] for r in sub])),
                    "null_pct": float(np.mean([r["ablation_2x2"][ab_k]["null_pct"] for r in sub])),
                    "twin_pct": float(np.mean([r["ablation_2x2"][ab_k]["twin_pct"] for r in sub])),
                    "echo_pct": float(np.mean([r["ablation_2x2"][ab_k]["echo_pct"] for r in sub])),
                }
        by_cohort[c_key] = entry

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "leak_free_train_fit_stress": by_cohort,
        "cached_120_split_channel_ledger": _summarize_120_cached_splits(),
    }
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 2] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


# ===========================================================================
# BLOCK 3: Full Canonical Non-Overlapping Holdouts, B=10,000 Bootstraps,
#          Covert-AFA, Gauss-Hermite vs Empirical Quantile VOI, & CfBH Shift
# ===========================================================================


def _load_full_canonical_cohort(cohort_key: str) -> tuple[FineGrainedCohort, np.ndarray | None]:
    """Load FULL unsampled cohort + cluster IDs for strictly non-overlapping canonical holdout."""
    seed = 2026
    if cohort_key == "UCI_Diabetes_Discharge":
        coh = load_fine_diabetes_discharge(include_stress_channels=False, seed=seed)
        return coh, coh.group_ids
    if cohort_key == "Home_Credit_7Table":
        # Full unsampled N = 307,511 borrowers!
        coh = load_fine_home_credit(include_stress_channels=False, max_rows=None, seed=seed)
        return coh, None
    if cohort_key == "IEEE_CIS_Fraud":
        # Full unsampled N = 590,540 chronological stream + card1 cluster IDs for bootstrap!
        coh = load_fine_ieee_cis(window_idx=None, n_windows=1, max_window_rows=600000, seed=seed)
        card_groups = coh.X_base["card1"].fillna(-1).to_numpy(dtype=np.int64)
        return coh, card_groups
    if cohort_key == "Scania_APS_Fleet":
        coh = load_fine_scania_aps(include_stress_channels=False, seed=seed)
        return coh, None
    if cohort_key == "NeurIPS_BAF":
        # Full unsampled N = 1,000,000 chronological onboarding stream!
        coh = load_fine_neurips_baf(
            window_idx=None, n_windows=1, max_window_rows=1000000, seed=seed
        )
        return coh, None
    if cohort_key == "UCI_Taiwan_Credit":
        coh = load_fine_taiwan_credit(restrict_baseline_current=False, seed=seed)
        return coh, None
    raise ValueError(cohort_key)


def _evaluate_canonical_cohort_worker(cohort_key: str) -> dict[str, Any]:
    seed = 2026
    coh, boot_groups_all = _load_full_canonical_cohort(cohort_key)

    split_df = coh.X_base.copy()
    split_df["__y__"] = coh.y
    grp_col = "__grp__" if coh.group_ids is not None else None
    tim_col = "__time__" if coh.time_order is not None else None
    if grp_col:
        split_df[grp_col] = coh.group_ids
    if tim_col:
        split_df[tim_col] = coh.time_order

    sp = freeze_split(split_df, "__y__", group_col=grp_col, time_col=tim_col, seed=seed)
    tr_i = np.asarray(sp.train_idx, dtype=int)
    ca_i = np.asarray(sp.cal_idx, dtype=int)
    te_i = np.asarray(sp.test_idx, dtype=int)

    Xb_tr = coh.X_base.iloc[tr_i].reset_index(drop=True)
    Xb_ca = coh.X_base.iloc[ca_i].reset_index(drop=True)
    Xb_te = coh.X_base.iloc[te_i].reset_index(drop=True)
    Z_tr = coh.Z_pool.iloc[tr_i].reset_index(drop=True)
    Z_ca = coh.Z_pool.iloc[ca_i].reset_index(drop=True)
    Z_te = coh.Z_pool.iloc[te_i].reset_index(drop=True)
    y_tr, y_ca, y_te = coh.y[tr_i], coh.y[ca_i], coh.y[te_i]
    grp_tr = coh.group_ids[tr_i] if coh.group_ids is not None else None
    tim_tr = coh.time_order[tr_i] if coh.time_order is not None else None
    boot_grp_te = boot_groups_all[te_i] if boot_groups_all is not None else None

    # Cap OOF + discovery training rows at 160k on 590k/1M streams while evaluating FULL test era!
    disc_sub = tr_i.size
    if disc_sub > 160000:
        # Keep most recent 160k training rows in chronological streams
        Xb_disc = Xb_tr.iloc[-160000:].reset_index(drop=True)
        Z_disc = Z_tr.iloc[-160000:].reset_index(drop=True)
        y_disc = y_tr[-160000:]
        grp_disc = grp_tr[-160000:] if grp_tr is not None else None
        tim_disc = tim_tr[-160000:] if tim_tr is not None else None
    else:
        Xb_disc, Z_disc, y_disc, grp_disc, tim_disc = Xb_tr, Z_tr, y_tr, grp_tr, tim_tr

    oof_b, val_m = _fit_oof_stage1_worker(
        Xb_disc, y_disc, coh.base_semantic_types, grp_disc, tim_disc, seed=seed, n_jobs=2
    )
    pilot_15 = sample_two_phase_pilot(
        y_disc, oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=seed
    )
    sel_15 = discover_sequential_conditional_gain(
        Xb_disc,
        Z_disc,
        y_disc,
        coh.candidates,
        oof_b,
        pilot_15,
        max_budget=coh.acquisition_budget,
        base_semantic_types=coh.base_semantic_types,
        groups_train=grp_disc,
        time_order_train=tim_disc,
        seed=seed,
    )
    sel_cols = [col for c in sel_15.selected_channels for col in c.columns]

    # Fit Stage-1 and Stage-2 downstream predictors on FULL Xb_tr / Z_tr
    p1_ca, p1_te = _fit_downstream_gbdt_worker(
        Xb_tr, y_tr, Xb_ca, y_ca, Xb_te, coh.base_semantic_types, seed=seed, n_jobs=2
    )
    if sel_cols:
        p2_ca, p2_te = _fit_downstream_gbdt_worker(
            pd.concat([Xb_tr, Z_tr[sel_cols]], axis=1),
            y_tr,
            pd.concat([Xb_ca, Z_ca[sel_cols]], axis=1),
            y_ca,
            pd.concat([Xb_te, Z_te[sel_cols]], axis=1),
            coh.base_semantic_types,
            seed=seed + 7,
            n_jobs=2,
        )
    else:
        p2_ca, p2_te = p1_ca.copy(), p1_te.copy()

    # 1. Gauss-Hermite Martingale VOI (Prop 2) & Uncentered VOI & Top-Risk
    router_gh = StreamingSelectiveRouter(
        coh.contract,
        per_instance_query_cost=coh.per_instance_query_cost,
        quadrature_mode="gauss_hermite",
        mean_centered=True,
    ).fit(y_ca, p1_ca, p2_ca)
    router_unc = StreamingSelectiveRouter(
        coh.contract,
        per_instance_query_cost=coh.per_instance_query_cost,
        quadrature_mode="gauss_hermite",
        mean_centered=False,
    ).fit(y_ca, p1_ca, p2_ca)
    # 2. Empirical Quantile Martingale VOI (K=33)
    router_emp = StreamingSelectiveRouter(
        coh.contract,
        per_instance_query_cost=coh.per_instance_query_cost,
        quadrature_mode="empirical_quantile",
        n_emp_quantiles=33,
        mean_centered=True,
    ).fit(y_ca, p1_ca, p2_ca)

    rng_u = np.random.default_rng(seed + 55)
    u_ca = rng_u.uniform(0.0, 1.0, p1_ca.size)
    u_te = rng_u.uniform(0.0, 1.0, p1_te.size)

    voi_gh_te = router_gh.estimate_voi(p1_te)
    voi_unc_te = router_unc.estimate_voi(p1_te)
    voi_emp_te = router_emp.estimate_voi(p1_te)

    _, q_gh20, _, _ = calibrated_randomized_threshold_masks(
        router_gh.voi_cal_, voi_gh_te, 0.20, u_ca, u_te
    )
    _, q_unc20, _, _ = calibrated_randomized_threshold_masks(
        router_unc.voi_cal_, voi_unc_te, 0.20, u_ca, u_te
    )
    _, q_emp20, _, _ = calibrated_randomized_threshold_masks(
        router_emp.voi_cal_, voi_emp_te, 0.20, u_ca, u_te
    )
    _, q_top20, _, _ = calibrated_randomized_threshold_masks(p1_ca, p1_te, 0.20, u_ca, u_te)

    q_gh_auto = voi_gh_te > coh.per_instance_query_cost
    q_unc_auto = voi_unc_te > coh.per_instance_query_cost
    q_emp_auto = voi_emp_te > coh.per_instance_query_cost
    q_zero = np.zeros(y_te.size, dtype=bool)
    q_full = np.ones(y_te.size, dtype=bool)

    p_gh20 = np.where(q_gh20, p2_te, p1_te)
    p_unc20 = np.where(q_unc20, p2_te, p1_te)
    p_emp20 = np.where(q_emp20, p2_te, p1_te)
    p_top20 = np.where(q_top20, p2_te, p1_te)
    p_gh_auto = np.where(q_gh_auto, p2_te, p1_te)
    p_unc_auto = np.where(q_unc_auto, p2_te, p1_te)
    p_emp_auto = np.where(q_emp_auto, p2_te, p1_te)

    # 3. Published Sequential Instancewise Greedy CMI (CovertGreedyCMIAFA) + 2 pure-noise channels
    Z_tr_afa, cands_afa = materialize_leak_free_stress_pool(coh, train_idx=tr_i, seed=seed)
    # Keep native channels + 2 pure-noise channels to measure noise-leakage alongside native pool
    keep_afa_cands = [
        c
        for c in cands_afa
        if c.mechanism_tag not in {"stage2_twin", "baseline_echo"}
        and c.channel_id not in {"null_channel_3", "null_channel_4"}
    ]
    afa = CovertGreedyCMIAFA(
        keep_afa_cands,
        coh.contract,
        coh.per_instance_query_cost,
        coh.acquisition_budget,
        max_steps_per_instance=3,
        n_jobs=2,
        seed=seed,
    )
    afa.fit(
        Xb_disc,
        Z_tr_afa.iloc[tr_i[-len(Xb_disc) :]].reset_index(drop=True),
        y_disc,
        oof_b,
        Xb_ca,
        Z_tr_afa.iloc[ca_i].reset_index(drop=True),
        y_ca,
        p1_ca,
    )
    afa_20 = afa.evaluate(
        Xb_te, Z_tr_afa.iloc[te_i].reset_index(drop=True), y_te, p1_te, target_query_rate=0.20
    )
    afa_auto = afa.evaluate(
        Xb_te, Z_tr_afa.iloc[te_i].reset_index(drop=True), y_te, p1_te, target_query_rate=None
    )

    prev_te = float(np.mean(y_te))
    cal_s1 = compute_calibration_summary(y_te, p1_te)
    cal_s2 = compute_calibration_summary(y_te, p2_te)
    cal_auto = compute_calibration_summary(y_te, p_gh_auto)

    def _snb(nb: float) -> float:
        return nb / max(1e-6, 1000.0 * prev_te)

    # B = 10,000 paired cluster/instance bootstraps
    bt_gh20_vs_s1 = paired_cluster_bootstrap_nb(
        y_te, p_gh20, q_gh20, p1_te, q_zero, coh.contract, coh.per_instance_query_cost, boot_grp_te
    )
    bt_gh20_vs_unc20 = paired_cluster_bootstrap_nb(
        y_te,
        p_gh20,
        q_gh20,
        p_unc20,
        q_unc20,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )
    bt_gh20_vs_top20 = paired_cluster_bootstrap_nb(
        y_te,
        p_gh20,
        q_gh20,
        p_top20,
        q_top20,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )
    bt_gh20_vs_afa20 = paired_cluster_bootstrap_nb(
        y_te,
        p_gh20,
        q_gh20,
        afa_20.probs_test,
        afa_20.queried_any_test,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )
    bt_auto_vs_s1 = paired_cluster_bootstrap_nb(
        y_te,
        p_gh_auto,
        q_gh_auto,
        p1_te,
        q_zero,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )
    bt_auto_vs_unc = paired_cluster_bootstrap_nb(
        y_te,
        p_gh_auto,
        q_gh_auto,
        p_unc_auto,
        q_unc_auto,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )
    bt_auto_vs_full = paired_cluster_bootstrap_nb(
        y_te,
        p_gh_auto,
        q_gh_auto,
        p2_te,
        q_full,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )
    bt_auto_vs_afa = paired_cluster_bootstrap_nb(
        y_te,
        p_gh_auto,
        q_gh_auto,
        afa_auto.probs_test,
        afa_auto.queried_any_test,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )
    bt_emp_vs_gh_auto = paired_cluster_bootstrap_nb(
        y_te,
        p_emp_auto,
        q_emp_auto,
        p_gh_auto,
        q_gh_auto,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )
    bt_emp_vs_gh_20 = paired_cluster_bootstrap_nb(
        y_te,
        p_emp20,
        q_emp20,
        p_gh20,
        q_gh20,
        coh.contract,
        coh.per_instance_query_cost,
        boot_grp_te,
    )

    gc.collect()
    return {
        "cohort_key": cohort_key,
        "n_total": int(len(coh.X_base)),
        "n_train": int(tr_i.size),
        "n_cal": int(ca_i.size),
        "n_test": int(te_i.size),
        "n_test_clusters": (
            int(np.unique(boot_grp_te).size) if boot_grp_te is not None else int(te_i.size)
        ),
        "prevalence_test": prev_te,
        "tau_star": float(coh.contract.cost_fp / (coh.contract.cost_fp + coh.contract.cost_fn)),
        "mcid": COHORT_MCIDS[cohort_key],
        "selected_ids_15pct": sel_15.selected_ids,
        "spent_15pct": float(sel_15.total_cost_spent),
        "auprc": {
            "stage1": float(average_precision_score(y_te, p1_te)),
            "stage2_full": float(average_precision_score(y_te, p2_te)),
            "gh_voi_20_routed": float(average_precision_score(y_te, p_gh20)),
            "gh_voi_auto_routed": float(average_precision_score(y_te, p_gh_auto)),
            "emp_voi_auto_routed": float(average_precision_score(y_te, p_emp_auto)),
            "covert_afa_20_routed": afa_20.auprc_routed,
            "covert_afa_auto_routed": afa_auto.auprc_routed,
        },
        "net_benefit_and_query_rates": {
            "stage1_0pct": {
                "nb": bt_auto_vs_s1["nb_ref"],
                "snb": _snb(bt_auto_vs_s1["nb_ref"]),
                "q": 0.0,
            },
            "top_risk_20pct": {"nb": bt_gh20_vs_top20["nb_ref"], "q": float(np.mean(q_top20))},
            "uncentered_20pct": {"nb": bt_gh20_vs_unc20["nb_ref"], "q": float(np.mean(q_unc20))},
            "gh_voi_20pct": {
                "nb": bt_gh20_vs_s1["nb_num"],
                "snb": _snb(bt_gh20_vs_s1["nb_num"]),
                "q": float(np.mean(q_gh20)),
            },
            "emp_voi_20pct": {
                "nb": bt_emp_vs_gh_20["nb_num"],
                "snb": _snb(bt_emp_vs_gh_20["nb_num"]),
                "q": float(np.mean(q_emp20)),
            },
            "covert_afa_20pct": {
                "nb": afa_20.net_benefit_per_1k,
                "q": afa_20.any_query_rate_test,
                "null_q": afa_20.null_channel_query_rate_test,
                "top_channels": afa_20.top_channels_by_freq,
            },
            "uncentered_auto": {"nb": bt_auto_vs_unc["nb_ref"], "q": float(np.mean(q_unc_auto))},
            "gh_voi_auto": {
                "nb": bt_auto_vs_s1["nb_num"],
                "snb": _snb(bt_auto_vs_s1["nb_num"]),
                "q": float(np.mean(q_gh_auto)),
            },
            "emp_voi_auto": {
                "nb": bt_emp_vs_gh_auto["nb_num"],
                "snb": _snb(bt_emp_vs_gh_auto["nb_num"]),
                "q": float(np.mean(q_emp_auto)),
            },
            "covert_afa_auto": {
                "nb": afa_auto.net_benefit_per_1k,
                "q": afa_auto.any_query_rate_test,
                "null_q": afa_auto.null_channel_query_rate_test,
                "top_channels": afa_auto.top_channels_by_freq,
            },
            "full_stage2_100pct": {
                "nb": bt_auto_vs_full["nb_ref"],
                "snb": _snb(bt_auto_vs_full["nb_ref"]),
                "q": 1.0,
            },
        },
        "bootstraps_10k": {
            "gh20_vs_stage1": bt_gh20_vs_s1,
            "gh20_vs_uncentered20": bt_gh20_vs_unc20,
            "gh20_vs_toprisk20": bt_gh20_vs_top20,
            "gh20_vs_covertafa20": bt_gh20_vs_afa20,
            "gh_auto_vs_stage1": bt_auto_vs_s1,
            "gh_auto_vs_uncentered_auto": bt_auto_vs_unc,
            "gh_auto_vs_full100": bt_auto_vs_full,
            "gh_auto_vs_covertafa_auto": bt_auto_vs_afa,
            "emp_auto_vs_gh_auto": bt_emp_vs_gh_auto,
            "emp_20_vs_gh_20": bt_emp_vs_gh_20,
        },
        "calibration": {
            "stage1": {
                "citl": cal_s1.cal_intercept,
                "slope": cal_s1.cal_slope,
                "ici": cal_s1.ici,
                "ece": cal_s1.ece,
            },
            "stage2_full": {
                "citl": cal_s2.cal_intercept,
                "slope": cal_s2.cal_slope,
                "ici": cal_s2.ici,
                "ece": cal_s2.ece,
            },
            "gh_voi_auto": {
                "citl": cal_auto.cal_intercept,
                "slope": cal_auto.cal_slope,
                "ici": cal_auto.ici,
                "ece": cal_auto.ece,
            },
        },
    }


def _run_disjoint_chrono_and_cfbh_shift_audit() -> dict[str, Any]:
    """Audit (a) strictly non-overlapping chronological eras on IEEE-CIS & NeurIPS BAF and
    (b) Conformal CfBH sensitivity under temporal distribution shift + local burn-in fix.
    """
    out_audit: dict[str, Any] = {"disjoint_4_eras": {}, "cfbh_temporal_shift": {}}

    for c_key in ("IEEE_CIS_Fraud", "NeurIPS_BAF"):
        coh, _ = _load_full_canonical_cohort(c_key)
        n_tot = len(coh.X_base)
        # Partition into 4 strictly mutually disjoint contiguous chronological eras!
        era_edges = np.linspace(0, n_tot, 5, dtype=int)
        era_deltas_15_100: list[float] = []
        era_auto_gains: list[float] = []
        for e_i in range(4):
            lo, hi = int(era_edges[e_i]), int(era_edges[e_i + 1])
            Xb_e = coh.X_base.iloc[lo:hi].reset_index(drop=True)
            Z_e = coh.Z_pool.iloc[lo:hi].reset_index(drop=True)
            y_e = coh.y[lo:hi]
            t_e = coh.time_order[lo:hi]
            s_df = Xb_e.copy()
            s_df["__y__"] = y_e
            s_df["__t__"] = t_e
            sp = freeze_split(s_df, "__y__", time_col="__t__", seed=2026 + e_i)
            tr, ca, te = np.asarray(sp.train_idx), np.asarray(sp.cal_idx), np.asarray(sp.test_idx)
            oof_b, val_m = _fit_oof_stage1_worker(
                Xb_e.iloc[tr], y_e[tr], coh.base_semantic_types, None, t_e[tr], seed=2026 + e_i
            )
            p15 = sample_two_phase_pilot(
                y_e[tr], oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=2026 + e_i
            )
            p100 = sample_two_phase_pilot(
                y_e[tr], oof_b, val_m, pilot_fraction=1.0, design="full", seed=2026 + e_i
            )
            s15 = discover_sequential_conditional_gain(
                Xb_e.iloc[tr].reset_index(drop=True),
                Z_e.iloc[tr].reset_index(drop=True),
                y_e[tr],
                coh.candidates,
                oof_b,
                p15,
                max_budget=coh.acquisition_budget,
                base_semantic_types=coh.base_semantic_types,
                time_order_train=t_e[tr],
                seed=2026 + e_i,
            )
            s100 = discover_sequential_conditional_gain(
                Xb_e.iloc[tr].reset_index(drop=True),
                Z_e.iloc[tr].reset_index(drop=True),
                y_e[tr],
                coh.candidates,
                oof_b,
                p100,
                max_budget=coh.acquisition_budget,
                base_semantic_types=coh.base_semantic_types,
                time_order_train=t_e[tr],
                seed=2026 + e_i,
            )
            cols15 = [c for ch in s15.selected_channels for c in ch.columns]
            cols100 = [c for ch in s100.selected_channels for c in ch.columns]
            p1_ca, p1_te = _fit_downstream_gbdt_worker(
                Xb_e.iloc[tr],
                y_e[tr],
                Xb_e.iloc[ca],
                y_e[ca],
                Xb_e.iloc[te],
                coh.base_semantic_types,
                seed=2026 + e_i,
            )
            p2_ca_15, p2_te_15 = _fit_downstream_gbdt_worker(
                pd.concat([Xb_e.iloc[tr], Z_e.iloc[tr][cols15]], axis=1)
                if cols15
                else Xb_e.iloc[tr],
                y_e[tr],
                pd.concat([Xb_e.iloc[ca], Z_e.iloc[ca][cols15]], axis=1)
                if cols15
                else Xb_e.iloc[ca],
                y_e[ca],
                pd.concat([Xb_e.iloc[te], Z_e.iloc[te][cols15]], axis=1)
                if cols15
                else Xb_e.iloc[te],
                coh.base_semantic_types,
                seed=2036 + e_i,
            )
            _, p2_te_100 = _fit_downstream_gbdt_worker(
                pd.concat([Xb_e.iloc[tr], Z_e.iloc[tr][cols100]], axis=1)
                if cols100
                else Xb_e.iloc[tr],
                y_e[tr],
                pd.concat([Xb_e.iloc[ca], Z_e.iloc[ca][cols100]], axis=1)
                if cols100
                else Xb_e.iloc[ca],
                y_e[ca],
                pd.concat([Xb_e.iloc[te], Z_e.iloc[te][cols100]], axis=1)
                if cols100
                else Xb_e.iloc[te],
                coh.base_semantic_types,
                seed=2046 + e_i,
            )
            era_deltas_15_100.append(
                float(
                    average_precision_score(y_e[te], p2_te_15)
                    - average_precision_score(y_e[te], p2_te_100)
                )
            )
            rtr = StreamingSelectiveRouter(coh.contract, coh.per_instance_query_cost).fit(
                y_e[ca], p1_ca, p2_ca_15
            )
            q_a = rtr.estimate_voi(p1_te) > coh.per_instance_query_cost
            p_a = np.where(q_a, p2_te_15, p1_te)
            bt = paired_cluster_bootstrap_nb(
                y_e[te],
                p_a,
                q_a,
                p1_te,
                np.zeros_like(q_a),
                coh.contract,
                coh.per_instance_query_cost,
                n_boot=2000,
            )
            era_auto_gains.append(float(bt["delta_nb"]))

            # On Era 0 -> Era 3 test, evaluate CfBH under stale calibration vs recent burn-in
            if e_i == 0:
                stale_cal_p = p2_ca_15
                stale_cal_y = y_e[ca]
                stale_tr_Xb = pd.concat([Xb_e.iloc[tr], Z_e.iloc[tr][cols15]], axis=1)
                stale_tr_y = y_e[tr]
                stale_ca_Xb = pd.concat([Xb_e.iloc[ca], Z_e.iloc[ca][cols15]], axis=1)
                stale_cols15 = cols15
            elif e_i == 3:
                # Predict Era-0 model on Era-3 test vs local Era-3 burn-in calibration (n=2,500)
                _, p_era3_from_era0 = _fit_downstream_gbdt_worker(
                    stale_tr_Xb,
                    stale_tr_y,
                    stale_ca_Xb,
                    stale_cal_y,
                    pd.concat([Xb_e.iloc[te], Z_e.iloc[te][stale_cols15]], axis=1),
                    coh.base_semantic_types,
                    seed=2099,
                )
                _, p_era3_burnin_from_era0 = _fit_downstream_gbdt_worker(
                    stale_tr_Xb,
                    stale_tr_y,
                    stale_ca_Xb,
                    stale_cal_y,
                    pd.concat([Xb_e.iloc[ca[:2500]], Z_e.iloc[ca[:2500]][stale_cols15]], axis=1),
                    coh.base_semantic_types,
                    seed=2099,
                )
                tfdr = 1.0 - coh.contract.target_precision
                cf_stale = conformal_fdr_selection(
                    stale_cal_p, stale_cal_y, p_era3_from_era0, y_e[te], target_fdr=tfdr
                )
                cf_local = conformal_fdr_selection(
                    p_era3_burnin_from_era0,
                    y_e[ca[:2500]],
                    p_era3_from_era0,
                    y_e[te],
                    target_fdr=tfdr,
                )
                cal_lag = compute_calibration_summary(y_e[te], p_era3_from_era0)
                out_audit["cfbh_temporal_shift"][c_key] = {
                    "target_fdr": tfdr,
                    "era0_prev": float(np.mean(stale_cal_y)),
                    "era3_prev": float(np.mean(y_e[te])),
                    "lagged_citl": cal_lag.cal_intercept,
                    "lagged_slope": cal_lag.cal_slope,
                    "stale_cal_fdp": cf_stale.realized_fdp,
                    "stale_cal_alerts_per_100": 100.0 * cf_stale.selection_rate,
                    "stale_cal_recall": cf_stale.realized_recall,
                    "local_burnin_2500_fdp": cf_local.realized_fdp,
                    "local_burnin_2500_alerts_per_100": 100.0 * cf_local.selection_rate,
                    "local_burnin_2500_recall": cf_local.realized_recall,
                }

        out_audit["disjoint_4_eras"][c_key] = {
            "era_deltas_auprc_15_minus_100": era_deltas_15_100,
            "mean_delta_auprc": float(np.mean(era_deltas_15_100)),
            "era_auto_nb_gains_vs_s1": era_auto_gains,
            "mean_auto_nb_gain": float(np.mean(era_auto_gains)),
        }

    return out_audit


def run_block_3(max_workers: int = 4) -> dict[str, Any]:
    out_file = PARTIAL_DIR / "block3_canonical_holdouts_and_covert.json"
    if out_file.exists():
        print("Loaded cached Block 3:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    print("[Block 3] Validating Covert-AFA synthetic oracle + 6 full canonical holdouts...")
    syn_oracle = validate_covert_afa_synthetic_oracle(n_samples=4000, seed=2026)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        coh_results = list(ex.map(_evaluate_canonical_cohort_worker, list(COHORT_MCIDS.keys())))

    chrono_audit = _run_disjoint_chrono_and_cfbh_shift_audit()

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "covert_afa_synthetic_oracle_n4000": syn_oracle,
        "canonical_holdouts": {r["cohort_key"]: r for r in coh_results},
        "disjoint_chrono_and_cfbh_shift": chrono_audit,
    }
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 3] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


# ===========================================================================
# BLOCK 4: Clinical Case Study Deep Audit on UCI Diabetes 130-US
#          (R=20 Estimator Triad, Exchange Rates, Skipped Bounds, Subgroups, Sensitivity)
# ===========================================================================


def run_block_4() -> dict[str, Any]:
    out_file = PARTIAL_DIR / "block4_clinical_diabetes_audit.json"
    if out_file.exists():
        print("Loaded cached Block 4:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    print("[Block 4] Running UCI Diabetes R=20 Triad, Exchange Rates, Subgroups, & Sensitivity...")
    coh = load_fine_diabetes_discharge(include_stress_channels=False, seed=2026)
    coh_rb = load_fine_diabetes_discharge(
        include_stress_channels=False, include_race_in_base=False, seed=2026
    )

    # Outer locked patient-disjoint test cohort (20% of patients untouched by R=20 inner splits)
    split_df = coh.X_base.copy()
    split_df["__y__"] = coh.y
    split_df["__grp__"] = coh.group_ids
    outer_sp = freeze_split(
        split_df,
        "__y__",
        group_col="__grp__",
        train_frac=0.65,
        cal_frac=0.075,
        val_frac=0.075,
        test_frac=0.20,
        seed=2026,
    )
    dev_idx = np.sort(
        np.concatenate([outer_sp.train_idx, outer_sp.cal_idx, outer_sp.val_idx]).astype(int)
    )
    te_idx = np.asarray(outer_sp.test_idx, dtype=int)

    Xb_dev = coh.X_base.iloc[dev_idx].reset_index(drop=True)
    Z_dev = coh.Z_pool.iloc[dev_idx].reset_index(drop=True)
    y_dev = coh.y[dev_idx]
    grp_dev = coh.group_ids[dev_idx]

    Xb_te = coh.X_base.iloc[te_idx].reset_index(drop=True)
    Z_te = coh.Z_pool.iloc[te_idx].reset_index(drop=True)
    y_te = coh.y[te_idx]
    grp_te = coh.group_ids[te_idx]

    # R = 20 patient-grouped inner splits of dev_idx evaluated on locked te_idx
    n_r = 20
    p1_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    p2_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    p_route_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    q_route_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    p_unc_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    q_unc_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    p_top20_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    q_top20_te_mat = np.zeros((n_r, te_idx.size), dtype=float)

    cal_p1_pool: list[np.ndarray] = []
    cal_p2_pool: list[np.ndarray] = []
    cal_y_pool: list[np.ndarray] = []
    per_split_nbs: dict[str, list[float]] = {
        "stage1": [],
        "gh_voi_auto": [],
        "unc_voi_auto": [],
        "full_100": [],
    }
    sel_freq: dict[str, int] = {}

    dev_df = Xb_dev.copy()
    dev_df["__y__"] = y_dev
    dev_df["__grp__"] = grp_dev

    for r_i in range(n_r):
        r_seed = 3000 + r_i
        isp = freeze_split(
            dev_df,
            "__y__",
            group_col="__grp__",
            train_frac=0.72,
            cal_frac=0.20,
            val_frac=0.04,
            test_frac=0.04,
            seed=r_seed,
        )
        itr = np.asarray(isp.train_idx, dtype=int)
        ica = np.concatenate([isp.cal_idx, isp.val_idx, isp.test_idx]).astype(int)

        oof_b, val_m = _fit_oof_stage1_worker(
            Xb_dev.iloc[itr].reset_index(drop=True),
            y_dev[itr],
            coh.base_semantic_types,
            grp_dev[itr],
            None,
            seed=r_seed,
            n_jobs=4,
        )
        pil = sample_two_phase_pilot(
            y_dev[itr], oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=r_seed
        )
        sel = discover_sequential_conditional_gain(
            Xb_dev.iloc[itr].reset_index(drop=True),
            Z_dev.iloc[itr].reset_index(drop=True),
            y_dev[itr],
            coh.candidates,
            oof_b,
            pil,
            max_budget=coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_dev[itr],
            seed=r_seed,
        )
        for cid in sel.selected_ids:
            sel_freq[cid] = sel_freq.get(cid, 0) + 1
        scols = [c for ch in sel.selected_channels for c in ch.columns]

        p1_ca, p1_t = _fit_downstream_gbdt_worker(
            Xb_dev.iloc[itr],
            y_dev[itr],
            Xb_dev.iloc[ica],
            y_dev[ica],
            Xb_te,
            coh.base_semantic_types,
            seed=r_seed,
            n_jobs=4,
        )
        p2_ca, p2_t = _fit_downstream_gbdt_worker(
            pd.concat([Xb_dev.iloc[itr], Z_dev.iloc[itr][scols]], axis=1)
            if scols
            else Xb_dev.iloc[itr],
            y_dev[itr],
            pd.concat([Xb_dev.iloc[ica], Z_dev.iloc[ica][scols]], axis=1)
            if scols
            else Xb_dev.iloc[ica],
            y_dev[ica],
            pd.concat([Xb_te, Z_te[scols]], axis=1) if scols else Xb_te,
            coh.base_semantic_types,
            seed=r_seed + 50,
            n_jobs=4,
        )
        p1_te_mat[r_i] = p1_t
        p2_te_mat[r_i] = p2_t
        cal_p1_pool.append(p1_ca)
        cal_p2_pool.append(p2_ca)
        cal_y_pool.append(y_dev[ica])

        r_gh = StreamingSelectiveRouter(
            coh.contract, coh.per_instance_query_cost, mean_centered=True
        ).fit(y_dev[ica], p1_ca, p2_ca)
        r_un = StreamingSelectiveRouter(
            coh.contract, coh.per_instance_query_cost, mean_centered=False
        ).fit(y_dev[ica], p1_ca, p2_ca)
        q_gh = r_gh.estimate_voi(p1_t) > coh.per_instance_query_cost
        q_un = r_un.estimate_voi(p1_t) > coh.per_instance_query_cost
        p_gh = np.where(q_gh, p2_t, p1_t)
        p_un = np.where(q_un, p2_t, p1_t)

        rng_u = np.random.default_rng(r_seed + 9)
        _, q_t20, _, _ = calibrated_randomized_threshold_masks(
            p1_ca, p1_t, 0.20, rng_u.uniform(size=p1_ca.size), rng_u.uniform(size=p1_t.size)
        )
        p_route_te_mat[r_i] = p_gh
        q_route_te_mat[r_i] = q_gh.astype(float)
        p_unc_te_mat[r_i] = p_un
        q_unc_te_mat[r_i] = q_un.astype(float)
        p_top20_te_mat[r_i] = np.where(q_t20, p2_t, p1_t)
        q_top20_te_mat[r_i] = q_t20.astype(float)

        cfp, cfn = coh.contract.cost_fp, coh.contract.cost_fn
        tau = cfp / (cfp + cfn)

        def _nb_single(
            pa: np.ndarray,
            qa: np.ndarray,
            _tau: float = tau,
            _cfp: float = cfp,
            _cfn: float = cfn,
        ) -> float:
            return float(
                1000.0
                * np.mean(
                    y_te * (pa >= _tau)
                    - (_cfp / _cfn) * (1.0 - y_te) * (pa >= _tau)
                    - (coh.per_instance_query_cost / _cfn) * qa
                )
            )

        per_split_nbs["stage1"].append(_nb_single(p1_t, np.zeros_like(p1_t)))
        per_split_nbs["gh_voi_auto"].append(_nb_single(p_gh, q_gh.astype(float)))
        per_split_nbs["unc_voi_auto"].append(_nb_single(p_un, q_un.astype(float)))
        per_split_nbs["full_100"].append(_nb_single(p2_t, np.ones_like(p2_t)))

    # 1. Estimator Triad:
    # (a) Bagged VOI (Primary Deployable Single-Draw Binary Decision Q_bag in {0, 1})
    p1_bag = np.mean(p1_te_mat, axis=0)
    p2_bag = np.mean(p2_te_mat, axis=0)
    bag_router_gh = StreamingSelectiveRouter(
        coh.contract, coh.per_instance_query_cost, mean_centered=True
    ).fit(np.concatenate(cal_y_pool), np.concatenate(cal_p1_pool), np.concatenate(cal_p2_pool))
    bag_router_emp = StreamingSelectiveRouter(
        coh.contract,
        coh.per_instance_query_cost,
        quadrature_mode="empirical_quantile",
        n_emp_quantiles=33,
        mean_centered=True,
    ).fit(np.concatenate(cal_y_pool), np.concatenate(cal_p1_pool), np.concatenate(cal_p2_pool))
    bag_router_unc = StreamingSelectiveRouter(
        coh.contract, coh.per_instance_query_cost, mean_centered=False
    ).fit(np.concatenate(cal_y_pool), np.concatenate(cal_p1_pool), np.concatenate(cal_p2_pool))

    q_bag_gh = bag_router_gh.estimate_voi(p1_bag) > coh.per_instance_query_cost
    p_bag_gh = np.where(q_bag_gh, p2_bag, p1_bag)
    q_bag_emp = bag_router_emp.estimate_voi(p1_bag) > coh.per_instance_query_cost
    p_bag_emp = np.where(q_bag_emp, p2_bag, p1_bag)
    q_bag_unc = bag_router_unc.estimate_voi(p1_bag) > coh.per_instance_query_cost
    p_bag_unc = np.where(q_bag_unc, p2_bag, p1_bag)

    # Also 20%-budget Bagged VOI vs 20%-budget Top-Risk
    rng_b = np.random.default_rng(2026)
    u_cp = rng_b.uniform(size=bag_router_gh.voi_cal_.size)
    u_tb = rng_b.uniform(size=p1_bag.size)
    _, q_bag_gh20, _, _ = calibrated_randomized_threshold_masks(
        bag_router_gh.voi_cal_, bag_router_gh.estimate_voi(p1_bag), 0.20, u_cp, u_tb
    )
    _, q_bag_top20, _, _ = calibrated_randomized_threshold_masks(
        np.concatenate(cal_p1_pool), p1_bag, 0.20, u_cp, u_tb
    )
    p_bag_gh20 = np.where(q_bag_gh20, p2_bag, p1_bag)
    p_bag_top20 = np.where(q_bag_top20, p2_bag, p1_bag)

    # (b) Symmetric Expected Policy (Sym. Ens., Q_bar in [0, 1])
    p_sym_gh = np.mean(p_route_te_mat, axis=0)
    q_sym_gh = np.mean(q_route_te_mat, axis=0)
    p_sym_unc = np.mean(p_unc_te_mat, axis=0)
    q_sym_unc = np.mean(q_unc_te_mat, axis=0)

    q_zero = np.zeros(te_idx.size, dtype=float)
    q_ones = np.ones(te_idx.size, dtype=float)

    triad_summary = {
        "n_test_encounters": int(te_idx.size),
        "n_test_patients": int(np.unique(grp_te).size),
        "prevalence_test": float(np.mean(y_te)),
        "selected_channel_counts_r20": sel_freq,
        "bagged_voi_primary": {
            "gh_auto_vs_stage1": paired_cluster_bootstrap_nb(
                y_te,
                p_bag_gh,
                q_bag_gh,
                p1_bag,
                q_zero,
                coh.contract,
                coh.per_instance_query_cost,
                grp_te,
            ),
            "gh_auto_vs_uncentered": paired_cluster_bootstrap_nb(
                y_te,
                p_bag_gh,
                q_bag_gh,
                p_bag_unc,
                q_bag_unc,
                coh.contract,
                coh.per_instance_query_cost,
                grp_te,
            ),
            "gh_auto_vs_full100": paired_cluster_bootstrap_nb(
                y_te,
                p_bag_gh,
                q_bag_gh,
                p2_bag,
                q_ones,
                coh.contract,
                coh.per_instance_query_cost,
                grp_te,
            ),
            "emp_auto_vs_gh_auto": paired_cluster_bootstrap_nb(
                y_te,
                p_bag_emp,
                q_bag_emp,
                p_bag_gh,
                q_bag_gh,
                coh.contract,
                coh.per_instance_query_cost,
                grp_te,
            ),
            "gh_20_vs_toprisk_20": paired_cluster_bootstrap_nb(
                y_te,
                p_bag_gh20,
                q_bag_gh20,
                p_bag_top20,
                q_bag_top20,
                coh.contract,
                coh.per_instance_query_cost,
                grp_te,
            ),
            "q_bag_gh_auto": float(np.mean(q_bag_gh)),
            "q_bag_emp_auto": float(np.mean(q_bag_emp)),
            "q_bag_unc_auto": float(np.mean(q_bag_unc)),
        },
        "symmetric_expected_policy": {
            "gh_sym_vs_stage1": paired_cluster_bootstrap_nb(
                y_te,
                p_sym_gh,
                q_sym_gh,
                p1_bag,
                q_zero,
                coh.contract,
                coh.per_instance_query_cost,
                grp_te,
            ),
            "gh_sym_vs_unc_sym": paired_cluster_bootstrap_nb(
                y_te,
                p_sym_gh,
                q_sym_gh,
                p_sym_unc,
                q_sym_unc,
                coh.contract,
                coh.per_instance_query_cost,
                grp_te,
            ),
            "q_sym_gh": float(np.mean(q_sym_gh)),
        },
        "per_split_unbagged": {
            k: {
                "mean": float(np.mean(v)),
                "se": float(np.std(v, ddof=1) / np.sqrt(len(v))),
            }
            for k, v in per_split_nbs.items()
        },
    }

    # 2. Clinical Referral Exchange-Rate Table across thresholds (Bagged VOI vs Stage-1)
    exchange_rows: list[dict[str, Any]] = []
    _, inv_pat = np.unique(grp_te, return_inverse=True)
    n_pat = int(np.max(inv_pat)) + 1
    cl_n = np.bincount(inv_pat, minlength=n_pat).astype(float)
    rng_ex = np.random.default_rng(2026)
    cnts_mat = rng_ex.multinomial(n_pat, np.full(n_pat, 1.0 / n_pat), size=10000)
    den_vec = np.maximum(1.0, cnts_mat @ cl_n)

    for tau_eval in (0.06, 1.0 / 9.0, 0.15, 0.20):
        cfn_implied = (1.0 - tau_eval) / tau_eval
        c_tmp = DecisionContract(
            task_name="Ex",
            target_precision=0.30,
            max_fpr_budget=0.05,
            cost_fp=1.0,
            cost_fn=cfn_implied,
        )
        r_tmp = StreamingSelectiveRouter(
            c_tmp, coh.per_instance_query_cost, mean_centered=True
        ).fit(np.concatenate(cal_y_pool), np.concatenate(cal_p1_pool), np.concatenate(cal_p2_pool))
        q_t = r_tmp.estimate_voi(p1_bag) > coh.per_instance_query_cost
        p_t = np.where(q_t, p2_bag, p1_bag)

        tp_s2 = 1000.0 * ((y_te == 1) & (p_t >= tau_eval)).astype(float)
        tp_s1 = 1000.0 * ((y_te == 1) & (p1_bag >= tau_eval)).astype(float)
        fp_s2 = 1000.0 * ((y_te == 0) & (p_t >= tau_eval)).astype(float)
        fp_s1 = 1000.0 * ((y_te == 0) & (p1_bag >= tau_eval)).astype(float)

        dtp_i = tp_s2 - tp_s1
        spared_fp_i = fp_s1 - fp_s2  # positive = false referrals spared!

        dtp_draws = (cnts_mat @ np.bincount(inv_pat, weights=dtp_i, minlength=n_pat)) / den_vec
        sp_draws = (cnts_mat @ np.bincount(inv_pat, weights=spared_fp_i, minlength=n_pat)) / den_vec

        def _dh_p(dr: np.ndarray) -> float:
            kn = int(np.sum(dr <= 0.0))
            kp = int(np.sum(dr >= 0.0))
            return min(1.0, 2.0 * (min(kn, kp) + 1.0) / 10001.0)

        bt_nb = paired_cluster_bootstrap_nb(
            y_te, p_t, q_t, p1_bag, q_zero, c_tmp, coh.per_instance_query_cost, grp_te
        )
        tp_mean = float(np.mean(tp_s2))
        fp_mean = float(np.mean(fp_s2))
        exchange_rows.append(
            {
                "tau": float(tau_eval),
                "breakeven_fp_per_tp": float(cfn_implied),
                "query_rate": float(np.mean(q_t)),
                "tp_per_1k_voi": tp_mean,
                "tp_per_1k_s1": float(np.mean(tp_s1)),
                "delta_tp_per_1k": float(np.mean(dtp_i)),
                "delta_tp_ci95": [
                    float(np.quantile(dtp_draws, 0.025)),
                    float(np.quantile(dtp_draws, 0.975)),
                ],
                "delta_tp_p_dh": _dh_p(dtp_draws),
                "fp_per_1k_voi": fp_mean,
                "fp_per_1k_s1": float(np.mean(fp_s1)),
                "ppv_voi": tp_mean / max(1e-6, tp_mean + fp_mean),
                "referrals_spared_per_1k": float(np.mean(spared_fp_i)),
                "referrals_spared_ci95": [
                    float(np.quantile(sp_draws, 0.025)),
                    float(np.quantile(sp_draws, 0.975)),
                ],
                "referrals_spared_p_dh": _dh_p(sp_draws),
                "delta_nb": bt_nb["delta_nb"],
                "delta_nb_ci95": [bt_nb["ci95_low"], bt_nb["ci95_high"]],
                "delta_nb_p_dh": bt_nb["p_dh"],
            }
        )

    # 3. Directional Skipped-Case (Q=0) Action-Disagreement Bounds at tau* = 1/9 (and q=20% budget)
    tau_star = 1.0 / 9.0
    skipped_audits: dict[str, Any] = {}
    for lbl, q_mask in (("autonomous_voi", q_bag_gh), ("budget_20pct_voi", q_bag_gh20)):
        sk = ~q_mask
        n_sk = int(np.sum(sk))
        n_pat_sk = int(np.unique(grp_te[sk]).size) if n_sk > 0 else 0
        a1_sk = p1_bag[sk] >= tau_star
        a2_sk = p2_bag[sk] >= tau_star
        y_sk = y_te[sk]
        flips = a1_sk != a2_sk
        neg_to_pos = (~a1_sk) & a2_sk
        pos_to_neg = a1_sk & (~a2_sk)
        # Beneficial vs harmful if Stage-2 had been queried on skipped patients
        ben = (neg_to_pos & (y_sk == 1)) | (pos_to_neg & (y_sk == 0))
        harm = (neg_to_pos & (y_sk == 0)) | (pos_to_neg & (y_sk == 1))
        k_fl = int(np.sum(flips))
        k_np = int(np.sum(neg_to_pos))
        k_pn = int(np.sum(pos_to_neg))
        skipped_audits[lbl] = {
            "n_skipped": n_sk,
            "n_skipped_patients": n_pat_sk,
            "flip_rate": float(k_fl) / max(1, n_sk),
            "flip_ub95_patient_conservative": clopper_pearson_upper_1sided(k_fl, max(1, n_pat_sk)),
            "neg_to_pos_rate": float(k_np) / max(1, n_sk),
            "neg_to_pos_ub95": clopper_pearson_upper_1sided(k_np, max(1, n_pat_sk)),
            "pos_to_neg_rate": float(k_pn) / max(1, n_sk),
            "pos_to_neg_ub95": clopper_pearson_upper_1sided(k_pn, max(1, n_pat_sk)),
            "would_be_beneficial_flips": int(np.sum(ben)),
            "would_be_harmful_flips": int(np.sum(harm)),
        }

    # 4. Subgroup Equity, Calibration (CITL, Slope, Smooth ICI), FNR/FPR Parity, & Race-Blind Check
    # Fit Race-Blind bagged comparison on identical outer split
    itr0 = np.asarray(
        freeze_split(dev_df, "__y__", group_col="__grp__", seed=3000).train_idx, dtype=int
    )
    ica0 = np.asarray(
        freeze_split(dev_df, "__y__", group_col="__grp__", seed=3000).cal_idx, dtype=int
    )
    scols_main = list(coh.candidates[0].columns)  # hie_prior_inpatient
    _, p2_rb_te = _fit_downstream_gbdt_worker(
        pd.concat(
            [
                coh_rb.X_base.iloc[dev_idx[itr0]].reset_index(drop=True),
                Z_dev.iloc[itr0][scols_main].reset_index(drop=True),
            ],
            axis=1,
        ),
        y_dev[itr0],
        pd.concat(
            [
                coh_rb.X_base.iloc[dev_idx[ica0]].reset_index(drop=True),
                Z_dev.iloc[ica0][scols_main].reset_index(drop=True),
            ],
            axis=1,
        ),
        y_dev[ica0],
        pd.concat(
            [
                coh_rb.X_base.iloc[te_idx].reset_index(drop=True),
                Z_te[scols_main].reset_index(drop=True),
            ],
            axis=1,
        ),
        coh_rb.base_semantic_types,
        seed=3099,
        n_jobs=4,
    )

    subgroup_audit: dict[str, Any] = {}
    for dim_name, sg_full in coh.fairness_subgroups.items():
        sg_te = sg_full[te_idx]
        strata_rows: list[dict[str, Any]] = []
        d_means: list[float] = []
        d_vars: list[float] = []
        for s_val in np.unique(sg_te):
            m = sg_te == s_val
            if int(np.sum(m)) < 50:
                continue
            ym = y_te[m]
            p1m = p1_bag[m]
            pm = p_bag_gh[m]
            qm = q_bag_gh[m]
            gm = grp_te[m]
            cal_m = compute_calibration_summary(ym, pm)
            bt_m = paired_cluster_bootstrap_nb(
                ym,
                pm,
                qm,
                p1m,
                np.zeros_like(qm, dtype=float),
                coh.contract,
                coh.per_instance_query_cost,
                gm,
                n_boot=10000,
                seed=4100,
            )
            se_est = (bt_m["ci95_high"] - bt_m["ci95_low"]) / 3.92
            d_means.append(bt_m["delta_nb"])
            d_vars.append(max(1e-6, se_est**2))

            act_m = pm >= tau_star
            pos_cnt = max(1, int(np.sum(ym == 1)))
            neg_cnt = max(1, int(np.sum(ym == 0)))
            fnr = float(np.sum((ym == 1) & (~act_m))) / float(pos_cnt)
            fpr = float(np.sum((ym == 0) & act_m)) / float(neg_cnt)

            strata_rows.append(
                {
                    "stratum": str(s_val),
                    "n_enc": int(np.sum(m)),
                    "prevalence": float(np.mean(ym)),
                    "query_rate": float(np.mean(qm)),
                    "auprc_s1": float(average_precision_score(ym, p1m)),
                    "auprc_routed": float(average_precision_score(ym, pm)),
                    "auprc_race_blind_s2": float(average_precision_score(ym, p2_rb_te[m])),
                    "delta_nb_vs_s1": bt_m["delta_nb"],
                    "ci95_low": bt_m["ci95_low"],
                    "ci95_high": bt_m["ci95_high"],
                    "p_dh": bt_m["p_dh"],
                    "citl": cal_m.cal_intercept,
                    "slope": cal_m.cal_slope,
                    "ici": cal_m.ici,
                    "ece": cal_m.ece,
                    "fnr_at_tau": fnr,
                    "fpr_at_tau": fpr,
                }
            )
        # Cochran's Q test for subgroup Delta-NB heterogeneity
        if len(d_means) >= 2:
            w_iv = 1.0 / np.asarray(d_vars)
            mu_bar = float(np.sum(w_iv * np.asarray(d_means)) / np.sum(w_iv))
            q_stat = float(np.sum(w_iv * ((np.asarray(d_means) - mu_bar) ** 2)))
            p_het = float(chi2.sf(q_stat, df=len(d_means) - 1))
        else:
            q_stat, p_het = 0.0, 1.0
        subgroup_audit[dim_name] = {
            "cochran_q": q_stat,
            "p_heterogeneity": p_het,
            "strata": strata_rows,
        }

    # 5. Hyperparameter Sensitivity Sweep (B_score in {4, 8, 12}, n_voi_bins in {8, 12, 20})
    sens_rows: list[dict[str, Any]] = []
    oof_b0, val_m0 = _fit_oof_stage1_worker(
        Xb_dev.iloc[itr0].reset_index(drop=True),
        y_dev[itr0],
        coh.base_semantic_types,
        grp_dev[itr0],
        None,
        seed=3000,
        n_jobs=4,
    )
    for b_sc in (4, 8, 12):
        pil_s = sample_two_phase_pilot(
            y_dev[itr0],
            oof_b0,
            val_m0,
            pilot_fraction=0.15,
            design="neyman",
            n_score_bins=b_sc,
            seed=3000,
        )
        sel_s = discover_sequential_conditional_gain(
            Xb_dev.iloc[itr0].reset_index(drop=True),
            Z_dev.iloc[itr0].reset_index(drop=True),
            y_dev[itr0],
            coh.candidates,
            oof_b0,
            pil_s,
            max_budget=coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_dev[itr0],
            seed=3000,
        )
        sens_rows.append(
            {
                "parameter": "n_score_bins",
                "value": b_sc,
                "selected_ids": sel_s.selected_ids,
                "spent": sel_s.total_cost_spent,
            }
        )
    for n_vb in (8, 12, 20):
        r_vb = StreamingSelectiveRouter(
            coh.contract, coh.per_instance_query_cost, n_voi_bins=n_vb, mean_centered=True
        ).fit(np.concatenate(cal_y_pool), np.concatenate(cal_p1_pool), np.concatenate(cal_p2_pool))
        qv = r_vb.estimate_voi(p1_bag) > coh.per_instance_query_cost
        pv = np.where(qv, p2_bag, p1_bag)
        btv = paired_cluster_bootstrap_nb(
            y_te,
            pv,
            qv,
            p1_bag,
            q_zero,
            coh.contract,
            coh.per_instance_query_cost,
            grp_te,
            n_boot=2000,
        )
        sens_rows.append(
            {
                "parameter": "n_voi_bins",
                "value": n_vb,
                "nb": btv["nb_num"],
                "delta_nb_vs_s1": btv["delta_nb"],
                "query_rate": float(np.mean(qv)),
            }
        )

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "r20_estimator_triad": triad_summary,
        "clinical_exchange_rates": exchange_rows,
        "skipped_case_disagreement_bounds": skipped_audits,
        "subgroup_equity_and_calibration": subgroup_audit,
        "hyperparameter_sensitivity": sens_rows,
    }
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 4] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


def main() -> None:
    t_start = time.time()
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    b1 = run_block_1(max_workers=8)
    b2 = run_block_2(max_workers=8)
    b3 = run_block_3(max_workers=3)
    b4 = run_block_4()

    master = {
        "version": "2.1.0",
        "v1_preprints_doi": "10.20944/preprints202611.2517.v1",
        "total_elapsed_sec": round(time.time() - t_start, 2),
        "block1_monte_carlo": b1,
        "block2_stress_and_ablations": b2,
        "block3_canonical_holdouts_and_covert": b3,
        "block4_clinical_diabetes_audit": b4,
    }
    OUT_JSON.write_text(json.dumps(master, indent=2), encoding="utf-8")
    el_s = master["total_elapsed_sec"]
    print(f"\nALL PAPER 1 v2.1 REMEDIATION BLOCKS COMPLETE IN {el_s}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
