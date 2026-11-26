"""Paper 1 (imboreceipt v2.2.0) Complete Pre-Submission Remediation & Verification Suite.

Executes all 5 modular verification blocks addressing Findings F1-F11 and all 5 Proposed Checks
from the independent editorial/biostatistical review of v2.1:

  Block 1 (Weak-Signal Sim A, BCC RMSE Ratios, & Twin-Geometry Sim C):
    - Weak-Signal Sim A1 (1,000 reps, N=6,000, rho=15% Neyman, 1 weak signal near Holm boundary
      + 8 nulls) and Sim A2 (500 reps, 2 nearly-tied weak signals where path order flips + 8 nulls),
      comparing 4-fold CV heuristic vs Theorem 1's disjoint 50/50 split-pilot (multi-step FWER,
      power, and adaptive CI coverage when signal separation fails; closes F2 & Check #3).
    - Sim B exact RMSE reductions vs both SRS and Balanced Case-Control (BCC; closes F10).
    - Sim C (200 reps, unit costs c_m=1.0, B=2.0) across BOTH (i) 1-column asymmetric twin
      Z_2 = Z_1 + eps (showing greedy root concentration in full-pool cpi_pilot vs 100% twin
      duplication in marginal baseline-stratified permutation) and (ii) multi-column twin channels
      matching real cohorts (showing off-manifold twin duplication in cpi_pilot vs 0% twin waste
      in Sequential Forward CV and 50/50 Split-Pilot; closes F1).

  Block 2 (Full 120-Split Leak-Free Stress-Pool & Native Verification Campaign):
    - Runs ALL R=20 splits across ALL 6 cohorts (120 splits total) on leak-free stress pools
      fitted strictly on train_idx, reconciling NeurIPS BAF and UCI Taiwan 5-split vs 20-split
      means and verifying native Table-2 leak-free invariance (closes F6 & Check #1).

  Block 3 (Stage-B Boundary-Uncertainty, Full-Query, c_query Sweep, & Covert-AFA Variants):
    - Evaluates all 6 full unsampled canonical holdouts against Full Stage-2 Querying (100%) and
      Boundary Uncertainty (-|p1 - tau*|) at BOTH q=20% and exact matched q=q_auto (B=10,000 paired
      cluster/instance bootstraps + 20-block temporal bootstrap on IEEE-CIS alongside card1).
    - Sweeps c_query in {0.00, 0.01, 0.04, 0.08, 0.16} across all 6 cohorts.
    - Evaluates CovertGreedyCMIAFA under flat_any vs proportional_cost fee accounting and
      decision_boundary_weighted=True (DB-CMI; closes F4, F9, F11, & Check #2).

  Block 4 (100-Replication Conformal FDR Shift vs Jin-Candes WCS & Local Burn-In):
    - Evaluates R=100 repeated block/stratified shift replications on IEEE-CIS and NeurIPS BAF
      comparing In-Era Exchangeable CfBH, Stale Era-0 Unweighted CfBH, Unlabeled Covariate-Shift
      Weighted Conformalized Selection (WCS; Jin & Candes, 2025), and Labeled Local Burn-In
      (n=2,500), with means, SEs, and 95% CIs (closes F8 & Check #4).

  Block 5 (Simulated Acquisition Scenario on Race-Blind UCI Diabetes across Both Holdouts):
    - Defaults to Race-Blind X_base (with Race-Included sensitivity), reconciling BOTH the
      canonical 55/15/15/15 split (N_test=14,909) and the outer 80/20 R=20 bagged split
      (N_test=19,999), with VOI vs Full 100%, VOI vs Boundary Uncertainty (matched q & 20%),
      c_query sweep, 3 skipped-case disagreement bounds (encounter CP, patient-cluster bootstrap,
      and conservative unique-patient CP), referral exchange table, and full subgroup audit
      including Hispanic AUPRC trajectories (closes F3, F5, F11, & Check #5).
"""

from __future__ import annotations

import os

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
from run_paper1_v2_1_remediation import (
    COHORT_MCIDS,
    OUT_DIR,
    _fit_downstream_gbdt_worker,
    _fit_oof_stage1_worker,
    _load_full_canonical_cohort,
    _load_native_cohort,
    _spurious_rates,
    clopper_pearson_upper_1sided,
    paired_cluster_bootstrap_nb,
    prop_summary,
)
from scipy.special import expit, logit
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from imboreceipt.bench.fine_grained_cohorts import (
    load_fine_diabetes_discharge,
    materialize_leak_free_stress_pool,
)
from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.bootstrap import holm_bonferroni_adjust
from imboreceipt.core.types import freeze_split
from imboreceipt.discover.selector import CandidateChannelSpec
from imboreceipt.discover.strata import make_leak_free_cv_splits
from imboreceipt.discover.two_phase import (
    discover_competitor_method,
    discover_sequential_conditional_gain,
    discover_sequential_split_pilot,
    evaluate_split_pilot_step1_inference,
    sample_two_phase_pilot,
)
from imboreceipt.frontier.baselines import TabularPreprocessor
from imboreceipt.lattice.covert_afa import CovertGreedyCMIAFA
from imboreceipt.lattice.streaming_routing import (
    StreamingSelectiveRouter,
    calibrated_randomized_threshold_masks,
    compute_calibration_summary,
    conformal_fdr_selection,
    weighted_conformal_fdr_selection,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON_V22 = OUT_DIR / "paper1_v2_2_remediation_results.json"
PARTIAL_V22 = OUT_DIR / "v2_2_blocks"
V21_BLOCKS = OUT_DIR / "v2_1_blocks"


# ===========================================================================
# BLOCK 1: Weak-Signal Sim A (A1 + A2), Sim B vs BCC, & Twin-Geometry Sim C
# ===========================================================================


def _weak_sim_a1_one_rep(rep: int) -> dict[str, Any]:
    """1 weak signal near Holm selection boundary (power ~55-65%) + 8 pure nulls."""
    r_seed = 11000 + rep
    rng = np.random.default_rng(r_seed)
    n = 6000
    x = rng.normal(0.0, 1.0, (n, 3))
    z1 = rng.normal(0.0, 1.0, n)
    # beta_1 = 0.54 -> Step-1 Z-stat hovers around 2.3-2.9 vs Holm cutoff 2.54!
    eta = -3.4 + 0.9 * x[:, 0] - 0.5 * x[:, 1] + 0.54 * z1
    y = rng.binomial(1, expit(eta))

    X_b = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    z_cols: dict[str, np.ndarray] = {"z_true": z1.astype(np.float32)}
    cands = [CandidateChannelSpec("ch_true", ("z_true",), 1.0, "true")]
    for j in range(1, 9):
        z_cols[f"z_null_{j}"] = rng.normal(0.0, 1.0, n).astype(np.float32)
        cands.append(CandidateChannelSpec(f"ch_null_{j}", (f"z_null_{j}",), 1.0, "null"))
    Z_df = pd.DataFrame(z_cols)

    oof_b, val_m = _fit_oof_stage1_worker(X_b, y, {}, None, None, seed=r_seed, n_jobs=1)
    pilot = sample_two_phase_pilot(
        y, oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=r_seed
    )

    cen_g = np.nan
    if rep < 200:
        pf = sample_two_phase_pilot(y, oof_b, val_m, pilot_fraction=1.0, design="full")
        rc = discover_sequential_conditional_gain(
            X_b, Z_df, y, [cands[0]], oof_b, pf, max_budget=1.0, seed=r_seed
        )
        cen_g = float(rc.initial_scores["ch_true"])

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
    res_sp_seq = discover_sequential_split_pilot(
        X_b, Z_df, y, cands, oof_b, pilot, max_budget=3.0, alpha_select=0.05, seed=r_seed
    )
    res_sp_s1 = evaluate_split_pilot_step1_inference(
        X_b, Z_df, y, [cands[0]], oof_b, pilot, alpha_select=0.05, seed=r_seed
    )
    gs = float(res_sp_s1["scores"]["ch_true"])
    ses = float(res_sp_s1["ses_total"]["ch_true"])
    ts = float(res_sp_s1["census_targets_conditional"]["ch_true"])

    return {
        "cen_g": cen_g,
        "cv_g": float(res_cv.initial_scores["ch_true"]),
        "cv_se": float(res_cv.initial_ses["ch_true"]),
        "cv_power": int("ch_true" in res_cv.selected_ids),
        "cv_null_fwer": int(any(cid.startswith("ch_null_") for cid in res_cv.selected_ids)),
        "sp_seq_power": int("ch_true" in res_sp_seq.selected_ids),
        "sp_seq_null_fwer": int(any(cid.startswith("ch_null_") for cid in res_sp_seq.selected_ids)),
        "sp_cov_cond": int(abs(gs - ts) <= 1.96 * ses),
    }


def _weak_sim_a2_competing_one_rep(rep: int) -> dict[str, Any]:
    """2 nearly-tied weak signals (beta_1=0.60, beta_2=0.58; random Step-1 winner) + 8 nulls."""
    r_seed = 15000 + rep
    rng = np.random.default_rng(r_seed)
    n = 6000
    x = rng.normal(0.0, 1.0, (n, 3))
    z1 = rng.normal(0.0, 1.0, n)
    z2 = rng.normal(0.0, 1.0, n)
    eta = -3.4 + 0.9 * x[:, 0] - 0.5 * x[:, 1] + 0.60 * z1 + 0.58 * z2
    y = rng.binomial(1, expit(eta))

    X_b = pd.DataFrame(x, columns=["x0", "x1", "x2"])
    z_cols: dict[str, np.ndarray] = {
        "z_w1": z1.astype(np.float32),
        "z_w2": z2.astype(np.float32),
    }
    cands = [
        CandidateChannelSpec("ch_w1", ("z_w1",), 1.0, "true"),
        CandidateChannelSpec("ch_w2", ("z_w2",), 1.0, "true"),
    ]
    for j in range(1, 9):
        z_cols[f"z_null_{j}"] = rng.normal(0.0, 1.0, n).astype(np.float32)
        cands.append(CandidateChannelSpec(f"ch_null_{j}", (f"z_null_{j}",), 1.0, "null"))
    Z_df = pd.DataFrame(z_cols)

    oof_b, val_m = _fit_oof_stage1_worker(X_b, y, {}, None, None, seed=r_seed, n_jobs=1)
    pilot = sample_two_phase_pilot(
        y, oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=r_seed
    )

    res_cv = discover_sequential_conditional_gain(
        X_b, Z_df, y, cands, oof_b, pilot, max_budget=3.0, alpha_select=0.05, seed=r_seed
    )
    res_sp = discover_sequential_split_pilot(
        X_b, Z_df, y, cands, oof_b, pilot, max_budget=3.0, alpha_select=0.05, seed=r_seed
    )
    first_cv = res_cv.selected_ids[0] if res_cv.selected_ids else "none"
    return {
        "cv_first_w1": int(first_cv == "ch_w1"),
        "cv_first_w2": int(first_cv == "ch_w2"),
        "cv_both_signals": int({"ch_w1", "ch_w2"}.issubset(set(res_cv.selected_ids))),
        "cv_any_signal": int(bool({"ch_w1", "ch_w2"} & set(res_cv.selected_ids))),
        "cv_null_fwer": int(any(cid.startswith("ch_null_") for cid in res_cv.selected_ids)),
        "sp_any_signal": int(bool({"ch_w1", "ch_w2"} & set(res_sp.selected_ids))),
        "sp_both_signals": int({"ch_w1", "ch_w2"}.issubset(set(res_sp.selected_ids))),
        "sp_null_fwer": int(any(cid.startswith("ch_null_") for cid in res_sp.selected_ids)),
    }


def _single_channel_perm_select(
    X_b: pd.DataFrame,
    Z_df: pd.DataFrame,
    y: np.ndarray,
    cands: list[CandidateChannelSpec],
    oof_b: np.ndarray,
    pilot: Any,
    max_budget: float,
    seed: int,
) -> list[str]:
    """Marginal per-channel baseline-stratified permutation: fit eta_m(X_base, Z_m) per channel."""
    idx = pilot.sampled_indices
    yp = y[idx]
    wp = pilot.ht_weights
    wp_n = wp * (float(wp.size) / max(1e-12, float(np.sum(wp))))
    init_p = logit(np.clip(oof_b[idx], 1e-12, 1.0 - 1e-12))
    q_edges = np.unique(np.quantile(oof_b[idx], np.linspace(0.0, 1.0, 6)))
    p_bins = (
        np.digitize(oof_b[idx], q_edges[1:-1], right=False)
        if q_edges.size > 2
        else np.zeros(yp.size, dtype=int)
    )
    cv = make_leak_free_cv_splits(yp, n_folds=4, seed=seed)
    rng = np.random.default_rng(seed + 91)

    scores: dict[str, float] = {}
    pvals: list[float] = []
    for c in cands:
        cols = list(c.columns)
        df_m = pd.concat(
            [X_b.iloc[idx].reset_index(drop=True), Z_df.iloc[idx][cols].reset_index(drop=True)],
            axis=1,
        )
        oof_f = np.zeros(yp.size, dtype=float)
        oof_perm = np.zeros(yp.size, dtype=float)
        for f_i, (tr, va) in enumerate(cv):
            prep = TabularPreprocessor()
            Xtr = prep.fit_transform(df_m.iloc[tr])
            Xva = prep.transform(df_m.iloc[va])
            clf = lgb.LGBMClassifier(
                n_estimators=55,
                learning_rate=0.05,
                num_leaves=15,
                max_depth=4,
                min_child_samples=max(8, min(25, int(tr.size // 22))),
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=seed + f_i,
                n_jobs=1,
                verbose=-1,
            )
            clf.fit(Xtr, yp[tr], sample_weight=wp_n[tr], init_score=init_p[tr])
            tr_raw = init_p[tr] + np.asarray(clf.predict(Xtr, raw_score=True), dtype=float)
            va_raw = init_p[va] + np.asarray(clf.predict(Xva, raw_score=True), dtype=float)
            cal = LogisticRegression(C=2.0, solver="lbfgs", max_iter=200)
            cal.fit(tr_raw.reshape(-1, 1), yp[tr], sample_weight=wp_n[tr])
            oof_f[va] = np.clip(cal.predict_proba(va_raw.reshape(-1, 1))[:, 1], 1e-12, 1.0 - 1e-12)

            df_va_p = df_m.iloc[va].copy()
            va_bins = p_bins[va]
            for col in cols:
                arr = df_va_p[col].to_numpy().copy()
                for b_id in np.unique(va_bins):
                    loc = np.flatnonzero(va_bins == b_id)
                    if loc.size > 1:
                        arr[loc] = arr[rng.permutation(loc)]
                df_va_p[col] = arr
            Xva_p = prep.transform(df_va_p)
            vap_raw = init_p[va] + np.asarray(clf.predict(Xva_p, raw_score=True), dtype=float)
            oof_perm[va] = np.clip(
                cal.predict_proba(vap_raw.reshape(-1, 1))[:, 1], 1e-12, 1.0 - 1e-12
            )

        ll_f = yp * np.log(oof_f) + (1 - yp) * np.log(1.0 - oof_f)
        ll_p = yp * np.log(oof_perm) + (1 - yp) * np.log(1.0 - oof_perm)
        psi = ll_f - ll_p
        g = float(np.sum(wp * psi) / np.sum(wp))
        se = float(np.std(psi, ddof=1) / np.sqrt(yp.size))
        scores[c.channel_id] = g
        pvals.append(float(norm.sf(g / max(1e-12, se))))

    adj = holm_bonferroni_adjust(pvals)
    ranked = sorted(
        [c for c, ap in zip(cands, adj, strict=True) if ap < 0.05 and scores[c.channel_id] > 0.0],
        key=lambda c: scores[c.channel_id] / c.acquisition_cost,
        reverse=True,
    )
    sel: list[str] = []
    spent = 0.0
    for c in ranked:
        if spent + c.acquisition_cost <= max_budget + 1e-9:
            sel.append(c.channel_id)
            spent += c.acquisition_cost
    return sel


def _sim_c_multicol_and_1col_extra_rep(rep: int) -> dict[str, Any]:
    """Evaluate:
    (a) Extra arms on Sim C1 (1-col asymmetric twin, exact seed 6000+rep): 50/50 split-pilot &
        per-channel baseline-stratified permutation.
    (b) Full 9-method suite on Sim C2 (multi-column symmetric twin channels sharing U1+U2,
        matching real multi-column stress channels in Table 3, unit costs c_m=1.0, B=2.0).
    """
    # --- Part A: Exact Sim C1 DGP (seed = 6000 + rep) ---
    s1 = 6000 + rep
    rng1 = np.random.default_rng(s1)
    n1 = 7000
    x1 = rng1.normal(0.0, 1.0, (n1, 3))
    za = rng1.normal(0.0, 1.0, n1)
    za_twin = za + rng1.normal(0.0, 0.18, n1)
    zb = rng1.normal(0.0, 1.0, n1)
    y1 = rng1.binomial(1, expit(-2.8 + 0.9 * x1[:, 0] - 0.5 * x1[:, 1] + 1.25 * za + 0.80 * zb))
    Xb1 = pd.DataFrame(x1, columns=["x0", "x1", "x2"])
    zd1: dict[str, np.ndarray] = {
        "za": za.astype(np.float32),
        "za_twin": za_twin.astype(np.float32),
        "zb": zb.astype(np.float32),
        "echo_1": (0.85 * x1[:, 0] + rng1.normal(0.0, 0.25, n1)).astype(np.float32),
        "echo_2": (0.65 * (x1[:, 0] ** 2) - 0.5 * x1[:, 1] + rng1.normal(0.0, 0.30, n1)).astype(
            np.float32
        ),
    }
    cands1 = [
        CandidateChannelSpec("ch_A", ("za",), 1.0, "mech_A"),
        CandidateChannelSpec("ch_A_twin", ("za_twin",), 1.0, "twin"),
        CandidateChannelSpec("ch_B", ("zb",), 1.0, "mech_B"),
        CandidateChannelSpec("ch_echo_1", ("echo_1",), 1.0, "echo"),
        CandidateChannelSpec("ch_echo_2", ("echo_2",), 1.0, "echo"),
    ]
    for k in range(1, 12):
        zd1[f"null_{k}"] = rng1.normal(0.0, 1.0, n1).astype(np.float32)
        cands1.append(CandidateChannelSpec(f"ch_null_{k}", (f"null_{k}",), 1.0, "null"))
    Z1 = pd.DataFrame(zd1)
    oof1, vm1 = _fit_oof_stage1_worker(Xb1, y1, {}, None, None, seed=s1, n_jobs=1)
    pil1 = sample_two_phase_pilot(y1, oof1, vm1, pilot_fraction=0.18, design="neyman", seed=s1)

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

    c1_sp = discover_sequential_split_pilot(
        Xb1, Z1, y1, cands1, oof1, pil1, max_budget=2.0, seed=s1
    )
    c1_sperm = _single_channel_perm_select(Xb1, Z1, y1, cands1, oof1, pil1, max_budget=2.0, seed=s1)

    # --- Part B: Sim C2 Multi-Column Symmetric Twin Channels (seed = 7000 + rep) ---
    s2 = 7000 + rep
    rng2 = np.random.default_rng(s2)
    n2 = 7000
    x2 = rng2.normal(0.0, 1.0, (n2, 3))
    u1 = rng2.normal(0.0, 1.0, n2)
    u2 = rng2.normal(0.0, 1.0, n2)
    z1a = u1 + rng2.normal(0.0, 0.15, n2)
    z1b = u2 + rng2.normal(0.0, 0.35, n2)
    z2a = u1 + rng2.normal(0.0, 0.35, n2)
    z2b = u2 + rng2.normal(0.0, 0.15, n2)
    zb2 = rng2.normal(0.0, 1.0, n2)
    y2 = rng2.binomial(
        1, expit(-2.4 + 0.9 * x2[:, 0] - 0.5 * x2[:, 1] + 2.15 * (u1 + u2) + 1.05 * zb2)
    )
    Xb2 = pd.DataFrame(x2, columns=["x0", "x1", "x2"])
    zd2: dict[str, np.ndarray] = {
        "z1a": z1a.astype(np.float32),
        "z1b": z1b.astype(np.float32),
        "z2a": z2a.astype(np.float32),
        "z2b": z2b.astype(np.float32),
        "zb": zb2.astype(np.float32),
        "echo_1": (0.85 * x2[:, 0] + rng2.normal(0.0, 0.25, n2)).astype(np.float32),
        "echo_2": (0.65 * (x2[:, 0] ** 2) - 0.5 * x2[:, 1] + rng2.normal(0.0, 0.30, n2)).astype(
            np.float32
        ),
    }
    cands2 = [
        CandidateChannelSpec("ch_A", ("z1a", "z1b"), 1.0, "mech_A"),
        CandidateChannelSpec("ch_A_twin", ("z2a", "z2b"), 1.0, "twin"),
        CandidateChannelSpec("ch_B", ("zb",), 1.0, "mech_B"),
        CandidateChannelSpec("ch_echo_1", ("echo_1",), 1.0, "echo"),
        CandidateChannelSpec("ch_echo_2", ("echo_2",), 1.0, "echo"),
    ]
    for k in range(1, 12):
        zd2[f"null_{k}"] = rng2.normal(0.0, 1.0, n2).astype(np.float32)
        cands2.append(CandidateChannelSpec(f"ch_null_{k}", (f"null_{k}",), 1.0, "null"))
    Z2 = pd.DataFrame(zd2)

    oof2, vm2 = _fit_oof_stage1_worker(Xb2, y2, {}, None, None, seed=s2, n_jobs=1)
    pil2 = sample_two_phase_pilot(y2, oof2, vm2, pilot_fraction=0.18, design="neyman", seed=s2)

    r2_seq = discover_sequential_conditional_gain(
        Xb2, Z2, y2, cands2, oof2, pil2, max_budget=2.0, seed=s2
    )
    r2_sp = discover_sequential_split_pilot(
        Xb2, Z2, y2, cands2, oof2, pil2, max_budget=2.0, seed=s2
    )
    r2_one = discover_sequential_conditional_gain(
        Xb2, Z2, y2, cands2, oof2, pil2, max_budget=2.0, sequential=False, seed=s2
    )
    r2_cpi = discover_competitor_method(
        "cpi_pilot", Xb2, Z2, y2, cands2, oof2, pil2, max_budget=2.0, seed=s2
    )
    r2_sperm = _single_channel_perm_select(Xb2, Z2, y2, cands2, oof2, pil2, max_budget=2.0, seed=s2)
    r2_cham = discover_competitor_method(
        "chamma_residual_cpi_pilot", Xb2, Z2, y2, cands2, oof2, pil2, max_budget=2.0, seed=s2
    )
    r2_loco = discover_competitor_method(
        "loco_pilot", Xb2, Z2, y2, cands2, oof2, pil2, max_budget=2.0, seed=s2
    )
    r2_marg = discover_competitor_method(
        "global_mi", Xb2, Z2, y2, cands2, oof2, pil2, max_budget=2.0, seed=s2
    )

    return {
        "c1_split_pilot": _grade(c1_sp.selected_ids),
        "c1_single_ch_perm": _grade(c1_sperm),
        "c2_seq_cv": _grade(r2_seq.selected_ids),
        "c2_seq_split_pilot": _grade(r2_sp.selected_ids),
        "c2_oneshot_cmi": _grade(r2_one.selected_ids),
        "c2_fullpool_cpi_pilot": _grade(r2_cpi.selected_ids),
        "c2_single_ch_perm": _grade(r2_sperm),
        "c2_chamma_cpi": _grade(r2_cham.selected_ids),
        "c2_loco": _grade(r2_loco.selected_ids),
        "c2_marginal_mi": _grade(r2_marg.selected_ids),
    }


def run_block_1_v22(max_workers: int = 8) -> dict[str, Any]:
    out_file = PARTIAL_V22 / "block1_v2_2_weak_signal_and_twins.json"
    if out_file.exists():
        print("Loaded cached Block 1 v2.2:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    print("[Block 1 v2.2] Running 1,000 Weak-Sim-A1 + 500 Competing-Sim-A2 + 200 Twin-Sim-C...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        a1_rows = list(ex.map(_weak_sim_a1_one_rep, range(1000), chunksize=10))
        a2_rows = list(ex.map(_weak_sim_a2_competing_one_rep, range(500), chunksize=10))
        c_rows = list(ex.map(_sim_c_multicol_and_1col_extra_rep, range(200), chunksize=5))

    cen_mean = float(np.mean([r["cen_g"] for r in a1_rows if not np.isnan(r["cen_g"])]))
    cv_g = np.array([r["cv_g"] for r in a1_rows])
    cv_se = np.array([r["cv_se"] for r in a1_rows])

    weak_a1_summary = {
        "n_replicates": 1000,
        "dgp": "eta = -3.4 + 0.9*x0 - 0.5*x1 + 0.54*z_true + 8 pure nulls, N=6000, rho=15% Neyman",
        "census_mean_gain": cen_mean,
        "cv_4fold_holm": {
            "mean_gain": float(np.mean(cv_g)),
            "mean_z_ratio": float(np.mean(cv_g / np.maximum(1e-9, cv_se))),
            "power": prop_summary([r["cv_power"] for r in a1_rows]),
            "sequential_null_fwer": prop_summary([r["cv_null_fwer"] for r in a1_rows]),
            "ci95_cov_vs_census": prop_summary(
                (np.abs(cv_g - cen_mean) <= 1.96 * cv_se).astype(int)
            ),
        },
        "split_pilot_50_50_holm": {
            "power": prop_summary([r["sp_seq_power"] for r in a1_rows]),
            "sequential_null_fwer": prop_summary([r["sp_seq_null_fwer"] for r in a1_rows]),
            "ci95_cov_data_adaptive": prop_summary([r["sp_cov_cond"] for r in a1_rows]),
        },
    }

    weak_a2_summary = {
        "n_replicates": 500,
        "dgp": "2 nearly-tied weak signals (0.60, 0.58) + 8 pure nulls, N=6000, rho=15% Neyman",
        "cv_4fold_holm": {
            "step1_picks_w1_rate": float(np.mean([r["cv_first_w1"] for r in a2_rows])),
            "step1_picks_w2_rate": float(np.mean([r["cv_first_w2"] for r in a2_rows])),
            "any_signal_power": prop_summary([r["cv_any_signal"] for r in a2_rows]),
            "both_signals_power": prop_summary([r["cv_both_signals"] for r in a2_rows]),
            "sequential_null_fwer": prop_summary([r["cv_null_fwer"] for r in a2_rows]),
        },
        "split_pilot_50_50_holm": {
            "any_signal_power": prop_summary([r["sp_any_signal"] for r in a2_rows]),
            "both_signals_power": prop_summary([r["sp_both_signals"] for r in a2_rows]),
            "sequential_null_fwer": prop_summary([r["sp_null_fwer"] for r in a2_rows]),
        },
    }

    # Sim B exact % reductions vs SRS AND vs BCC from cached v2.1 Block 1
    b1_v21 = json.loads((V21_BLOCKS / "block1_monte_carlo.json").read_text(encoding="utf-8"))
    rmse_d = b1_v21["sim_b_200rep"]["rmse"]
    sim_b_ratios: dict[str, Any] = {}
    for r_lbl in ("05", "10", "20"):
        srs = rmse_d[f"srs_{r_lbl}"]["rmse"]
        bcc = rmse_d[f"bcc_{r_lbl}"]["rmse"]
        ht = rmse_d[f"neyman_{r_lbl}"]["rmse"]
        bc = rmse_d[f"neyman_bc_{r_lbl}"]["rmse"]
        sim_b_ratios[f"rho_{r_lbl}pct"] = {
            "srs_rmse": srs,
            "bcc_rmse": bcc,
            "neyman_ht_rmse": ht,
            "neyman_bc_rmse": bc,
            "ht_pct_change_vs_srs": 100.0 * (ht / srs - 1.0),
            "ht_pct_change_vs_bcc": 100.0 * (ht / bcc - 1.0),
            "bc_pct_change_vs_srs": 100.0 * (bc / srs - 1.0),
            "bc_pct_change_vs_bcc": 100.0 * (bc / bcc - 1.0),
        }

    # Combine Sim C1 (from v2.1 + new split-pilot & single-ch perm) and Sim C2 (multi-col twins)
    c1_methods = dict(b1_v21["sim_c_200rep"]["methods"])
    for extra_k in ("c1_split_pilot", "c1_single_ch_perm"):
        c1_methods[extra_k] = {
            "both_mech_recovered": prop_summary([r[extra_k]["both_mech"] for r in c_rows]),
            "dup_twin_rate": prop_summary([r[extra_k]["dup_twin"] for r in c_rows]),
            "echo_rate": prop_summary([r[extra_k]["echo"] for r in c_rows]),
            "null_fwer": prop_summary([r[extra_k]["null"] for r in c_rows]),
        }

    c2_methods: dict[str, Any] = {}
    for k2 in (
        "c2_marginal_mi",
        "c2_oneshot_cmi",
        "c2_single_ch_perm",
        "c2_fullpool_cpi_pilot",
        "c2_chamma_cpi",
        "c2_loco",
        "c2_seq_cv",
        "c2_seq_split_pilot",
    ):
        c2_methods[k2] = {
            "both_mech_recovered": prop_summary([r[k2]["both_mech"] for r in c_rows]),
            "dup_twin_rate": prop_summary([r[k2]["dup_twin"] for r in c_rows]),
            "echo_rate": prop_summary([r[k2]["echo"] for r in c_rows]),
            "null_fwer": prop_summary([r[k2]["null"] for r in c_rows]),
        }

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "strong_sim_a_1000rep_ref": b1_v21["sim_a_1000rep"],
        "weak_sim_a1_1000rep": weak_a1_summary,
        "competing_weak_sim_a2_500rep": weak_a2_summary,
        "sim_b_vs_srs_and_bcc": sim_b_ratios,
        "sim_c1_1col_asymmetric_twin_200rep": c1_methods,
        "sim_c2_multicol_symmetric_twin_200rep": c2_methods,
    }
    PARTIAL_V22.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 1 v2.2] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


# ===========================================================================
# BLOCK 2: Full 120-Split Leak-Free Stress Pool + Native Table-2 Audit (Closes F6)
# ===========================================================================


def _full_leakfree_stress_split_worker(args: tuple[str, int]) -> dict[str, Any]:
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

    # Strictly train_idx-fitted stress pool (zero cal/test leakage!)
    Z_stress, cands_stress = materialize_leak_free_stress_pool(coh, train_idx=tr_i, seed=seed)
    cand_map = {c.channel_id: c for c in cands_stress}

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

    eval_cache: dict[tuple[str, ...], float] = {}

    def _eval_ids(sel_ids: list[str]) -> dict[str, Any]:
        key = tuple(sorted(sel_ids))
        if key not in eval_cache:
            cols = [col for cid in key for col in cand_map[cid].columns]
            df_tr = pd.concat([Xb_tr, Zs_tr[cols]], axis=1) if cols else Xb_tr
            df_ca = pd.concat([Xb_ca, Zs_ca[cols]], axis=1) if cols else Xb_ca
            df_te = pd.concat([Xb_te, Zs_te[cols]], axis=1) if cols else Xb_te
            _, p_te = _fit_downstream_gbdt_worker(
                df_tr, y_tr, df_ca, y_ca, df_te, coh.base_semantic_types, seed=seed, n_jobs=1
            )
            eval_cache[key] = float(average_precision_score(y_te, p_te))
        nr, tr, er = _spurious_rates(sel_ids, cands_stress)
        spent = float(sum(cand_map[cid].acquisition_cost for cid in sel_ids))
        return {
            "auprc": eval_cache[key],
            "selected_ids": sel_ids,
            "spent": spent,
            "null_pct": nr,
            "twin_pct": tr,
            "echo_pct": er,
        }

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
    r_sp = discover_sequential_split_pilot(
        Xb_tr,
        Zs_tr,
        y_tr,
        cands_stress,
        oof_b,
        pilot,
        max_budget=coh.acquisition_budget,
        base_semantic_types=coh.base_semantic_types,
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
    r_oneshot = discover_sequential_conditional_gain(
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
        sequential=False,
        seed=seed,
    )
    r_cpi = discover_competitor_method(
        "cpi_pilot",
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
    r_loco = discover_competitor_method(
        "loco_pilot",
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
    r_ko = discover_competitor_method(
        "crt_knockoff_pilot",
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
    r_unwt = discover_competitor_method(
        "unweighted_pilot_no_lcb",
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
    r_marg = discover_competitor_method(
        "global_mi",
        Xb_tr,
        Zs_tr,
        y_tr,
        cands_stress,
        oof_b,
        pilot,
        max_budget=coh.acquisition_budget,
        base_semantic_types=coh.base_semantic_types,
        seed=seed,
    )

    out_row: dict[str, Any] = {
        "cohort_key": cohort_key,
        "split_idx": split_idx,
        "lazy_exact_match": int(r_prop.selected_ids == r_nolazy.selected_ids),
        "arms": {
            "proposed_ht_holm": _eval_ids(r_prop.selected_ids),
            "no_lazy_ht_holm": _eval_ids(r_nolazy.selected_ids),
            "split_pilot_50_50_holm": _eval_ids(r_sp.selected_ids),
            "breslow_cain_holm": _eval_ids(r_bc.selected_ids),
            "oneshot_cmi": _eval_ids(r_oneshot.selected_ids),
            "cpi_pilot": _eval_ids(r_cpi.selected_ids),
            "chamma_residual_cpi": _eval_ids(r_chamma.selected_ids),
            "loco_pilot": _eval_ids(r_loco.selected_ids),
            "crt_knockoff_pilot": _eval_ids(r_ko.selected_ids),
            "unweighted_pilot_no_lcb": _eval_ids(r_unwt.selected_ids),
            "global_mi": _eval_ids(r_marg.selected_ids),
        },
    }

    # On NeurIPS_BAF and UCI_Taiwan_Credit, also verify native 15% & 100% arms match Table 2!
    if cohort_key in {"NeurIPS_BAF", "UCI_Taiwan_Credit"}:
        nat_s15 = discover_sequential_conditional_gain(
            Xb_tr,
            coh.Z_pool.iloc[tr_i].reset_index(drop=True),
            y_tr,
            coh.candidates,
            oof_b,
            pilot,
            max_budget=coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            seed=seed,
        )
        pil_full = sample_two_phase_pilot(
            y_tr, oof_b, val_m, pilot_fraction=1.0, design="full", seed=seed
        )
        nat_s100 = discover_sequential_conditional_gain(
            Xb_tr,
            coh.Z_pool.iloc[tr_i].reset_index(drop=True),
            y_tr,
            coh.candidates,
            oof_b,
            pil_full,
            max_budget=coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            seed=seed,
        )
        c15 = [col for c in nat_s15.selected_channels for col in c.columns]
        c100 = [col for c in nat_s100.selected_channels for col in c.columns]
        _, p_nat15 = _fit_downstream_gbdt_worker(
            pd.concat([Xb_tr, coh.Z_pool.iloc[tr_i][c15].reset_index(drop=True)], axis=1)
            if c15
            else Xb_tr,
            y_tr,
            pd.concat([Xb_ca, coh.Z_pool.iloc[ca_i][c15].reset_index(drop=True)], axis=1)
            if c15
            else Xb_ca,
            y_ca,
            pd.concat([Xb_te, coh.Z_pool.iloc[te_i][c15].reset_index(drop=True)], axis=1)
            if c15
            else Xb_te,
            coh.base_semantic_types,
            seed=seed,
            n_jobs=1,
        )
        _, p_nat100 = _fit_downstream_gbdt_worker(
            pd.concat([Xb_tr, coh.Z_pool.iloc[tr_i][c100].reset_index(drop=True)], axis=1)
            if c100
            else Xb_tr,
            y_tr,
            pd.concat([Xb_ca, coh.Z_pool.iloc[ca_i][c100].reset_index(drop=True)], axis=1)
            if c100
            else Xb_ca,
            y_ca,
            pd.concat([Xb_te, coh.Z_pool.iloc[te_i][c100].reset_index(drop=True)], axis=1)
            if c100
            else Xb_te,
            coh.base_semantic_types,
            seed=seed,
            n_jobs=1,
        )
        out_row["native_recheck"] = {
            "neyman_15pct_auprc": float(average_precision_score(y_te, p_nat15)),
            "full_100pct_auprc": float(average_precision_score(y_te, p_nat100)),
        }

    gc.collect()
    return out_row


def run_block_2_v22(max_workers: int = 8) -> dict[str, Any]:
    out_file = PARTIAL_V22 / "block2_v2_2_full_120split_leakfree_stress.json"
    if out_file.exists():
        print("Loaded cached Block 2 v2.2:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    tasks = [(c_key, s_i) for c_key in COHORT_MCIDS for s_i in range(20)]
    print(f"[Block 2 v2.2] Running ALL {len(tasks)} leak-free stress splits across 6 cohorts...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_full_leakfree_stress_split_worker, tasks, chunksize=1))

    ckpt_dir = OUT_DIR / "checkpoints"
    arm_names = list(rows[0]["arms"].keys())
    by_cohort: dict[str, Any] = {}
    for c_key in COHORT_MCIDS:
        sub = sorted(
            [r for r in rows if r["cohort_key"] == c_key], key=lambda d: int(d["split_idx"])
        )
        v1_stress_auprc: list[float] = []
        v1_nat15_auprc: list[float] = []
        v1_nat100_auprc: list[float] = []
        for s_i in range(20):
            ck = json.loads((ckpt_dir / f"{c_key}_split{s_i:02d}.json").read_text(encoding="utf-8"))
            v1_stress_auprc.append(float(ck["stress_arms"]["neyman_15pct_seq"]["auprc"]))
            v1_nat15_auprc.append(float(ck["native_arms"]["neyman_15pct_seq"]["auprc"]))
            v1_nat100_auprc.append(float(ck["native_arms"]["full_100pct_seq"]["auprc"]))

        lf_prop = [float(r["arms"]["proposed_ht_holm"]["auprc"]) for r in sub]
        entry: dict[str, Any] = {
            "n_splits": len(sub),
            "lazy_exact_match_count": int(sum(r["lazy_exact_match"] for r in sub)),
            "f6_reconciliation": {
                "v1_cached_20split_stress_mean": float(np.mean(v1_stress_auprc)),
                "v1_cached_first5_stress_mean": float(np.mean(v1_stress_auprc[:5])),
                "leakfree_first5_stress_mean": float(np.mean(lf_prop[:5])),
                "leakfree_20split_stress_mean": float(np.mean(lf_prop)),
                "paired_20split_diff_lf_minus_v1": float(
                    np.mean(np.array(lf_prop) - np.array(v1_stress_auprc))
                ),
                "paired_20split_se": float(
                    np.std(np.array(lf_prop) - np.array(v1_stress_auprc), ddof=1) / np.sqrt(20.0)
                ),
            },
            "arms": {},
        }
        if "native_recheck" in sub[0]:
            n15 = [float(r["native_recheck"]["neyman_15pct_auprc"]) for r in sub]
            n100 = [float(r["native_recheck"]["full_100pct_auprc"]) for r in sub]
            entry["native_table2_verification"] = {
                "v1_native_15pct_mean": float(np.mean(v1_nat15_auprc)),
                "rerun_native_15pct_mean": float(np.mean(n15)),
                "v1_native_100pct_mean": float(np.mean(v1_nat100_auprc)),
                "rerun_native_100pct_mean": float(np.mean(n100)),
            }

        for a_name in arm_names:
            auprcs = np.array([r["arms"][a_name]["auprc"] for r in sub], dtype=float)
            entry["arms"][a_name] = {
                "auprc_mean": float(np.mean(auprcs)),
                "auprc_sd": float(np.std(auprcs, ddof=1)),
                "auprc_se": float(np.std(auprcs, ddof=1) / np.sqrt(20.0)),
                "spent_mean": float(np.mean([r["arms"][a_name]["spent"] for r in sub])),
                "null_pct": float(np.mean([r["arms"][a_name]["null_pct"] for r in sub])),
                "twin_pct": float(np.mean([r["arms"][a_name]["twin_pct"] for r in sub])),
                "echo_pct": float(np.mean([r["arms"][a_name]["echo_pct"] for r in sub])),
            }
        by_cohort[c_key] = entry

    # Also retain clean 2x2 ablation from v2.1 Block 2
    b2_v21 = json.loads(
        (V21_BLOCKS / "block2_stress_and_ablations.json").read_text(encoding="utf-8")
    )
    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "leak_free_120split_stress": by_cohort,
        "ablation_2x2_diabetes_scania": {
            k: b2_v21["leak_free_train_fit_stress"][k]["ablation_2x2"]
            for k in ("UCI_Diabetes_Discharge", "Scania_APS_Fleet")
        },
        "cached_120_split_channel_ledger": b2_v21["cached_120_split_channel_ledger"],
    }
    PARTIAL_V22.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 2 v2.2] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


# ===========================================================================
# BLOCK 3: Stage-B Boundary Uncertainty, Full 100%, c_query Sweep, & DB-CMI
# ===========================================================================


def _evaluate_canonical_v22_worker(cohort_key: str) -> dict[str, Any]:
    seed = 2026
    coh, boot_groups_all = _load_full_canonical_cohort(cohort_key)
    prev_cohort = float(np.mean(coh.y))

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

    disc_sub = tr_i.size
    if disc_sub > 160000:
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

    tau_star = float(coh.contract.cost_fp / (coh.contract.cost_fp + coh.contract.cost_fn))
    router_gh = StreamingSelectiveRouter(
        coh.contract,
        coh.per_instance_query_cost,
        quadrature_mode="gauss_hermite",
        mean_centered=True,
    ).fit(y_ca, p1_ca, p2_ca)
    router_unc = StreamingSelectiveRouter(
        coh.contract,
        coh.per_instance_query_cost,
        quadrature_mode="gauss_hermite",
        mean_centered=False,
    ).fit(y_ca, p1_ca, p2_ca)
    router_emp = StreamingSelectiveRouter(
        coh.contract,
        coh.per_instance_query_cost,
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
    bu_ca = -np.abs(p1_ca - tau_star)
    bu_te = -np.abs(p1_te - tau_star)

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
    _, q_bu20, _, _ = calibrated_randomized_threshold_masks(bu_ca, bu_te, 0.20, u_ca, u_te)

    q_gh_auto = voi_gh_te > coh.per_instance_query_cost
    q_unc_auto = voi_unc_te > coh.per_instance_query_cost
    q_emp_auto = voi_emp_te > coh.per_instance_query_cost
    q_zero = np.zeros(y_te.size, dtype=bool)
    q_full = np.ones(y_te.size, dtype=bool)

    # Exact matched-q Boundary Uncertainty: select exact same count k_auto closest to tau*!
    k_auto = int(np.sum(q_gh_auto))
    q_bu_matched = np.zeros(y_te.size, dtype=bool)
    if k_auto > 0:
        top_bu_idx = np.argsort(-bu_te, kind="mergesort")[:k_auto]
        q_bu_matched[top_bu_idx] = True

    p_gh20 = np.where(q_gh20, p2_te, p1_te)
    p_unc20 = np.where(q_unc20, p2_te, p1_te)
    p_top20 = np.where(q_top20, p2_te, p1_te)
    p_bu20 = np.where(q_bu20, p2_te, p1_te)
    p_gh_auto = np.where(q_gh_auto, p2_te, p1_te)
    p_unc_auto = np.where(q_unc_auto, p2_te, p1_te)
    p_emp_auto = np.where(q_emp_auto, p2_te, p1_te)
    p_bu_matched = np.where(q_bu_matched, p2_te, p1_te)

    # Covert-AFA variants: flat_any vs proportional_cost vs decision_boundary_weighted (DB-CMI)
    Z_tr_afa, cands_afa = materialize_leak_free_stress_pool(coh, train_idx=tr_i, seed=seed)
    keep_afa_cands = [
        c
        for c in cands_afa
        if c.mechanism_tag not in {"stage2_twin", "baseline_echo"}
        and c.channel_id not in {"null_channel_3", "null_channel_4"}
    ]
    ref_b_cost = max(1.0, float(sel_15.total_cost_spent))
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
    Z_te_afa = Z_tr_afa.iloc[te_i].reset_index(drop=True)
    afa_20_flat = afa.evaluate(Xb_te, Z_te_afa, y_te, p1_te, target_query_rate=0.20)
    afa_20_prop = afa.evaluate(
        Xb_te,
        Z_te_afa,
        y_te,
        p1_te,
        target_query_rate=0.20,
        fee_accounting="proportional_cost",
        ref_bundle_cost=ref_b_cost,
    )
    afa_20_dbcmi_flat = afa.evaluate(
        Xb_te, Z_te_afa, y_te, p1_te, target_query_rate=0.20, decision_boundary_weighted=True
    )
    afa_20_dbcmi_prop = afa.evaluate(
        Xb_te,
        Z_te_afa,
        y_te,
        p1_te,
        target_query_rate=0.20,
        decision_boundary_weighted=True,
        fee_accounting="proportional_cost",
        ref_bundle_cost=ref_b_cost,
    )
    afa_auto_flat = afa.evaluate(Xb_te, Z_te_afa, y_te, p1_te, target_query_rate=None)
    afa_auto_prop = afa.evaluate(
        Xb_te,
        Z_te_afa,
        y_te,
        p1_te,
        target_query_rate=None,
        fee_accounting="proportional_cost",
        ref_bundle_cost=ref_b_cost,
    )

    # c_query sensitivity sweep across {0.00, 0.01, 0.04, 0.08, 0.16}
    c_query_sweep: list[dict[str, Any]] = []
    for cq in (0.00, 0.01, 0.04, 0.08, 0.16):
        q_cq = voi_gh_te > cq
        p_cq = np.where(q_cq, p2_te, p1_te)
        k_cq = int(np.sum(q_cq))
        q_bu_cq = np.zeros(y_te.size, dtype=bool)
        if k_cq > 0:
            q_bu_cq[np.argsort(-bu_te, kind="mergesort")[:k_cq]] = True
        p_bu_cq = np.where(q_bu_cq, p2_te, p1_te)

        bt_vs_full = paired_cluster_bootstrap_nb(
            y_te, p_cq, q_cq, p2_te, q_full, coh.contract, cq, boot_grp_te, n_boot=5000
        )
        bt_vs_bu = paired_cluster_bootstrap_nb(
            y_te, p_cq, q_cq, p_bu_cq, q_bu_cq, coh.contract, cq, boot_grp_te, n_boot=5000
        )
        bt_vs_s1 = paired_cluster_bootstrap_nb(
            y_te, p_cq, q_cq, p1_te, q_zero, coh.contract, cq, boot_grp_te, n_boot=5000
        )
        bt_gh20_vs_bu20_cq = paired_cluster_bootstrap_nb(
            y_te, p_gh20, q_gh20, p_bu20, q_bu20, coh.contract, cq, boot_grp_te, n_boot=5000
        )
        c_query_sweep.append(
            {
                "c_query": cq,
                "q_auto": float(np.mean(q_cq)),
                "nb_voi_auto": bt_vs_full["nb_num"],
                "nb_full_100": bt_vs_full["nb_ref"],
                "nb_bu_matched": bt_vs_bu["nb_ref"],
                "nb_stage1_0": bt_vs_s1["nb_ref"],
                "delta_auto_vs_full100": bt_vs_full,
                "delta_auto_vs_bu_matched": bt_vs_bu,
                "delta_auto_vs_stage1": bt_vs_s1,
                "delta_voi20_vs_bu20": bt_gh20_vs_bu20_cq,
            }
        )

    # B = 10,000 primary Stage-B contrasts at default c_query = 0.04
    cq0 = coh.per_instance_query_cost
    afa_costs_vec = np.array([c.acquisition_cost for c in keep_afa_cands], dtype=float)
    q_vec_afa20_prop = (afa_20_prop.channel_masks_test @ afa_costs_vec) / max(1e-6, ref_b_cost)
    q_vec_dbcmi20_prop = (afa_20_dbcmi_prop.channel_masks_test @ afa_costs_vec) / max(
        1e-6, ref_b_cost
    )
    prop_q_afa20 = float(np.mean(q_vec_afa20_prop))
    prop_q_dbcmi20 = float(np.mean(q_vec_dbcmi20_prop))
    # Per-instance proportional billing vector for Covert-AFA
    bootstraps = {
        "gh_auto_vs_full100": paired_cluster_bootstrap_nb(
            y_te, p_gh_auto, q_gh_auto, p2_te, q_full, coh.contract, cq0, boot_grp_te
        ),
        "gh_auto_vs_bu_matched": paired_cluster_bootstrap_nb(
            y_te, p_gh_auto, q_gh_auto, p_bu_matched, q_bu_matched, coh.contract, cq0, boot_grp_te
        ),
        "gh20_vs_bu20": paired_cluster_bootstrap_nb(
            y_te, p_gh20, q_gh20, p_bu20, q_bu20, coh.contract, cq0, boot_grp_te
        ),
        "gh20_vs_toprisk20": paired_cluster_bootstrap_nb(
            y_te, p_gh20, q_gh20, p_top20, q_top20, coh.contract, cq0, boot_grp_te
        ),
        "gh_auto_vs_uncentered_auto": paired_cluster_bootstrap_nb(
            y_te, p_gh_auto, q_gh_auto, p_unc_auto, q_unc_auto, coh.contract, cq0, boot_grp_te
        ),
        "gh20_vs_uncentered20": paired_cluster_bootstrap_nb(
            y_te, p_gh20, q_gh20, p_unc20, q_unc20, coh.contract, cq0, boot_grp_te
        ),
        "gh_auto_vs_stage1": paired_cluster_bootstrap_nb(
            y_te, p_gh_auto, q_gh_auto, p1_te, q_zero, coh.contract, cq0, boot_grp_te
        ),
        "full100_vs_stage1": paired_cluster_bootstrap_nb(
            y_te, p2_te, q_full, p1_te, q_zero, coh.contract, cq0, boot_grp_te
        ),
        "gh20_vs_covertafa20_flat": paired_cluster_bootstrap_nb(
            y_te,
            p_gh20,
            q_gh20,
            afa_20_flat.probs_test,
            afa_20_flat.queried_any_test,
            coh.contract,
            cq0,
            boot_grp_te,
        ),
        "gh20_vs_covertafa20_prop": paired_cluster_bootstrap_nb(
            y_te,
            p_gh20,
            q_gh20,
            afa_20_prop.probs_test,
            q_vec_afa20_prop,
            coh.contract,
            cq0,
            boot_grp_te,
        ),
        "gh20_vs_dbcmi20_flat": paired_cluster_bootstrap_nb(
            y_te,
            p_gh20,
            q_gh20,
            afa_20_dbcmi_flat.probs_test,
            afa_20_dbcmi_flat.queried_any_test,
            coh.contract,
            cq0,
            boot_grp_te,
        ),
        "gh20_vs_dbcmi20_prop": paired_cluster_bootstrap_nb(
            y_te,
            p_gh20,
            q_gh20,
            afa_20_dbcmi_prop.probs_test,
            q_vec_dbcmi20_prop,
            coh.contract,
            cq0,
            boot_grp_te,
        ),
        "emp_auto_vs_gh_auto": paired_cluster_bootstrap_nb(
            y_te, p_emp_auto, q_emp_auto, p_gh_auto, q_gh_auto, coh.contract, cq0, boot_grp_te
        ),
    }

    # On IEEE_CIS_Fraud, ALSO compute 20-block contiguous temporal block bootstrap CIs (F11)
    temporal_block_boot: dict[str, Any] | None = None
    if cohort_key == "IEEE_CIS_Fraud":
        blk_ids = np.minimum(19, (np.arange(y_te.size) * 20) // max(1, y_te.size))
        temporal_block_boot = {
            "n_temporal_blocks": 20,
            "gh_auto_vs_full100": paired_cluster_bootstrap_nb(
                y_te, p_gh_auto, q_gh_auto, p2_te, q_full, coh.contract, cq0, blk_ids
            ),
            "gh_auto_vs_bu_matched": paired_cluster_bootstrap_nb(
                y_te, p_gh_auto, q_gh_auto, p_bu_matched, q_bu_matched, coh.contract, cq0, blk_ids
            ),
            "gh20_vs_bu20": paired_cluster_bootstrap_nb(
                y_te, p_gh20, q_gh20, p_bu20, q_bu20, coh.contract, cq0, blk_ids
            ),
            "gh_auto_vs_stage1": paired_cluster_bootstrap_nb(
                y_te, p_gh_auto, q_gh_auto, p1_te, q_zero, coh.contract, cq0, blk_ids
            ),
        }

    cal_s1 = compute_calibration_summary(y_te, p1_te)
    cal_s2 = compute_calibration_summary(y_te, p2_te)
    cal_auto = compute_calibration_summary(y_te, p_gh_auto)
    prev_te = float(np.mean(y_te))

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
        "prevalence_cohort": prev_cohort,
        "prevalence_test": prev_te,
        "tau_star": tau_star,
        "selected_ids_15pct": sel_15.selected_ids,
        "spent_15pct": float(sel_15.total_cost_spent),
        "arms_default_cq004": {
            "stage1_0pct": {"nb": bootstraps["gh_auto_vs_stage1"]["nb_ref"], "q": 0.0},
            "full_stage2_100pct": {"nb": bootstraps["gh_auto_vs_full100"]["nb_ref"], "q": 1.0},
            "gh_voi_auto": {
                "nb": bootstraps["gh_auto_vs_full100"]["nb_num"],
                "q": float(np.mean(q_gh_auto)),
            },
            "uncentered_voi_auto": {
                "nb": bootstraps["gh_auto_vs_uncentered_auto"]["nb_ref"],
                "q": float(np.mean(q_unc_auto)),
            },
            "emp_voi_auto": {
                "nb": bootstraps["emp_auto_vs_gh_auto"]["nb_num"],
                "q": float(np.mean(q_emp_auto)),
            },
            "bu_matched_q_auto": {
                "nb": bootstraps["gh_auto_vs_bu_matched"]["nb_ref"],
                "q": float(np.mean(q_bu_matched)),
            },
            "gh_voi_20pct": {
                "nb": bootstraps["gh20_vs_bu20"]["nb_num"],
                "q": float(np.mean(q_gh20)),
            },
            "bu_20pct": {"nb": bootstraps["gh20_vs_bu20"]["nb_ref"], "q": float(np.mean(q_bu20))},
            "top_risk_20pct": {
                "nb": bootstraps["gh20_vs_toprisk20"]["nb_ref"],
                "q": float(np.mean(q_top20)),
            },
            "uncentered_20pct": {
                "nb": bootstraps["gh20_vs_uncentered20"]["nb_ref"],
                "q": float(np.mean(q_unc20)),
            },
            "covert_afa_20pct_flat": {
                "nb": afa_20_flat.net_benefit_per_1k,
                "q_any": afa_20_flat.any_query_rate_test,
                "null_q": afa_20_flat.null_channel_query_rate_test,
            },
            "covert_afa_20pct_proportional": {
                "nb": afa_20_prop.net_benefit_per_1k,
                "q_any": afa_20_prop.any_query_rate_test,
                "q_prop_equiv": prop_q_afa20,
            },
            "dbcmi_afa_20pct_flat": {
                "nb": afa_20_dbcmi_flat.net_benefit_per_1k,
                "q_any": afa_20_dbcmi_flat.any_query_rate_test,
                "null_q": afa_20_dbcmi_flat.null_channel_query_rate_test,
            },
            "dbcmi_afa_20pct_proportional": {
                "nb": afa_20_dbcmi_prop.net_benefit_per_1k,
                "q_any": afa_20_dbcmi_prop.any_query_rate_test,
                "q_prop_equiv": prop_q_dbcmi20,
            },
            "covert_afa_auto_flat": {
                "nb": afa_auto_flat.net_benefit_per_1k,
                "q_any": afa_auto_flat.any_query_rate_test,
            },
            "covert_afa_auto_proportional": {
                "nb": afa_auto_prop.net_benefit_per_1k,
                "q_any": afa_auto_prop.any_query_rate_test,
            },
        },
        "bootstraps_10k": bootstraps,
        "ieee_temporal_20block_bootstrap": temporal_block_boot,
        "c_query_sweep": c_query_sweep,
        "auprc": {
            "stage1": float(average_precision_score(y_te, p1_te)),
            "stage2_full": float(average_precision_score(y_te, p2_te)),
            "gh_voi_auto": float(average_precision_score(y_te, p_gh_auto)),
            "bu_matched_auto": float(average_precision_score(y_te, p_bu_matched)),
        },
        "calibration": {
            "stage1": {"citl": cal_s1.cal_intercept, "slope": cal_s1.cal_slope, "ici": cal_s1.ici},
            "stage2_full": {
                "citl": cal_s2.cal_intercept,
                "slope": cal_s2.cal_slope,
                "ici": cal_s2.ici,
            },
            "gh_voi_auto": {
                "citl": cal_auto.cal_intercept,
                "slope": cal_auto.cal_slope,
                "ici": cal_auto.ici,
            },
        },
    }


def run_block_3_v22(max_workers: int = 3) -> dict[str, Any]:
    out_file = PARTIAL_V22 / "block3_v2_2_stageb_comparators_and_feesweep.json"
    if out_file.exists():
        print("Loaded cached Block 3 v2.2:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    print("[Block 3 v2.2] Evaluating 6 canonical holdouts (BU, Full 100%, c_query, DB-CMI)...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_evaluate_canonical_v22_worker, list(COHORT_MCIDS.keys())))

    b3_v21 = json.loads(
        (V21_BLOCKS / "block3_canonical_holdouts_and_covert.json").read_text(encoding="utf-8")
    )
    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "covert_afa_synthetic_oracle_n4000": b3_v21["covert_afa_synthetic_oracle_n4000"],
        "canonical_holdouts": {r["cohort_key"]: r for r in rows},
        "disjoint_4_eras": b3_v21["disjoint_chrono_and_cfbh_shift"]["disjoint_4_eras"],
    }
    PARTIAL_V22.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 3 v2.2] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


# ===========================================================================
# BLOCK 4: 100-Replication Conformal FDR Shift vs Jin-Candes WCS & Burn-In
# ===========================================================================


def run_block_4_v22() -> dict[str, Any]:
    out_file = PARTIAL_V22 / "block4_v2_2_cfbh_shift_and_wcs.json"
    if out_file.exists():
        print("Loaded cached Block 4 v2.2:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    print(
        "[Block 4 v2.2] Running R=100 Conformal FDR shift replications (CfBH vs WCS vs Burn-In)..."
    )
    shift_out: dict[str, Any] = {}

    for c_key in ("IEEE_CIS_Fraud", "NeurIPS_BAF"):
        coh, _ = _load_full_canonical_cohort(c_key)
        n_tot = len(coh.X_base)
        era_edges = np.linspace(0, n_tot, 5, dtype=int)

        # Era 0 (source) and Era 3 (shifted target)
        lo0, hi0 = int(era_edges[0]), int(era_edges[1])
        lo3, hi3 = int(era_edges[3]), int(era_edges[4])

        Xb0 = coh.X_base.iloc[lo0:hi0].reset_index(drop=True)
        Z0 = coh.Z_pool.iloc[lo0:hi0].reset_index(drop=True)
        y0 = coh.y[lo0:hi0]
        t0_arr = coh.time_order[lo0:hi0]

        Xb3 = coh.X_base.iloc[lo3:hi3].reset_index(drop=True)
        Z3 = coh.Z_pool.iloc[lo3:hi3].reset_index(drop=True)
        y3 = coh.y[lo3:hi3]
        t3_arr = coh.time_order[lo3:hi3]

        s0_df = Xb0.copy()
        s0_df["__y__"] = y0
        s0_df["__t__"] = t0_arr
        sp0 = freeze_split(s0_df, "__y__", time_col="__t__", seed=2026)
        tr0, ca0 = np.asarray(sp0.train_idx), np.asarray(sp0.cal_idx)

        s3_df = Xb3.copy()
        s3_df["__y__"] = y3
        s3_df["__t__"] = t3_arr
        sp3 = freeze_split(s3_df, "__y__", time_col="__t__", seed=2029)
        ca3, te3 = np.asarray(sp3.cal_idx), np.asarray(sp3.test_idx)

        oof0, vm0 = _fit_oof_stage1_worker(
            Xb0.iloc[tr0], y0[tr0], coh.base_semantic_types, None, t0_arr[tr0], seed=2026, n_jobs=4
        )
        pil0 = sample_two_phase_pilot(
            y0[tr0], oof0, vm0, pilot_fraction=0.15, design="neyman", seed=2026
        )
        sel0 = discover_sequential_conditional_gain(
            Xb0.iloc[tr0].reset_index(drop=True),
            Z0.iloc[tr0].reset_index(drop=True),
            y0[tr0],
            coh.candidates,
            oof0,
            pil0,
            max_budget=coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            time_order_train=t0_arr[tr0],
            seed=2026,
        )
        scols = [c for ch in sel0.selected_channels for c in ch.columns]

        df_tr0 = pd.concat([Xb0.iloc[tr0], Z0.iloc[tr0][scols]], axis=1) if scols else Xb0.iloc[tr0]
        df_ca0 = pd.concat([Xb0.iloc[ca0], Z0.iloc[ca0][scols]], axis=1) if scols else Xb0.iloc[ca0]
        df_ca3 = pd.concat([Xb3.iloc[ca3], Z3.iloc[ca3][scols]], axis=1) if scols else Xb3.iloc[ca3]
        df_te3 = pd.concat([Xb3.iloc[te3], Z3.iloc[te3][scols]], axis=1) if scols else Xb3.iloc[te3]

        p_ca0, p_te3 = _fit_downstream_gbdt_worker(
            df_tr0, y0[tr0], df_ca0, y0[ca0], df_te3, coh.base_semantic_types, seed=2099, n_jobs=4
        )
        _, p_ca3 = _fit_downstream_gbdt_worker(
            df_tr0, y0[tr0], df_ca0, y0[ca0], df_ca3, coh.base_semantic_types, seed=2099, n_jobs=4
        )

        # Unlabeled density-ratio estimator for Jin-Candes WCS: P(Era=3|X,Z_S) / P(Era=0|X,Z_S)
        # Drop monotonic transaction timestamp/month identifiers if any exist in X_base
        drop_t = [c for c in df_ca0.columns if c.lower() in {"transactiondt", "month"}]
        dom_ca0 = df_ca0.drop(columns=drop_t, errors="ignore")
        dom_te3 = df_te3.drop(columns=drop_t, errors="ignore")
        n_d0 = min(15000, len(dom_ca0))
        n_d3 = min(15000, len(dom_te3))
        rng_d = np.random.default_rng(2026)
        i_d0 = rng_d.choice(len(dom_ca0), size=n_d0, replace=False)
        i_d3 = rng_d.choice(len(dom_te3), size=n_d3, replace=False)
        dom_df = pd.concat([dom_ca0.iloc[i_d0], dom_te3.iloc[i_d3]], axis=0).reset_index(drop=True)
        dom_y = np.concatenate([np.zeros(n_d0, dtype=int), np.ones(n_d3, dtype=int)])
        dom_sem = {k: v for k, v in coh.base_semantic_types.items() if k in dom_df.columns}
        prep_dom = TabularPreprocessor(semantic_types=dom_sem)
        Xd_tr = prep_dom.fit_transform(dom_df)
        dom_clf = lgb.LGBMClassifier(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=50,
            reg_lambda=5.0,
            random_state=2026,
            n_jobs=4,
            verbose=-1,
        )
        dom_clf.fit(Xd_tr, dom_y)
        q_dom_ca0 = np.clip(dom_clf.predict_proba(prep_dom.transform(dom_ca0))[:, 1], 0.02, 0.98)
        q_dom_te3 = np.clip(dom_clf.predict_proba(prep_dom.transform(dom_te3))[:, 1], 0.02, 0.98)
        prior_corr = float(n_d0) / float(n_d3)
        w_ca0_all = prior_corr * (q_dom_ca0 / (1.0 - q_dom_ca0))
        w_te3_all = prior_corr * (q_dom_te3 / (1.0 - q_dom_te3))

        tfdr = 1.0 - coh.contract.target_precision
        y_ca0 = y0[ca0]
        y_ca3 = y3[ca3]
        y_te3 = y3[te3]

        # R = 100 repeated random sub-era / bootstrap replications
        n_reps = 100
        rng_r = np.random.default_rng(7700)
        rec: dict[str, list[dict[str, float]]] = {
            "in_era_exchangeable_cfbh": [],
            "stale_era0_unweighted_cfbh": [],
            "unlabeled_covariate_shift_wcs": [],
            "labeled_burnin_2500_cfbh": [],
        }

        n_c0_sub = min(5000, y_ca0.size)
        n_te_sub = min(10000, y_te3.size // 2)
        for r_i in range(n_reps):
            # Exchangeable partition of Era-3 test stream into disjoint cal_exch & eval_exch
            perm_te = rng_r.permutation(y_te3.size)
            idx_ex_cal = perm_te[:4000]
            idx_ev = perm_te[4000 : 4000 + n_te_sub]
            idx_c0 = rng_r.choice(y_ca0.size, size=n_c0_sub, replace=False)
            idx_bi = rng_r.choice(y_ca3.size, size=min(2500, y_ca3.size), replace=False)

            r_ex = conformal_fdr_selection(
                p_te3[idx_ex_cal], y_te3[idx_ex_cal], p_te3[idx_ev], y_te3[idx_ev], tfdr, seed=r_i
            )
            r_st = conformal_fdr_selection(
                p_ca0[idx_c0], y_ca0[idx_c0], p_te3[idx_ev], y_te3[idx_ev], tfdr, seed=r_i
            )
            r_wcs = weighted_conformal_fdr_selection(
                p_ca0[idx_c0],
                y_ca0[idx_c0],
                w_ca0_all[idx_c0],
                p_te3[idx_ev],
                y_te3[idx_ev],
                w_te3_all[idx_ev],
                tfdr,
                seed=r_i,
            )
            r_bi = conformal_fdr_selection(
                p_ca3[idx_bi], y_ca3[idx_bi], p_te3[idx_ev], y_te3[idx_ev], tfdr, seed=r_i
            )
            for k_m, res_m in (
                ("in_era_exchangeable_cfbh", r_ex),
                ("stale_era0_unweighted_cfbh", r_st),
                ("unlabeled_covariate_shift_wcs", r_wcs),
                ("labeled_burnin_2500_cfbh", r_bi),
            ):
                rec[k_m].append(
                    {
                        "fdp": res_m.realized_fdp,
                        "recall": res_m.realized_recall,
                        "alerts_per_100": 100.0 * res_m.selection_rate,
                    }
                )

        cal_lag = compute_calibration_summary(y_te3, p_te3)
        methods_sum: dict[str, Any] = {}
        for k_m, lst in rec.items():
            fdps = np.array([d["fdp"] for d in lst], dtype=float)
            recs = np.array([d["recall"] for d in lst], dtype=float)
            alrts = np.array([d["alerts_per_100"] for d in lst], dtype=float)
            methods_sum[k_m] = {
                "mean_fdr": float(np.mean(fdps)),
                "se_fdr": float(np.std(fdps, ddof=1) / np.sqrt(n_reps)),
                "q025_fdr": float(np.quantile(fdps, 0.025)),
                "q975_fdr": float(np.quantile(fdps, 0.975)),
                "mean_recall": float(np.mean(recs)),
                "se_recall": float(np.std(recs, ddof=1) / np.sqrt(n_reps)),
                "mean_alerts_per_100": float(np.mean(alrts)),
            }

        # Paired difference Stale vs Exchangeable & Burn-In vs Stale & Burn-In vs WCS
        st_arr = np.array([d["fdp"] for d in rec["stale_era0_unweighted_cfbh"]])
        ex_arr = np.array([d["fdp"] for d in rec["in_era_exchangeable_cfbh"]])
        wcs_arr = np.array([d["fdp"] for d in rec["unlabeled_covariate_shift_wcs"]])
        bi_arr = np.array([d["fdp"] for d in rec["labeled_burnin_2500_cfbh"]])

        shift_out[c_key] = {
            "n_replications": n_reps,
            "target_fdr": tfdr,
            "era0_prev": float(np.mean(y_ca0)),
            "era3_prev": float(np.mean(y_te3)),
            "lagged_citl": cal_lag.cal_intercept,
            "lagged_slope": cal_lag.cal_slope,
            "methods": methods_sum,
            "paired_fdr_diffs": {
                "stale_minus_exchangeable_mean": float(np.mean(st_arr - ex_arr)),
                "stale_minus_exchangeable_se": float(
                    np.std(st_arr - ex_arr, ddof=1) / np.sqrt(n_reps)
                ),
                "wcs_minus_stale_mean": float(np.mean(wcs_arr - st_arr)),
                "burnin_minus_stale_mean": float(np.mean(bi_arr - st_arr)),
                "burnin_minus_wcs_mean": float(np.mean(bi_arr - wcs_arr)),
                "burnin_minus_wcs_se": float(np.std(bi_arr - wcs_arr, ddof=1) / np.sqrt(n_reps)),
            },
        }
        gc.collect()

    out = {"elapsed_sec": round(time.time() - t0, 2), "cfbh_wcs_burnin_100rep": shift_out}
    PARTIAL_V22.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 4 v2.2] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


# ===========================================================================
# BLOCK 5: Race-Blind UCI Diabetes Simulated Acquisition Scenario (Both Splits)
# ===========================================================================


def _skipped_bounds_summary(
    y_te: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    q_mask: np.ndarray,
    grp_te: np.ndarray,
    tau_star: float,
    seed: int = 2026,
) -> dict[str, Any]:
    sk = ~q_mask
    n_sk = int(np.sum(sk))
    grp_sk = grp_te[sk]
    n_pat_sk = int(np.unique(grp_sk).size) if n_sk > 0 else 0
    a1_sk = p1[sk] >= tau_star
    a2_sk = p2[sk] >= tau_star
    y_sk = y_te[sk]
    flips = (a1_sk != a2_sk).astype(float)
    neg_to_pos = ((~a1_sk) & a2_sk).astype(float)
    pos_to_neg = (a1_sk & (~a2_sk)).astype(float)
    ben = ((neg_to_pos > 0) & (y_sk == 1)) | ((pos_to_neg > 0) & (y_sk == 0))
    harm = ((neg_to_pos > 0) & (y_sk == 0)) | ((pos_to_neg > 0) & (y_sk == 1))

    k_fl = int(np.sum(flips))
    k_np = int(np.sum(neg_to_pos))
    k_pn = int(np.sum(pos_to_neg))

    # Patient-cluster ratio bootstrap 1-sided 95% upper bound on skipped encounters
    _, inv_p = np.unique(grp_sk, return_inverse=True)
    g_cnt = int(np.max(inv_p)) + 1
    cl_fl = np.bincount(inv_p, weights=flips, minlength=g_cnt)
    cl_np = np.bincount(inv_p, weights=neg_to_pos, minlength=g_cnt)
    cl_pn = np.bincount(inv_p, weights=pos_to_neg, minlength=g_cnt)
    cl_n = np.bincount(inv_p, minlength=g_cnt).astype(float)
    rng = np.random.default_rng(seed)
    cnts = rng.multinomial(g_cnt, np.full(g_cnt, 1.0 / g_cnt), size=10000)
    den = np.maximum(1.0, cnts @ cl_n)
    ub_boot_fl = float(np.quantile((cnts @ cl_fl) / den, 0.95))
    ub_boot_np = float(np.quantile((cnts @ cl_np) / den, 0.95))
    ub_boot_pn = float(np.quantile((cnts @ cl_pn) / den, 0.95))

    return {
        "n_skipped_encounters": n_sk,
        "n_skipped_unique_patients": n_pat_sk,
        "flip_count": k_fl,
        "flip_rate": float(k_fl) / max(1, n_sk),
        "flip_ub95_encounter_cp": clopper_pearson_upper_1sided(k_fl, max(1, n_sk)),
        "flip_ub95_patient_cluster_boot": ub_boot_fl,
        "flip_ub95_conservative_unique_patient_cp": clopper_pearson_upper_1sided(
            k_fl, max(1, n_pat_sk)
        ),
        "neg_to_pos_count": k_np,
        "neg_to_pos_rate": float(k_np) / max(1, n_sk),
        "neg_to_pos_ub95_cluster_boot": ub_boot_np,
        "pos_to_neg_count": k_pn,
        "pos_to_neg_rate": float(k_pn) / max(1, n_sk),
        "pos_to_neg_ub95_cluster_boot": ub_boot_pn,
        "would_be_beneficial_flips": int(np.sum(ben)),
        "would_be_harmful_flips": int(np.sum(harm)),
    }


def run_block_5_v22() -> dict[str, Any]:
    out_file = PARTIAL_V22 / "block5_v2_2_raceblind_diabetes_scenario.json"
    if out_file.exists():
        print("Loaded cached Block 5 v2.2:", out_file)
        return json.loads(out_file.read_text(encoding="utf-8"))

    t0 = time.time()
    print("[Block 5 v2.2] Running Race-Blind Primary UCI Diabetes Scenario across BOTH splits...")
    coh_rb = load_fine_diabetes_discharge(
        include_stress_channels=False, include_race_in_base=False, seed=2026
    )
    tau_star = float(coh_rb.contract.cost_fp / (coh_rb.contract.cost_fp + coh_rb.contract.cost_fn))
    cq0 = coh_rb.per_instance_query_cost

    # --- Split 1: Canonical 55/15/15/15 Single Split (N_test = 14,909, matching Table 4) ---
    sp_can_df = coh_rb.X_base.copy()
    sp_can_df["__y__"] = coh_rb.y
    sp_can_df["__grp__"] = coh_rb.group_ids
    sp_can = freeze_split(sp_can_df, "__y__", group_col="__grp__", seed=2026)
    tr_c, ca_c, te_c = (
        np.asarray(sp_can.train_idx, dtype=int),
        np.asarray(sp_can.cal_idx, dtype=int),
        np.asarray(sp_can.test_idx, dtype=int),
    )
    oof_c, vm_c = _fit_oof_stage1_worker(
        coh_rb.X_base.iloc[tr_c].reset_index(drop=True),
        coh_rb.y[tr_c],
        coh_rb.base_semantic_types,
        coh_rb.group_ids[tr_c],
        None,
        seed=2026,
        n_jobs=4,
    )
    pil_c = sample_two_phase_pilot(
        coh_rb.y[tr_c], oof_c, vm_c, pilot_fraction=0.15, design="neyman", seed=2026
    )
    sel_c = discover_sequential_conditional_gain(
        coh_rb.X_base.iloc[tr_c].reset_index(drop=True),
        coh_rb.Z_pool.iloc[tr_c].reset_index(drop=True),
        coh_rb.y[tr_c],
        coh_rb.candidates,
        oof_c,
        pil_c,
        max_budget=coh_rb.acquisition_budget,
        base_semantic_types=coh_rb.base_semantic_types,
        groups_train=coh_rb.group_ids[tr_c],
        seed=2026,
    )
    scols_c = [c for ch in sel_c.selected_channels for c in ch.columns]
    p1_ca_c, p1_te_c = _fit_downstream_gbdt_worker(
        coh_rb.X_base.iloc[tr_c],
        coh_rb.y[tr_c],
        coh_rb.X_base.iloc[ca_c],
        coh_rb.y[ca_c],
        coh_rb.X_base.iloc[te_c],
        coh_rb.base_semantic_types,
        seed=2026,
        n_jobs=4,
    )
    p2_ca_c, p2_te_c = _fit_downstream_gbdt_worker(
        pd.concat([coh_rb.X_base.iloc[tr_c], coh_rb.Z_pool.iloc[tr_c][scols_c]], axis=1),
        coh_rb.y[tr_c],
        pd.concat([coh_rb.X_base.iloc[ca_c], coh_rb.Z_pool.iloc[ca_c][scols_c]], axis=1),
        coh_rb.y[ca_c],
        pd.concat([coh_rb.X_base.iloc[te_c], coh_rb.Z_pool.iloc[te_c][scols_c]], axis=1),
        coh_rb.base_semantic_types,
        seed=2033,
        n_jobs=4,
    )
    rtr_c = StreamingSelectiveRouter(coh_rb.contract, cq0, mean_centered=True).fit(
        coh_rb.y[ca_c], p1_ca_c, p2_ca_c
    )
    q_c_auto = rtr_c.estimate_voi(p1_te_c) > cq0
    p_c_auto = np.where(q_c_auto, p2_te_c, p1_te_c)
    bu_te_c = -np.abs(p1_te_c - tau_star)
    q_c_bu = np.zeros(te_c.size, dtype=bool)
    q_c_bu[np.argsort(-bu_te_c, kind="mergesort")[: int(np.sum(q_c_auto))]] = True
    p_c_bu = np.where(q_c_bu, p2_te_c, p1_te_c)
    grp_te_c = coh_rb.group_ids[te_c]
    y_te_c = coh_rb.y[te_c]

    canonical_14909_summary = {
        "split_design": "55/15/15/15 single patient-grouped split (seed=2026, matching Table 4)",
        "n_test_encounters": int(te_c.size),
        "n_test_patients": int(np.unique(grp_te_c).size),
        "prevalence_test": float(np.mean(y_te_c)),
        "selected_ids": sel_c.selected_ids,
        "q_auto": float(np.mean(q_c_auto)),
        "voi_auto_vs_full100": paired_cluster_bootstrap_nb(
            y_te_c,
            p_c_auto,
            q_c_auto,
            p2_te_c,
            np.ones_like(q_c_auto),
            coh_rb.contract,
            cq0,
            grp_te_c,
        ),
        "voi_auto_vs_bu_matched": paired_cluster_bootstrap_nb(
            y_te_c, p_c_auto, q_c_auto, p_c_bu, q_c_bu, coh_rb.contract, cq0, grp_te_c
        ),
        "voi_auto_vs_stage1": paired_cluster_bootstrap_nb(
            y_te_c,
            p_c_auto,
            q_c_auto,
            p1_te_c,
            np.zeros_like(q_c_auto),
            coh_rb.contract,
            cq0,
            grp_te_c,
        ),
        "full100_vs_stage1": paired_cluster_bootstrap_nb(
            y_te_c,
            p2_te_c,
            np.ones_like(q_c_auto),
            p1_te_c,
            np.zeros_like(q_c_auto),
            coh_rb.contract,
            cq0,
            grp_te_c,
        ),
    }

    # --- Split 2: Outer 80/20 Patient-Disjoint Holdout (N_test = 19,999, R = 20 Inner Splits) ---
    outer_df = coh_rb.X_base.copy()
    outer_df["__y__"] = coh_rb.y
    outer_df["__grp__"] = coh_rb.group_ids
    outer_sp = freeze_split(
        outer_df,
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

    Xb_dev = coh_rb.X_base.iloc[dev_idx].reset_index(drop=True)
    Z_dev = coh_rb.Z_pool.iloc[dev_idx].reset_index(drop=True)
    y_dev = coh_rb.y[dev_idx]
    grp_dev = coh_rb.group_ids[dev_idx]

    Xb_te = coh_rb.X_base.iloc[te_idx].reset_index(drop=True)
    Z_te = coh_rb.Z_pool.iloc[te_idx].reset_index(drop=True)
    y_te = coh_rb.y[te_idx]
    grp_te = coh_rb.group_ids[te_idx]

    n_r = 20
    p1_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    p2_te_mat = np.zeros((n_r, te_idx.size), dtype=float)
    p_route_mat = np.zeros((n_r, te_idx.size), dtype=float)
    q_route_mat = np.zeros((n_r, te_idx.size), dtype=float)
    p_unc_mat = np.zeros((n_r, te_idx.size), dtype=float)
    q_unc_mat = np.zeros((n_r, te_idx.size), dtype=float)

    cal_p1_pool: list[np.ndarray] = []
    cal_p2_pool: list[np.ndarray] = []
    cal_y_pool: list[np.ndarray] = []
    per_split_nbs: dict[str, list[float]] = {
        "stage1": [],
        "gh_voi_auto": [],
        "unc_voi_auto": [],
        "bu_matched": [],
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
            coh_rb.base_semantic_types,
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
            coh_rb.candidates,
            oof_b,
            pil,
            max_budget=coh_rb.acquisition_budget,
            base_semantic_types=coh_rb.base_semantic_types,
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
            coh_rb.base_semantic_types,
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
            coh_rb.base_semantic_types,
            seed=r_seed + 50,
            n_jobs=4,
        )
        p1_te_mat[r_i] = p1_t
        p2_te_mat[r_i] = p2_t
        cal_p1_pool.append(p1_ca)
        cal_p2_pool.append(p2_ca)
        cal_y_pool.append(y_dev[ica])

        r_gh = StreamingSelectiveRouter(coh_rb.contract, cq0, mean_centered=True).fit(
            y_dev[ica], p1_ca, p2_ca
        )
        r_un = StreamingSelectiveRouter(coh_rb.contract, cq0, mean_centered=False).fit(
            y_dev[ica], p1_ca, p2_ca
        )
        q_gh = r_gh.estimate_voi(p1_t) > cq0
        q_un = r_un.estimate_voi(p1_t) > cq0
        p_gh = np.where(q_gh, p2_t, p1_t)
        p_un = np.where(q_un, p2_t, p1_t)

        bu_s = -np.abs(p1_t - tau_star)
        q_bu_s = np.zeros(te_idx.size, dtype=bool)
        k_s = int(np.sum(q_gh))
        if k_s > 0:
            q_bu_s[np.argsort(-bu_s, kind="mergesort")[:k_s]] = True
        p_bu_s = np.where(q_bu_s, p2_t, p1_t)

        p_route_mat[r_i] = p_gh
        q_route_mat[r_i] = q_gh.astype(float)
        p_unc_mat[r_i] = p_un
        q_unc_mat[r_i] = q_un.astype(float)

        cfp, cfn = coh_rb.contract.cost_fp, coh_rb.contract.cost_fn

        def _nb_s(
            pa: np.ndarray, qa: np.ndarray, _cfp: float = cfp, _cfn: float = cfn
        ) -> float:
            return float(
                1000.0
                * np.mean(
                    y_te * (pa >= tau_star)
                    - (_cfp / _cfn) * (1.0 - y_te) * (pa >= tau_star)
                    - (cq0 / _cfn) * qa
                )
            )

        per_split_nbs["stage1"].append(_nb_s(p1_t, np.zeros_like(p1_t)))
        per_split_nbs["gh_voi_auto"].append(_nb_s(p_gh, q_gh.astype(float)))
        per_split_nbs["unc_voi_auto"].append(_nb_s(p_un, q_un.astype(float)))
        per_split_nbs["bu_matched"].append(_nb_s(p_bu_s, q_bu_s.astype(float)))
        per_split_nbs["full_100"].append(_nb_s(p2_t, np.ones_like(p2_t)))

    p1_bag = np.mean(p1_te_mat, axis=0)
    p2_bag = np.mean(p2_te_mat, axis=0)
    cal_y_all = np.concatenate(cal_y_pool)
    cal_p1_all = np.concatenate(cal_p1_pool)
    cal_p2_all = np.concatenate(cal_p2_pool)

    bag_gh = StreamingSelectiveRouter(coh_rb.contract, cq0, mean_centered=True).fit(
        cal_y_all, cal_p1_all, cal_p2_all
    )
    bag_emp = StreamingSelectiveRouter(
        coh_rb.contract,
        cq0,
        quadrature_mode="empirical_quantile",
        n_emp_quantiles=33,
        mean_centered=True,
    ).fit(cal_y_all, cal_p1_all, cal_p2_all)
    bag_unc = StreamingSelectiveRouter(coh_rb.contract, cq0, mean_centered=False).fit(
        cal_y_all, cal_p1_all, cal_p2_all
    )

    voi_bag_gh = bag_gh.estimate_voi(p1_bag)
    q_bag_gh = voi_bag_gh > cq0
    p_bag_gh = np.where(q_bag_gh, p2_bag, p1_bag)
    q_bag_emp = bag_emp.estimate_voi(p1_bag) > cq0
    p_bag_emp = np.where(q_bag_emp, p2_bag, p1_bag)
    q_bag_unc = bag_unc.estimate_voi(p1_bag) > cq0
    p_bag_unc = np.where(q_bag_unc, p2_bag, p1_bag)

    bu_bag = -np.abs(p1_bag - tau_star)
    q_bag_bu_match = np.zeros(te_idx.size, dtype=bool)
    k_bag = int(np.sum(q_bag_gh))
    if k_bag > 0:
        q_bag_bu_match[np.argsort(-bu_bag, kind="mergesort")[:k_bag]] = True
    p_bag_bu_match = np.where(q_bag_bu_match, p2_bag, p1_bag)

    rng_b = np.random.default_rng(2026)
    u_cp = rng_b.uniform(size=cal_p1_all.size)
    u_tb = rng_b.uniform(size=p1_bag.size)
    _, q_bag_gh20, _, _ = calibrated_randomized_threshold_masks(
        bag_gh.voi_cal_, voi_bag_gh, 0.20, u_cp, u_tb
    )
    _, q_bag_bu20, _, _ = calibrated_randomized_threshold_masks(
        -np.abs(cal_p1_all - tau_star), bu_bag, 0.20, u_cp, u_tb
    )
    _, q_bag_top20, _, _ = calibrated_randomized_threshold_masks(
        cal_p1_all, p1_bag, 0.20, u_cp, u_tb
    )
    p_bag_gh20 = np.where(q_bag_gh20, p2_bag, p1_bag)
    p_bag_bu20 = np.where(q_bag_bu20, p2_bag, p1_bag)
    p_bag_top20 = np.where(q_bag_top20, p2_bag, p1_bag)

    p_sym_gh = np.mean(p_route_mat, axis=0)
    q_sym_gh = np.mean(q_route_mat, axis=0)
    p_sym_unc = np.mean(p_unc_mat, axis=0)
    q_sym_unc = np.mean(q_unc_mat, axis=0)

    q_zero = np.zeros(te_idx.size, dtype=float)
    q_ones = np.ones(te_idx.size, dtype=float)

    # c_query sweep on Bagged Diabetes scenario
    diabetes_cq_sweep: list[dict[str, Any]] = []
    for cq in (0.00, 0.01, 0.02, 0.04, 0.08, 0.16):
        q_cq = voi_bag_gh > cq
        p_cq = np.where(q_cq, p2_bag, p1_bag)
        k_cq = int(np.sum(q_cq))
        q_bu_cq = np.zeros(te_idx.size, dtype=bool)
        if k_cq > 0:
            q_bu_cq[np.argsort(-bu_bag, kind="mergesort")[:k_cq]] = True
        p_bu_cq = np.where(q_bu_cq, p2_bag, p1_bag)
        bt_f = paired_cluster_bootstrap_nb(
            y_te, p_cq, q_cq, p2_bag, q_ones, coh_rb.contract, cq, grp_te, n_boot=5000
        )
        bt_bu = paired_cluster_bootstrap_nb(
            y_te, p_cq, q_cq, p_bu_cq, q_bu_cq, coh_rb.contract, cq, grp_te, n_boot=5000
        )
        bt_s1 = paired_cluster_bootstrap_nb(
            y_te, p_cq, q_cq, p1_bag, q_zero, coh_rb.contract, cq, grp_te, n_boot=5000
        )
        diabetes_cq_sweep.append(
            {
                "c_query": cq,
                "q_auto": float(np.mean(q_cq)),
                "nb_voi_auto": bt_f["nb_num"],
                "nb_full_100": bt_f["nb_ref"],
                "nb_bu_matched": bt_bu["nb_ref"],
                "nb_stage1_0": bt_s1["nb_ref"],
                "voi_vs_full100": bt_f,
                "voi_vs_bu_matched": bt_bu,
                "voi_vs_stage1": bt_s1,
            }
        )

    # Referral exchange table across thresholds (reporting BOTH vs Full 100% AND vs Stage-1!)
    exchange_rows: list[dict[str, Any]] = []
    _, inv_pat = np.unique(grp_te, return_inverse=True)
    n_pat = int(np.max(inv_pat)) + 1
    cl_n = np.bincount(inv_pat, minlength=n_pat).astype(float)
    rng_ex = np.random.default_rng(2026)
    cnts_mat = rng_ex.multinomial(n_pat, np.full(n_pat, 1.0 / n_pat), size=10000)
    den_vec = np.maximum(1.0, cnts_mat @ cl_n)

    for tau_eval in (0.06, 1.0 / 9.0, 0.15, 0.20):
        cfn_imp = (1.0 - tau_eval) / tau_eval
        c_tmp = DecisionContract(
            task_name="Ex", target_precision=0.30, max_fpr_budget=0.05, cost_fp=1.0, cost_fn=cfn_imp
        )
        r_tmp = StreamingSelectiveRouter(c_tmp, cq0, mean_centered=True).fit(
            cal_y_all, cal_p1_all, cal_p2_all
        )
        q_t = r_tmp.estimate_voi(p1_bag) > cq0
        p_t = np.where(q_t, p2_bag, p1_bag)

        tp_voi = 1000.0 * ((y_te == 1) & (p_t >= tau_eval)).astype(float)
        tp_full = 1000.0 * ((y_te == 1) & (p2_bag >= tau_eval)).astype(float)
        tp_s1 = 1000.0 * ((y_te == 1) & (p1_bag >= tau_eval)).astype(float)
        fp_voi = 1000.0 * ((y_te == 0) & (p_t >= tau_eval)).astype(float)
        fp_full = 1000.0 * ((y_te == 0) & (p2_bag >= tau_eval)).astype(float)
        fp_s1 = 1000.0 * ((y_te == 0) & (p1_bag >= tau_eval)).astype(float)

        dtp_s1 = tp_voi - tp_s1
        sp_s1 = fp_s1 - fp_voi
        dtp_draws = (cnts_mat @ np.bincount(inv_pat, weights=dtp_s1, minlength=n_pat)) / den_vec
        sp_draws = (cnts_mat @ np.bincount(inv_pat, weights=sp_s1, minlength=n_pat)) / den_vec

        bt_vs_f = paired_cluster_bootstrap_nb(y_te, p_t, q_t, p2_bag, q_ones, c_tmp, cq0, grp_te)
        bt_vs_s = paired_cluster_bootstrap_nb(y_te, p_t, q_t, p1_bag, q_zero, c_tmp, cq0, grp_te)
        tp_m = float(np.mean(tp_voi))
        fp_m = float(np.mean(fp_voi))
        exchange_rows.append(
            {
                "tau": float(tau_eval),
                "breakeven_fp_per_tp": float(cfn_imp),
                "query_rate": float(np.mean(q_t)),
                "queries_skipped_pct": 100.0 * (1.0 - float(np.mean(q_t))),
                "tp_per_1k_voi": tp_m,
                "tp_per_1k_full100": float(np.mean(tp_full)),
                "tp_per_1k_s1": float(np.mean(tp_s1)),
                "fp_per_1k_voi": fp_m,
                "fp_per_1k_full100": float(np.mean(fp_full)),
                "fp_per_1k_s1": float(np.mean(fp_s1)),
                "ppv_voi": tp_m / max(1e-6, tp_m + fp_m),
                "delta_tp_vs_s1": float(np.mean(dtp_s1)),
                "delta_tp_vs_s1_ci95": [
                    float(np.quantile(dtp_draws, 0.025)),
                    float(np.quantile(dtp_draws, 0.975)),
                ],
                "referrals_spared_vs_s1": float(np.mean(sp_s1)),
                "referrals_spared_vs_s1_ci95": [
                    float(np.quantile(sp_draws, 0.025)),
                    float(np.quantile(sp_draws, 0.975)),
                ],
                "delta_nb_vs_full100": bt_vs_f["delta_nb"],
                "delta_nb_vs_full100_ci95": [bt_vs_f["ci95_low"], bt_vs_f["ci95_high"]],
                "delta_nb_vs_s1": bt_vs_s["delta_nb"],
                "delta_nb_vs_s1_ci95": [bt_vs_s["ci95_low"], bt_vs_s["ci95_high"]],
            }
        )

    skipped_audits = {
        "autonomous_voi": _skipped_bounds_summary(
            y_te, p1_bag, p2_bag, q_bag_gh, grp_te, tau_star, seed=2026
        ),
        "budget_20pct_voi": _skipped_bounds_summary(
            y_te, p1_bag, p2_bag, q_bag_gh20, grp_te, tau_star, seed=2027
        ),
    }

    # Subgroup audit on Primary Race-Blind Model (Stage-1, Full Stage-2, and Routed AUPRC)
    subgroup_audit: dict[str, Any] = {}
    for dim_name, sg_full in coh_rb.fairness_subgroups.items():
        sg_te = sg_full[te_idx]
        strata_rows: list[dict[str, Any]] = []
        d_means_f: list[float] = []
        d_vars_f: list[float] = []
        d_means_s: list[float] = []
        d_vars_s: list[float] = []
        for s_val in np.unique(sg_te):
            m = sg_te == s_val
            if int(np.sum(m)) < 50:
                continue
            ym = y_te[m]
            p1m = p1_bag[m]
            p2m = p2_bag[m]
            pm = p_bag_gh[m]
            qm = q_bag_gh[m]
            gm = grp_te[m]
            cal_m = compute_calibration_summary(ym, pm)
            bt_f = paired_cluster_bootstrap_nb(
                ym,
                pm,
                qm,
                p2m,
                np.ones_like(qm, dtype=float),
                coh_rb.contract,
                cq0,
                gm,
                n_boot=10000,
            )
            bt_s = paired_cluster_bootstrap_nb(
                ym,
                pm,
                qm,
                p1m,
                np.zeros_like(qm, dtype=float),
                coh_rb.contract,
                cq0,
                gm,
                n_boot=10000,
            )
            se_f = max(1e-4, (bt_f["ci95_high"] - bt_f["ci95_low"]) / 3.92)
            se_s = max(1e-4, (bt_s["ci95_high"] - bt_s["ci95_low"]) / 3.92)
            d_means_f.append(bt_f["delta_nb"])
            d_vars_f.append(se_f**2)
            d_means_s.append(bt_s["delta_nb"])
            d_vars_s.append(se_s**2)

            act_m = pm >= tau_star
            fnr = float(np.sum((ym == 1) & (~act_m))) / max(1, int(np.sum(ym == 1)))
            fpr = float(np.sum((ym == 0) & act_m)) / max(1, int(np.sum(ym == 0)))
            strata_rows.append(
                {
                    "stratum": str(s_val),
                    "n_enc": int(np.sum(m)),
                    "prevalence": float(np.mean(ym)),
                    "query_rate": float(np.mean(qm)),
                    "auprc_s1": float(average_precision_score(ym, p1m)),
                    "auprc_full_s2": float(average_precision_score(ym, p2m)),
                    "auprc_routed": float(average_precision_score(ym, pm)),
                    "delta_nb_vs_full100": bt_f["delta_nb"],
                    "vs_full100_ci95": [bt_f["ci95_low"], bt_f["ci95_high"]],
                    "delta_nb_vs_s1": bt_s["delta_nb"],
                    "vs_s1_ci95": [bt_s["ci95_low"], bt_s["ci95_high"]],
                    "citl": cal_m.cal_intercept,
                    "slope": cal_m.cal_slope,
                    "ici": cal_m.ici,
                    "fnr_at_tau": fnr,
                    "fpr_at_tau": fpr,
                }
            )

        def _cochran(ms: list[float], vs: list[float]) -> tuple[float, float]:
            if len(ms) < 2:
                return (0.0, 1.0)
            w = 1.0 / np.asarray(vs)
            mu = float(np.sum(w * np.asarray(ms)) / np.sum(w))
            q_st = float(np.sum(w * ((np.asarray(ms) - mu) ** 2)))
            return (q_st, float(chi2.sf(q_st, df=len(ms) - 1)))

        qf, pf = _cochran(d_means_f, d_vars_f)
        qs, ps = _cochran(d_means_s, d_vars_s)
        subgroup_audit[dim_name] = {
            "cochran_q_vs_full100": qf,
            "p_het_vs_full100": pf,
            "cochran_q_vs_s1": qs,
            "p_het_vs_s1": ps,
            "strata": strata_rows,
        }

    # Reference Race-Included outer-split summary from cached v2.1 Block 4 for transparency
    b4_v21 = json.loads(
        (V21_BLOCKS / "block4_clinical_diabetes_audit.json").read_text(encoding="utf-8")
    )

    out = {
        "elapsed_sec": round(time.time() - t0, 2),
        "primary_model_spec": "Race-Blind X_base (include_race_in_base=False; race post hoc)",
        "canonical_14909_split_race_blind": canonical_14909_summary,
        "outer_19999_r20_triad_race_blind": {
            "n_test_encounters": int(te_idx.size),
            "n_test_patients": int(np.unique(grp_te).size),
            "prevalence_test": float(np.mean(y_te)),
            "selected_channel_counts_r20": sel_freq,
            "bagged_voi_primary": {
                "q_bag_gh_auto": float(np.mean(q_bag_gh)),
                "q_bag_emp_auto": float(np.mean(q_bag_emp)),
                "q_bag_unc_auto": float(np.mean(q_bag_unc)),
                "gh_auto_vs_full100": paired_cluster_bootstrap_nb(
                    y_te, p_bag_gh, q_bag_gh, p2_bag, q_ones, coh_rb.contract, cq0, grp_te
                ),
                "gh_auto_vs_bu_matched": paired_cluster_bootstrap_nb(
                    y_te,
                    p_bag_gh,
                    q_bag_gh,
                    p_bag_bu_match,
                    q_bag_bu_match,
                    coh_rb.contract,
                    cq0,
                    grp_te,
                ),
                "gh20_vs_bu20": paired_cluster_bootstrap_nb(
                    y_te,
                    p_bag_gh20,
                    q_bag_gh20,
                    p_bag_bu20,
                    q_bag_bu20,
                    coh_rb.contract,
                    cq0,
                    grp_te,
                ),
                "gh20_vs_toprisk20": paired_cluster_bootstrap_nb(
                    y_te,
                    p_bag_gh20,
                    q_bag_gh20,
                    p_bag_top20,
                    q_bag_top20,
                    coh_rb.contract,
                    cq0,
                    grp_te,
                ),
                "gh_auto_vs_uncentered": paired_cluster_bootstrap_nb(
                    y_te, p_bag_gh, q_bag_gh, p_bag_unc, q_bag_unc, coh_rb.contract, cq0, grp_te
                ),
                "emp_auto_vs_gh_auto": paired_cluster_bootstrap_nb(
                    y_te, p_bag_emp, q_bag_emp, p_bag_gh, q_bag_gh, coh_rb.contract, cq0, grp_te
                ),
                "gh_auto_vs_stage1": paired_cluster_bootstrap_nb(
                    y_te, p_bag_gh, q_bag_gh, p1_bag, q_zero, coh_rb.contract, cq0, grp_te
                ),
                "full100_vs_stage1": paired_cluster_bootstrap_nb(
                    y_te, p2_bag, q_ones, p1_bag, q_zero, coh_rb.contract, cq0, grp_te
                ),
            },
            "symmetric_expected_policy": {
                "q_sym_gh": float(np.mean(q_sym_gh)),
                "gh_sym_vs_full100": paired_cluster_bootstrap_nb(
                    y_te, p_sym_gh, q_sym_gh, p2_bag, q_ones, coh_rb.contract, cq0, grp_te
                ),
                "gh_sym_vs_unc_sym": paired_cluster_bootstrap_nb(
                    y_te, p_sym_gh, q_sym_gh, p_sym_unc, q_sym_unc, coh_rb.contract, cq0, grp_te
                ),
                "gh_sym_vs_stage1": paired_cluster_bootstrap_nb(
                    y_te, p_sym_gh, q_sym_gh, p1_bag, q_zero, coh_rb.contract, cq0, grp_te
                ),
            },
            "per_split_unbagged": {
                k: {
                    "mean": float(np.mean(v)),
                    "se": float(np.std(v, ddof=1) / np.sqrt(len(v))),
                }
                for k, v in per_split_nbs.items()
            },
        },
        "c_query_sweep_race_blind": diabetes_cq_sweep,
        "referral_exchange_rates_race_blind": exchange_rows,
        "skipped_case_disagreement_bounds_race_blind": skipped_audits,
        "subgroup_equity_and_calibration_race_blind": subgroup_audit,
        "race_included_sensitivity_ref": b4_v21["r20_estimator_triad"]["bagged_voi_primary"],
        "hyperparameter_sensitivity_ref": b4_v21["hyperparameter_sensitivity"],
    }
    PARTIAL_V22.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[Block 5 v2.2] Completed in {out['elapsed_sec']}s -> {out_file}")
    return out


def main() -> None:
    t_start = time.time()
    PARTIAL_V22.mkdir(parents=True, exist_ok=True)
    b1 = run_block_1_v22(max_workers=8)
    b2 = run_block_2_v22(max_workers=8)
    b3 = run_block_3_v22(max_workers=3)
    b4 = run_block_4_v22()
    b5 = run_block_5_v22()

    master = {
        "version": "2.2.0",
        "v1_preprints_doi": "10.20944/preprints202611.2517.v1",
        "total_elapsed_sec": round(time.time() - t_start, 2),
        "block1_weak_signal_and_twins": b1,
        "block2_full_120split_leakfree_stress": b2,
        "block3_stageb_comparators_and_feesweep": b3,
        "block4_cfbh_shift_and_wcs": b4,
        "block5_raceblind_diabetes_scenario": b5,
    }
    OUT_JSON_V22.write_text(json.dumps(master, indent=2), encoding="utf-8")
    el_sec = master["total_elapsed_sec"]
    print(f"\nALL PAPER 1 v2.2 REMEDIATION BLOCKS COMPLETE IN {el_sec}s -> {OUT_JSON_V22}")


if __name__ == "__main__":
    main()
