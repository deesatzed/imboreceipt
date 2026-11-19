"""Adversarial simulation suite probing strengths AND failure modes of CREDO (Review Sec. 6.D)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.types import freeze_split
from imboreceipt.discover.selector import (
    CandidateChannelSpec,
    _bernoulli_loglik,
    _fit_oof_probs_on_splits,
    score_channels_baseline_strategy,
    score_channels_conditional_gain,
    score_channels_credo_reg,
    select_channels_within_budget,
)
from imboreceipt.discover.strata import (
    build_consensus_residual_strata,
    make_leak_free_cv_splits,
    sample_case_cohort_pilot,
)
from imboreceipt.frontier.baselines import fit_single_calibrated_gbdt
from imboreceipt.lattice.foveation import BinaryConformalFoveator


def _eval_selector_hit(
    chosen: list[CandidateChannelSpec],
    target_set: set[str],
) -> float:
    if not chosen:
        return 0.0
    return float(sum(1 for c in chosen if c.channel_id in target_set) / len(chosen))


def run_sim1_pure_redundant_echo(seed: int = 42, n: int = 8000) -> dict[str, Any]:
    """Scenario 1: Redundant linear & nonlinear echoes of X_base vs weak orthogonal signal."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    z_orth = rng.normal(0.0, 1.0, n)
    eta = expit(-4.3 + 1.35 * x1 - 0.95 * x2 + 0.95 * z_orth)
    y = rng.binomial(1, eta).astype(int)

    X_base = pd.DataFrame({"x1": x1, "x2": x2})
    Z_pool = pd.DataFrame(
        {
            "echo_lin": 1.4 * x1 - 1.0 * x2 + rng.normal(0.0, 0.20, n),
            "echo_nonlin": np.tanh(1.5 * x1) + 0.5 * (x2**2) + rng.normal(0.0, 0.20, n),
            "orth_true": z_orth + rng.normal(0.0, 0.35, n),
            "noise_decoy": rng.normal(0.0, 1.0, n),
        }
    )
    cands = [
        CandidateChannelSpec("ch_echo_lin", ("echo_lin",), 1.0, "redundant_echo"),
        CandidateChannelSpec("ch_echo_nonlin", ("echo_nonlin",), 1.0, "redundant_echo"),
        CandidateChannelSpec("ch_orth_true", ("orth_true",), 1.0, "orthogonal"),
        CandidateChannelSpec("ch_noise", ("noise_decoy",), 1.0, "decoy"),
    ]
    spec_map = {c.channel_id: c for c in cands}
    contract = DecisionContract(
        task_name="Sim1_Redundant", target_precision=0.45, max_fpr_budget=0.01
    )

    strata = build_consensus_residual_strata(X_base, y, seed=seed)
    pilot = sample_case_cohort_pilot(y, strata.oof_consensus_prob, seed=seed)

    cmi_sc = score_channels_conditional_gain(X_base, Z_pool, y, cands, strata, pilot, seed=seed)
    reg_sc = score_channels_credo_reg(X_base, Z_pool, y, cands, strata, pilot, contract, seed=seed)
    mi_sc = score_channels_baseline_strategy(
        "global_mi", X_base, Z_pool, y, cands, strata, seed=seed
    )
    pmi_sc = score_channels_baseline_strategy(
        "pilot_mi", X_base, Z_pool, y, cands, strata, pilot=pilot, seed=seed
    )

    truth = {"ch_orth_true"}
    return {
        "scenario": "1_pure_redundant_echo",
        "prevalence": float(y.mean()),
        "pilot_fraction": pilot.fraction_of_cohort_measured,
        "global_mi_precision": _eval_selector_hit(
            select_channels_within_budget(mi_sc, spec_map, 1.0), truth
        ),
        "pilot_mi_precision": _eval_selector_hit(
            select_channels_within_budget(pmi_sc, spec_map, 1.0), truth
        ),
        "pilot_cmi_precision": _eval_selector_hit(
            select_channels_within_budget(cmi_sc, spec_map, 1.0, require_fdr=True), truth
        ),
        "credo_reg_precision": _eval_selector_hit(
            select_channels_within_budget(reg_sc, spec_map, 1.0, require_fdr=True), truth
        ),
    }


def run_sim2_correlated_informative_residual(seed: int = 42, n: int = 8000) -> dict[str, Any]:
    """Scenario 2: Highly correlated channel (r ~ 0.92 with X_base) carrying strong residual signal.

    Tests whether aggressive collinearity discounting discards useful residual information inside a
    channel correlated with X_base, whereas conditional log-loss gain G_m recovers it cleanly.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    u_res = rng.normal(0.0, 1.0, n)
    u_weak = rng.normal(0.0, 1.0, n)

    eta = expit(-4.5 + 1.15 * x1 + 0.5 * x2 + 1.55 * u_res + 0.45 * u_weak)
    y = rng.binomial(1, eta).astype(int)

    # Correlated channel shares 0.92 loading on x1, PLUS carries 0.42 * u_res!
    z_corr_informative = 0.92 * x1 + 0.42 * u_res + rng.normal(0.0, 0.08, n)
    r_emp = float(np.corrcoef(x1, z_corr_informative)[0, 1])

    X_base = pd.DataFrame({"x1": x1, "x2": x2})
    Z_pool = pd.DataFrame(
        {
            "z_corr_res": z_corr_informative,
            "z_orth_weak": u_weak + rng.normal(0.0, 0.5, n),
            "z_pure_echo": 0.95 * x1 + rng.normal(0.0, 0.25, n),
        }
    )
    cands = [
        CandidateChannelSpec("ch_corr_strong_res", ("z_corr_res",), 1.0, "correlated_residual"),
        CandidateChannelSpec("ch_orth_weak", ("z_orth_weak",), 1.0, "weak_orthogonal"),
        CandidateChannelSpec("ch_pure_echo", ("z_pure_echo",), 1.0, "redundant_echo"),
    ]
    spec_map = {c.channel_id: c for c in cands}
    contract = DecisionContract(
        task_name="Sim2_CorrelatedResidual", target_precision=0.50, max_fpr_budget=0.01
    )

    strata = build_consensus_residual_strata(X_base, y, seed=seed)
    pilot = sample_case_cohort_pilot(y, strata.oof_consensus_prob, seed=seed)

    cmi_sc = score_channels_conditional_gain(X_base, Z_pool, y, cands, strata, pilot, seed=seed)
    reg_sc = score_channels_credo_reg(X_base, Z_pool, y, cands, strata, pilot, contract, seed=seed)

    # Naive collinearity-penalized score (shows failure of penalizing r^2(Z, X_base) directly)
    naive_col_penalized: dict[str, float] = {}
    for rec in cmi_sc:
        col_name = spec_map[rec.channel_id].columns[0]
        r2 = (
            float(
                np.corrcoef(
                    x1[pilot.sampled_indices], Z_pool[col_name].iloc[pilot.sampled_indices]
                )[0, 1]
            )
            ** 2
        )
        naive_col_penalized[rec.channel_id] = rec.raw_score * max(0.0, 1.0 - 0.95 * r2)
    naive_top = max(naive_col_penalized, key=lambda k: naive_col_penalized[k])

    truth = {"ch_corr_strong_res"}
    return {
        "scenario": "2_correlated_informative_residual",
        "correlation_with_xbase": r_emp,
        "pilot_cmi_top1_correct": float(
            select_channels_within_budget(cmi_sc, spec_map, 1.0)[0].channel_id in truth
        ),
        "credo_reg_top1_correct": float(
            select_channels_within_budget(reg_sc, spec_map, 1.0)[0].channel_id in truth
        ),
        "naive_collinearity_penalty_top1_correct": float(naive_top in truth),
    }


def run_sim3_pure_interaction_xor(seed: int = 42, n: int = 9000) -> dict[str, Any]:
    """Scenario 3: Pure interaction / XOR channel (Z marginally independent of Y AND of S(X_base)).

    Demonstrates why conditioning only on compressed scalar score S(X_base) fails on symmetric
    interactions, whereas full-feature [X_base, Z_m] conditional log-loss gain succeeds.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    z_xor = rng.normal(0.0, 1.0, n)
    z_weak = rng.normal(0.0, 1.0, n)

    # Exact parity XOR interaction: sign(x1) * sign(z_xor) in {-1, +1} has IDENTICAL conditional
    # distribution given z_xor alone (so exact I(Y; Z_xor) == 0, unlike continuous x1*z which leaks
    # |z| via Jensen's inequality in the rare tail!) AND exact I(Y; Z_xor | S(X_base)) == 0!
    bx = np.where(x1 >= 0.0, 1.0, -1.0)
    bz = np.where(z_xor >= 0.0, 1.0, -1.0)
    eta = expit(-4.3 + 0.55 * (x1**2 - 1.0) + 1.95 * (bx * bz) + 0.45 * z_weak)
    y = rng.binomial(1, eta).astype(int)

    X_base = pd.DataFrame({"x1": x1, "x2": x2})
    Z_pool = pd.DataFrame(
        {
            "z_xor": z_xor,
            "z_weak": z_weak,
            "z_noise": rng.normal(0.0, 1.0, n),
        }
    )
    cands = [
        CandidateChannelSpec("ch_xor_interaction", ("z_xor",), 1.0, "interaction"),
        CandidateChannelSpec("ch_weak_main", ("z_weak",), 1.0, "weak_main"),
        CandidateChannelSpec("ch_noise", ("z_noise",), 1.0, "decoy"),
    ]
    spec_map = {c.channel_id: c for c in cands}

    strata = build_consensus_residual_strata(X_base, y, seed=seed)
    pilot = sample_case_cohort_pilot(y, strata.oof_consensus_prob, seed=seed)

    # Full [X_base, Z_m] conditional gain
    cmi_full = score_channels_conditional_gain(X_base, Z_pool, y, cands, strata, pilot, seed=seed)
    mi_sc = score_channels_baseline_strategy(
        "global_mi", X_base, Z_pool, y, cands, strata, seed=seed
    )

    # Compressed-score-only conditioning [base_logit, Z_m] (loses sign(x1)!)
    idx = pilot.sampled_indices
    yp = y[idx]
    wp = pilot.sampling_weights
    splits = make_leak_free_cv_splits(yp, n_folds=4, seed=seed)
    s_only_df = pd.DataFrame(
        {"s_logit": logit(np.clip(strata.oof_consensus_prob[idx], 1e-6, 1.0 - 1e-6))}
    )
    p0_s, v0_s = _fit_oof_probs_on_splits(s_only_df, yp, splits, sample_weight=wp, seed=seed)
    ll0_s = _bernoulli_loglik(yp, p0_s)
    comp_gains: dict[str, float] = {}
    for c in cands:
        aug_s = pd.concat(
            [s_only_df, Z_pool.iloc[idx][list(c.columns)].reset_index(drop=True)], axis=1
        )
        pm_s, vm_s = _fit_oof_probs_on_splits(aug_s, yp, splits, sample_weight=wp, seed=seed)
        ev = v0_s & vm_s
        comp_gains[c.channel_id] = float(
            np.sum(wp[ev] * (_bernoulli_loglik(yp, pm_s) - ll0_s)[ev]) / np.sum(wp[ev])
        )
    comp_top = max(comp_gains, key=lambda k: comp_gains[k])

    truth = {"ch_xor_interaction"}
    return {
        "scenario": "3_pure_interaction_xor",
        "global_mi_top1_correct": float(
            select_channels_within_budget(mi_sc, spec_map, 1.0)[0].channel_id in truth
        ),
        "compressed_score_only_top1_correct": float(comp_top in truth),
        "full_xbase_pilot_cmi_top1_correct": float(
            select_channels_within_budget(cmi_full, spec_map, 1.0)[0].channel_id in truth
        ),
    }


def run_sim4_misspecified_baseline_score(seed: int = 42, n: int = 5000) -> dict[str, Any]:
    """Scenario 4: Severely misspecified/noisy Stage-1 pilot baseline score (Failure-Mode Probe).

    Evaluates what happens when the pilot stratification score is corrupted by severe observation
    noise vs well-specified: shows degradation of score-stratified matching back toward unmatched
    random sampling and leakage of nonlinear X_base shadows if baseline model is linear-only.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    z_true = rng.normal(0.0, 1.0, n)
    # True risk has sharp non-monotone ring + z_true in the ring
    ring = np.exp(-((x1**2 + x2**2 - 2.2) ** 2))
    eta = expit(-4.6 + 2.4 * ring + 1.35 * ring * (z_true > 0.5).astype(float))
    y = rng.binomial(1, eta).astype(int)

    X_base = pd.DataFrame({"x1": x1, "x2": x2})
    Z_pool = pd.DataFrame(
        {
            "z_true": z_true,
            "z_ring_shadow": ring + rng.normal(0.0, 0.15, n),
            "z_noise": rng.normal(0.0, 1.0, n),
        }
    )
    cands = [
        CandidateChannelSpec("ch_true_signal", ("z_true",), 1.0, "orthogonal"),
        CandidateChannelSpec("ch_nonlin_xbase_shadow", ("z_ring_shadow",), 1.0, "xbase_shadow"),
        CandidateChannelSpec("ch_noise", ("z_noise",), 1.0, "decoy"),
    ]
    spec_map = {c.channel_id: c for c in cands}

    # Case A: Well-specified GBDT+GLM consensus strata + GBDT CMI
    strata_good = build_consensus_residual_strata(X_base, y, seed=seed)
    pilot_good = sample_case_cohort_pilot(y, strata_good.oof_consensus_prob, seed=seed)
    cmi_good = score_channels_conditional_gain(
        X_base, Z_pool, y, cands, strata_good, pilot_good, seed=seed
    )

    # Case B: Badly misspecified linear-only baseline score AND linear-only conditional model!
    corrupt_scores = expit(-3.5 + 0.1 * x1 + rng.normal(0.0, 2.0, n))
    pilot_corrupt = sample_case_cohort_pilot(y, corrupt_scores, seed=seed)
    idx_c = pilot_corrupt.sampled_indices
    yc = y[idx_c]
    wc = pilot_corrupt.sampling_weights
    splits_c = make_leak_free_cv_splits(yc, n_folds=4, seed=seed)
    p0_lin, v0_l = _fit_oof_probs_on_splits(
        X_base.iloc[idx_c].reset_index(drop=True), yc, splits_c, wc, use_gbdt=False, seed=seed
    )
    ll0_l = _bernoulli_loglik(yc, p0_lin)
    lin_gains: dict[str, float] = {}
    for c in cands:
        aug_l = pd.concat(
            [
                X_base.iloc[idx_c].reset_index(drop=True),
                Z_pool.iloc[idx_c][list(c.columns)].reset_index(drop=True),
            ],
            axis=1,
        )
        pm_l, vm_l = _fit_oof_probs_on_splits(aug_l, yc, splits_c, wc, use_gbdt=False, seed=seed)
        ev = v0_l & vm_l
        lin_gains[c.channel_id] = float(
            np.sum(wc[ev] * (_bernoulli_loglik(yc, pm_l) - ll0_l)[ev]) / np.sum(wc[ev])
        )
    misspec_top = max(lin_gains, key=lambda k: lin_gains[k])

    return {
        "scenario": "4_misspecified_baseline_failure_probe",
        "well_specified_gbdt_cmi_top1_true": float(
            select_channels_within_budget(cmi_good, spec_map, 1.0)[0].channel_id == "ch_true_signal"
        ),
        "misspecified_linear_cmi_top1_true": float(misspec_top == "ch_true_signal"),
        "misspecified_fooled_by_nonlinear_shadow": float(misspec_top == "ch_nonlin_xbase_shadow"),
    }


def run_sim5_sparse_trigger_m_sensitivity(seed: int = 42, n: int = 16000) -> dict[str, Any]:
    """Scenario 5: Ultra-rare prevalence with sparse trigger; sensitivity over m in {1..16}."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    sparse_trig = rng.binomial(1, 0.03, n)
    eta = expit(-5.9 + 1.05 * x1 + 0.55 * x2 + 2.85 * sparse_trig)
    y = rng.binomial(1, eta).astype(int)

    df_all = pd.DataFrame({"x1": x1, "x2": x2, "__y__": y})
    sp = freeze_split(df_all, "__y__", seed=seed)
    tr, cal, tst = np.array(sp.train_idx), np.array(sp.cal_idx), np.array(sp.test_idx)

    X_base = pd.DataFrame({"x1": x1, "x2": x2})
    Z_pool = pd.DataFrame(
        {
            "sparse_biomarker": sparse_trig + rng.normal(0.0, 0.12, n),
            "echo_x1": x1 + rng.normal(0.0, 0.25, n),
            "decoy1": rng.normal(0.0, 1.0, n),
            "decoy2": rng.normal(0.0, 1.0, n),
        }
    )
    cands = [
        CandidateChannelSpec("ch_sparse_true", ("sparse_biomarker",), 1.0, "sparse_trigger"),
        CandidateChannelSpec("ch_echo", ("echo_x1",), 1.0, "redundant_echo"),
        CandidateChannelSpec("ch_decoy1", ("decoy1",), 1.0, "decoy"),
        CandidateChannelSpec("ch_decoy2", ("decoy2",), 1.0, "decoy"),
    ]
    spec_map = {c.channel_id: c for c in cands}

    strata = build_consensus_residual_strata(X_base.iloc[tr], y[tr], seed=seed)
    by_m: dict[str, dict[str, float]] = {}
    for m_val in (1, 2, 4, 8, 16):
        p_m = sample_case_cohort_pilot(
            y[tr], strata.oof_consensus_prob, controls_per_positive=m_val, seed=seed
        )
        sc_m = score_channels_conditional_gain(
            X_base.iloc[tr], Z_pool.iloc[tr], y[tr], cands, strata, p_m, seed=seed
        )
        chosen = select_channels_within_budget(sc_m, spec_map, 1.0)
        z_cols = [col for c in chosen for col in c.columns]
        arm = fit_single_calibrated_gbdt(
            pd.concat(
                [
                    X_base.iloc[tr].reset_index(drop=True),
                    Z_pool.iloc[tr][z_cols].reset_index(drop=True),
                ],
                axis=1,
            ),
            y[tr],
            pd.concat(
                [
                    X_base.iloc[cal].reset_index(drop=True),
                    Z_pool.iloc[cal][z_cols].reset_index(drop=True),
                ],
                axis=1,
            ),
            y[cal],
            seed=seed,
        )
        ap_tst = float(
            average_precision_score(
                y[tst],
                arm.predict_proba(
                    pd.concat(
                        [
                            X_base.iloc[tst].reset_index(drop=True),
                            Z_pool.iloc[tst][z_cols].reset_index(drop=True),
                        ],
                        axis=1,
                    )
                ),
            )
        )
        by_m[f"m={m_val}"] = {
            "pilot_fraction": p_m.fraction_of_cohort_measured,
            "top1_hit": float(chosen[0].channel_id == "ch_sparse_true"),
            "test_auprc": ap_tst,
        }

    return {
        "scenario": "5_sparse_trigger_m_sensitivity",
        "prevalence": float(y.mean()),
        "n_pos_train": int(y[tr].sum()),
        "m_sensitivity": by_m,
    }


def run_sim6_mid_risk_informative_routing(seed: int = 42, n: int = 12000) -> dict[str, Any]:
    """Scenario 6: Mid-risk region is informative while highest-risk region is irreducible!

    Designed explicitly to reveal where BOTH Top-Risk-K and CREDO Upper-Tail fail:
    - Upper-risk pocket (10% of cohort, p1 ~ 0.86): outcome is irreducible at 86% risk (Stage-2 Z
      has zero effect here), yet target_precision = 0.95 keeps D_q(x) > 0!
    - Mid-risk pocket (10% of cohort, p1 ~ 0.24): Stage-2 Z cleanly separates true cases (~92%)
      from controls (~2%)!
    - Low-risk bulk (80% of cohort, p1 ~ 0.002): benign background.
    Under scarce budget K ~ 10%, Top-Risk-K and CREDO Upper-Tail spend their budget on the
    irreducible upper pocket, whereas VOI-K and Literal Conformal {0,1} query the informative
    mid-risk pocket and achieve substantially higher AUPRC and Net Utility!
    """
    rng = np.random.default_rng(seed)
    regime = rng.choice([0, 1, 2], size=n, p=[0.80, 0.10, 0.10])
    x_score = np.where(
        regime == 2,
        rng.normal(2.8, 0.20, n),
        np.where(regime == 1, rng.normal(1.0, 0.20, n), rng.normal(-1.6, 0.45, n)),
    )
    z_mid = rng.normal(0.0, 1.0, n)

    true_prob = np.where(
        regime == 2,
        0.86,  # Irreducible in Stage 2 (Z perp Y | X here), yet < target_precision=0.95!
        np.where(
            regime == 1,
            expit(-1.5 + 3.2 * z_mid),  # Strongly resolved by Stage 2!
            0.002,
        ),
    )
    y = rng.binomial(1, true_prob).astype(int)

    df_all = pd.DataFrame({"x": x_score, "r": regime.astype(float), "z": z_mid, "__y__": y})
    sp = freeze_split(df_all, "__y__", seed=seed)
    tr, cal, val, tst = (
        np.array(sp.train_idx),
        np.array(sp.cal_idx),
        np.array(sp.val_idx),
        np.array(sp.test_idx),
    )

    Xb = df_all[["x", "r"]]
    XZ = df_all[["x", "r", "z"]]
    s1 = fit_single_calibrated_gbdt(Xb.iloc[tr], y[tr], Xb.iloc[cal], y[cal], seed=seed)
    s2 = fit_single_calibrated_gbdt(XZ.iloc[tr], y[tr], XZ.iloc[cal], y[cal], seed=seed)

    contract = DecisionContract(
        task_name="Sim6_MidRiskInformative",
        target_precision=0.95,
        max_fpr_budget=0.02,
        cost_fp=1.0,
        cost_fn=4.0,
        graded_action_cost_fp=None,
    )
    # Set Mondrian alphas so {1} covers regime 2, {0,1} covers regime 1, and total {1} U {0,1} is
    # ~11-12% when alpha_pos is tight (0.28) and alpha_neg is 0.11:
    fov = BinaryConformalFoveator(alpha_neg=0.11, alpha_pos=0.28, stage2_query_cost_per_case=0.05)
    fov.fit(s1.predict_proba(Xb.iloc[cal]), y[cal])
    rep = fov.evaluate_foveation(
        y[tst],
        s1.predict_proba(Xb.iloc[tst]),
        s2.predict_proba(XZ.iloc[tst]),
        contract,
        stage1_val_probs=s1.predict_proba(Xb.iloc[val]),
        stage2_val_probs=s2.predict_proba(XZ.iloc[val]),
        y_val=y[val],
        fixed_query_rate=0.10,
    )

    return {
        "scenario": "6_mid_risk_informative_routing_failure",
        "query_rate": rep.foveated_stage2_query_rate,
        "stage1_auprc": rep.stage1_only_auprc,
        "credo_foveation_auprc": rep.credo_deficit_foveated_auprc,
        "top_risk_k_auprc": rep.top_risk_k_auprc,
        "literal_conformal_01_auprc": rep.literal_conformal_01_auprc,
        "uncertainty_k_auprc": rep.uncertainty_k_auprc,
        "voi_k_auprc": rep.voi_k_auprc,
        "random_k_auprc": rep.random_routing_auprc,
        "full_stage2_auprc": rep.full_stage2_auprc,
        "voi_net_utility": rep.net_utility_voi_k_per_1k,
        "credo_net_utility": rep.net_utility_foveated_net_of_query_cost_per_1k,
        "top_risk_net_utility": rep.net_utility_top_risk_k_per_1k,
    }


def run_sim7_unequal_costs_and_drift(seed: int = 42, n: int = 10000) -> dict[str, Any]:
    """Scenario 7: Unequal channel costs + non-stationary temporal distribution shift.

    1. Cost trade-off under budget=2.0: one expensive channel (cost=2.0, gain=0.05) vs two cheap
       complementary channels (cost=1.0 each, combined gain=0.085) -> tests score_per_cost!
    2. Non-exchangeable temporal drift in final test era: shifts base scores upward in late era,
       demonstrating empirical gap between nominal (1 - alpha) and realized conformal coverage.
    """
    rng = np.random.default_rng(seed)
    t_time = np.linspace(0.0, 1.0, n)
    # Temporal covariate shift in final 20% of timeline!
    late_shift = np.maximum(0.0, (t_time - 0.70) * 3.2)
    x1 = rng.normal(0.0, 1.0, n) + late_shift
    z_exp = rng.normal(0.0, 1.0, n)
    z_cheap1 = rng.normal(0.0, 1.0, n)
    z_cheap2 = rng.normal(0.0, 1.0, n)

    eta = expit(-4.2 + 0.9 * (x1 - late_shift) + 1.05 * z_exp + 0.95 * z_cheap1 + 0.95 * z_cheap2)
    y = rng.binomial(1, eta).astype(int)

    df_all = pd.DataFrame({"x1": x1, "t": t_time, "__y__": y})
    sp = freeze_split(df_all, "__y__", time_col="t", seed=seed)
    tr, cal, val, tst = (
        np.array(sp.train_idx),
        np.array(sp.cal_idx),
        np.array(sp.val_idx),
        np.array(sp.test_idx),
    )

    X_base = pd.DataFrame({"x1": x1})
    Z_pool = pd.DataFrame({"z_exp": z_exp, "z_c1": z_cheap1, "z_c2": z_cheap2})
    cands = [
        CandidateChannelSpec(
            "ch_expensive_single", ("z_exp",), acquisition_cost=2.0, mechanism_tag="expensive"
        ),
        CandidateChannelSpec(
            "ch_cheap_pair_1", ("z_c1",), acquisition_cost=1.0, mechanism_tag="cheap_1"
        ),
        CandidateChannelSpec(
            "ch_cheap_pair_2", ("z_c2",), acquisition_cost=1.0, mechanism_tag="cheap_2"
        ),
    ]
    spec_map = {c.channel_id: c for c in cands}

    strata = build_consensus_residual_strata(
        X_base.iloc[tr], y[tr], time_order=t_time[tr], seed=seed
    )
    pilot = sample_case_cohort_pilot(
        y[tr], strata.oof_consensus_prob, oof_valid_mask=strata.oof_valid_mask, seed=seed
    )
    cmi_sc = score_channels_conditional_gain(
        X_base.iloc[tr],
        Z_pool.iloc[tr],
        y[tr],
        cands,
        strata,
        pilot,
        time_order_train=t_time[tr],
        seed=seed,
    )
    chosen_cost_aware = select_channels_within_budget(cmi_sc, spec_map, max_budget=2.0)

    # Compare test AP of cost-aware pair vs single expensive channel
    def _eval_cols(cols: list[str]) -> tuple[float, np.ndarray, np.ndarray]:
        arm = fit_single_calibrated_gbdt(
            pd.concat(
                [
                    X_base.iloc[tr].reset_index(drop=True),
                    Z_pool.iloc[tr][cols].reset_index(drop=True),
                ],
                axis=1,
            ),
            y[tr],
            pd.concat(
                [
                    X_base.iloc[cal].reset_index(drop=True),
                    Z_pool.iloc[cal][cols].reset_index(drop=True),
                ],
                axis=1,
            ),
            y[cal],
            seed=seed,
        )
        vp = arm.predict_proba(
            pd.concat(
                [
                    X_base.iloc[val].reset_index(drop=True),
                    Z_pool.iloc[val][cols].reset_index(drop=True),
                ],
                axis=1,
            )
        )
        tp = arm.predict_proba(
            pd.concat(
                [
                    X_base.iloc[tst].reset_index(drop=True),
                    Z_pool.iloc[tst][cols].reset_index(drop=True),
                ],
                axis=1,
            )
        )
        return float(average_precision_score(y[tst], tp)), vp, tp

    ap_pair, vp_pair, tp_pair = _eval_cols(["z_c1", "z_c2"])
    ap_exp, _, _ = _eval_cols(["z_exp"])

    s1 = fit_single_calibrated_gbdt(X_base.iloc[tr], y[tr], X_base.iloc[cal], y[cal], seed=seed)
    fov = BinaryConformalFoveator(alpha_neg=0.20, alpha_pos=0.20)
    fov.fit(s1.predict_proba(X_base.iloc[cal]), y[cal])
    rep = fov.evaluate_foveation(
        y[tst],
        s1.predict_proba(X_base.iloc[tst]),
        tp_pair,
        DecisionContract(task_name="Sim7_Drift", target_precision=0.45, max_fpr_budget=0.02),
        stage1_val_probs=s1.predict_proba(X_base.iloc[val]),
        stage2_val_probs=vp_pair,
        y_val=y[val],
    )

    return {
        "scenario": "7_unequal_costs_and_temporal_drift",
        "chosen_channels": [c.channel_id for c in chosen_cost_aware],
        "selected_complementary_pair": bool(
            {c.channel_id for c in chosen_cost_aware} == {"ch_cheap_pair_1", "ch_cheap_pair_2"}
        ),
        "test_auprc_cheap_pair": ap_pair,
        "test_auprc_expensive_single": ap_exp,
        "nominal_class0_coverage_target": 0.80,
        "realized_class0_coverage_under_drift": rep.stage1_class0_coverage,
        "drift_coverage_shortfall": max(0.0, 0.80 - rep.stage1_class0_coverage),
    }


def run_adversarial_simulation_suite(
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
) -> dict[str, Any]:
    """Run all 7 adversarial simulation regimes across ``seeds`` and aggregate summaries."""
    sim_fns = [
        run_sim1_pure_redundant_echo,
        run_sim2_correlated_informative_residual,
        run_sim3_pure_interaction_xor,
        run_sim4_misspecified_baseline_score,
        run_sim5_sparse_trigger_m_sensitivity,
        run_sim6_mid_risk_informative_routing,
        run_sim7_unequal_costs_and_drift,
    ]
    per_seed: dict[str, list[dict[str, Any]]] = {}
    for fn in sim_fns:
        runs = [fn(seed=s) for s in seeds]
        key = runs[0]["scenario"]
        per_seed[key] = runs

    # Aggregate numeric scalar fields across seeds
    aggregated: dict[str, Any] = {}
    for key, runs in per_seed.items():
        agg_entry: dict[str, Any] = {"n_seeds": len(seeds)}
        sample = runs[0]
        for field_k, val in sample.items():
            if field_k == "scenario":
                continue
            if isinstance(val, (int, float, bool)):
                arr = np.asarray([float(r[field_k]) for r in runs], dtype=float)
                agg_entry[field_k] = {
                    "mean": float(np.mean(arr)),
                    "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                }
            elif field_k == "m_sensitivity":
                m_agg: dict[str, Any] = {}
                for m_key in val.keys():
                    m_agg[m_key] = {
                        sub_k: {
                            "mean": float(
                                np.mean([r["m_sensitivity"][m_key][sub_k] for r in runs])
                            ),
                            "sd": float(
                                np.std([r["m_sensitivity"][m_key][sub_k] for r in runs], ddof=1)
                            )
                            if len(runs) > 1
                            else 0.0,
                        }
                        for sub_k in val[m_key].keys()
                    }
                agg_entry["m_sensitivity"] = m_agg
        aggregated[key] = agg_entry

    return {"seeds": list(seeds), "summary": aggregated, "per_seed": per_seed}
