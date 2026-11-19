"""Direct regression tests for all 5 competitor-review counterexamples & statistical checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from imboreceipt.bench.adversarial_sims import (
    run_sim1_pure_redundant_echo,
    run_sim2_correlated_informative_residual,
    run_sim3_pure_interaction_xor,
    run_sim6_mid_risk_informative_routing,
)
from imboreceipt.bench.datasets import load_tier2_diabetes130
from imboreceipt.contract.decision import DecisionContract, wilson_halfwidth, wilson_interval
from imboreceipt.core.types import freeze_split
from imboreceipt.discover.strata import make_leak_free_cv_splits, sample_case_cohort_pilot
from imboreceipt.frontier.ceilings import (
    evaluate_certified_frontier,
    select_neyman_pearson_threshold_with_guarantee,
)
from imboreceipt.lattice.foveation import BinaryConformalFoveator, build_literal_conformal_01_mask


def test_review_2_1_literal_conformal_never_skips_top_when_no_positive_singletons() -> None:
    """Review Sec. 2.1: When {1} is empty, literal conformal {0,1} must NOT skip top 30% of K!"""
    n = 200
    p1 = np.linspace(0.01, 0.45, n)
    # Make all points in_0=True and top 40 in_1=True so only_1 is empty and top 40 are {0, 1}:
    in_0 = np.ones(n, dtype=bool)
    in_1 = np.zeros(n, dtype=bool)
    in_1[-40:] = True  # top 40 are {0, 1}; remaining 160 are {0}; ZERO {1} singletons!

    mask = build_literal_conformal_01_mask(p1, in_0, in_1, q_neg=0.50, k_queries=40)
    assert int(np.sum(mask)) == 40
    # Must query the exact 40 ambiguous {0, 1} records at the top — zero artificial 30% skip!
    assert np.all(mask[-40:])
    assert not np.any(mask[:-40])


def test_review_2_2_and_5_2_diabetes_admission_xbase_and_outcome_independent_eligibility() -> None:
    """Diabetes X_base has ONLY 7 admission fields; eligibility mask is independent of Y."""
    coh = load_tier2_diabetes130(max_patients=8000, seed=42, include_synthetic_proxies=False)
    expected_base = {
        "race",
        "gender",
        "age_bucket",
        "admission_type_id",
        "admission_source_id",
        "payer_code",
        "weight_flag",
    }
    assert set(coh.X_base.columns) == expected_base
    assert "time_in_hospital" not in coh.X_base.columns
    assert "num_lab_procedures" not in coh.X_base.columns
    # Moved to Stage-2 inpatient workup channel
    cand_ids = {c.channel_id for c in coh.candidates}
    assert "ch_inpatient_workup" in cand_ids
    assert "ch_redundant_intake_proxy" not in cand_ids
    # Upfront eligibility excluded 2,423 raw encounters including 43 recorded <30
    assert coh.n_raw_rows_before_eligibility == 101766
    assert coh.n_excluded_ineligible == 2423
    assert coh.n_excluded_ineligible_positives == 43
    assert coh.competing_censor_mask is None


def test_review_2_3_strict_chronological_cv_and_joint_group_time_split() -> None:
    """Chronological CV counterexample (100 rows) must NEVER have train_t > val_t."""
    times = np.arange(100, dtype=float)
    y = np.zeros(100, dtype=int)
    y[[5, 18, 32, 47, 63, 81, 95]] = 1

    splits = make_leak_free_cv_splits(y, n_folds=5, time_order=times, seed=42)
    assert len(splits) >= 3
    for tr, va in splits:
        latest_tr = float(np.max(times[tr]))
        earliest_va = float(np.min(times[va]))
        assert latest_tr < earliest_va, f"Violated forward chronology: {latest_tr} >= {earliest_va}"

    # Joint group + chronology outer split must satisfy BOTH group disjointness AND chronology
    df = pd.DataFrame(
        {
            "x": np.linspace(0, 1, 120),
            "grp": np.arange(120) // 2,
            "t": np.arange(120, dtype=float),
            "y": (np.arange(120) % 7 == 0).astype(int),
        }
    )
    sp = freeze_split(df, "y", group_col="grp", time_col="t", seed=42)
    parts = [
        np.array(sp.train_idx),
        np.array(sp.cal_idx),
        np.array(sp.val_idx),
        np.array(sp.test_idx),
    ]
    for i in range(len(parts) - 1):
        assert np.max(df["t"].to_numpy()[parts[i]]) < np.min(df["t"].to_numpy()[parts[i + 1]])
        for j in range(i + 1, len(parts)):
            g_i = set(df["grp"].to_numpy()[parts[i]].tolist())
            g_j = set(df["grp"].to_numpy()[parts[j]].tolist())
            assert len(g_i & g_j) == 0


def test_review_2_4_neyman_pearson_tied_scores_counterexample_and_binomial_bound() -> None:
    """100 negative calibration observations all equal to 0.20 at target FPR 0.05."""
    y_val = np.array([0] * 100 + [1] * 10, dtype=int)
    scores_val = np.array([0.20] * 100 + [0.85] * 10, dtype=float)
    y_test = np.array([0] * 100 + [1] * 10, dtype=int)
    scores_test = np.array([0.20] * 100 + [0.85] * 10, dtype=float)

    tau, certified, v_bound = select_neyman_pearson_threshold_with_guarantee(
        y_val, scores_val, alpha_fpr=0.05, delta_violation=0.05
    )
    assert tau == 0.20
    assert certified is True
    assert v_bound <= 0.05

    contract = DecisionContract(task_name="np_tie_check", target_precision=0.5, max_fpr_budget=0.05)
    metrics = evaluate_certified_frontier(
        "tie_arm", y_val, scores_val, y_test, scores_test, contract
    )
    # Previously yielded 100% FPR; now yields 0.0% FPR (<= 5%) while catching positives at 0.85!
    assert metrics.np_realized_fpr == 0.0
    assert metrics.np_tpr_at_fpr_budget == 1.0
    assert metrics.np_certified is True


def test_review_2_5_pilot_horvitz_thompson_weights_exact_sum_and_six_routing_policies() -> None:
    """Stratified HT control weights sum identically to N_0; Wilson = 0.294 at 4/5."""
    rng = np.random.default_rng(123)
    y = (rng.uniform(0, 1, 2000) < 0.04).astype(int)
    scores = rng.uniform(0.001, 0.60, 2000)
    n0 = int(np.sum(y == 0))

    pilot = sample_case_cohort_pilot(y, scores, matching_mode="score_stratified", seed=123)
    y_s = y[pilot.sampled_indices]
    assert int(np.sum(y_s == 1)) == int(np.sum(y == 1))
    assert np.all(pilot.inclusion_probabilities > 0.0)
    # Sum of Horvitz-Thompson weights over sampled controls equals N_0 identically!
    ht_neg_sum = float(np.sum(pilot.sampling_weights[y_s == 0]))
    assert abs(ht_neg_sum - float(n0)) < 1e-8

    # Review Sec. 4.2 check: Wilson half-width at k=4, n=5 is ~0.294 (not 0.351 normal planning)
    hw_wilson = wilson_halfwidth(4, 5, alpha=0.05)
    lo_w, hi_w = wilson_interval(4, 5, alpha=0.05)
    assert abs(hw_wilson - 0.2941) < 0.002
    assert 0.0 <= lo_w < 0.80 < hi_w <= 1.0

    # Verify all 6 equal-budget routing policies run at identical query rate K / N
    fov = BinaryConformalFoveator(alpha_neg=0.20, alpha_pos=0.20).fit(scores[:800], y[:800])
    rep = fov.evaluate_foveation(
        y[800:],
        scores[800:],
        np.clip(scores[800:] + 0.25 * y[800:], 0.0, 0.99),
        DecisionContract(task_name="six_policies", target_precision=0.50, max_fpr_budget=0.02),
    )
    assert set(rep.policies.keys()) == {
        "credo_foveation",
        "literal_conformal_01",
        "top_risk_k",
        "uncertainty_k",
        "voi_k",
        "random_k",
    }
    rates = {m.query_rate for m in rep.policies.values()}
    assert max(rates) - min(rates) < 1e-9


def test_adversarial_simulations_discriminate_expected_regimes() -> None:
    """Adversarial simulations reproduce both strengths (Sim 1-3) and failure modes (Sim 6)."""
    s1 = run_sim1_pure_redundant_echo(seed=42, n=5000)
    assert s1["pilot_cmi_precision"] == 1.0
    assert s1["global_mi_precision"] == 0.0

    s2 = run_sim2_correlated_informative_residual(seed=42, n=5000)
    assert s2["pilot_cmi_top1_correct"] == 1.0

    s3 = run_sim3_pure_interaction_xor(seed=42, n=6000)
    assert s3["full_xbase_pilot_cmi_top1_correct"] == 1.0
    assert s3["global_mi_top1_correct"] == 0.0

    s6 = run_sim6_mid_risk_informative_routing(seed=42, n=9000)
    # In Sim 6 (mid-risk informative, top-risk irreducible), VOI-K & Literal {0,1} beat Top-Risk-K!
    assert s6["voi_k_auprc"] > s6["top_risk_k_auprc"]
    assert s6["literal_conformal_01_auprc"] > s6["top_risk_k_auprc"]
