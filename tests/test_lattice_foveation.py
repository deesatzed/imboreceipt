"""Tests for Stage O: LIGO time-shift coincidence, OR-of-ANDs Lattice, and Conformal Foveation."""

from __future__ import annotations

import numpy as np
import pytest

from imboreceipt.bench.gate05_sandbox import generate_gate05_cohort
from imboreceipt.bench.runner import run_multi_channel_cohort_benchmark
from imboreceipt.contract import DecisionContract
from imboreceipt.lattice import BinaryConformalFoveator, estimate_tail_coincidence_surrogate


def test_ligo_time_shift_surrogate_separates_independent_and_confounded_tails() -> None:
    """Verify Prediction 4: time-shift surrogate certifies 10^-4 FPR vs confounded blowup."""
    rng = np.random.default_rng(31)
    m = 5000
    a_neg = rng.normal(0, 1, m)
    b_indep = rng.normal(0, 1, m)
    # Shared rare spike confounder on 3% of negatives
    spike = rng.binomial(1, 0.03, m) * 3.5
    b_conf = 0.3 * rng.normal(0, 1, m) + spike
    a_conf = a_neg + spike

    rep_ind = estimate_tail_coincidence_surrogate(
        a_neg, b_indep, alpha_a=0.01, alpha_b=0.01, seed=31
    )
    rep_conf = estimate_tail_coincidence_surrogate(
        a_conf, b_conf, alpha_a=0.01, alpha_b=0.01, seed=31
    )

    assert rep_ind.shifted_background_fpr == pytest.approx(0.0001, abs=0.00008)
    assert rep_ind.projected_joint_fpr < 0.0004
    assert 0.65 <= rep_ind.negative_tail_coincidence_kappa0 <= 1.45

    assert rep_conf.negative_tail_coincidence_kappa0 > 3.5
    assert rep_conf.projected_joint_fpr > rep_ind.projected_joint_fpr * 10.0


def test_conformal_foveation_coverage_and_sensor_savings() -> None:
    """Verify finite-sample class-wise conformal coverage and Stage-2 active sensing savings."""
    rng = np.random.default_rng(55)
    n_cal = 2000
    n_test = 3000
    y_cal = rng.binomial(1, 0.08, n_cal)
    y_test = rng.binomial(1, 0.08, n_test)

    # Noisy Stage-1 channel vs sharper Stage-2 confirmation channel
    from scipy.special import expit

    s1_cal = expit(-2.4 + 1.3 * y_cal + rng.normal(0, 1.0, n_cal))
    s1_test = expit(-2.4 + 1.3 * y_test + rng.normal(0, 1.0, n_test))
    s2_test = expit(-2.8 + 3.1 * y_test + rng.normal(0, 0.8, n_test))

    fov = BinaryConformalFoveator(alpha_neg=0.20, alpha_pos=0.20, stage2_query_cost_per_case=0.20)
    fov.fit(s1_cal, y_cal)

    contract = DecisionContract(
        task_name="fov_test",
        target_precision=0.45,
        max_fpr_budget=0.02,
        cost_fp=1.0,
        cost_fn=20.0,
        graded_action_cost_fp=0.20,
    )
    rep = fov.evaluate_foveation(y_test, s1_test, s2_test, contract)

    # Finite-sample Mondrian coverage guarantee (>= 1 - 0.20 - Monte Carlo SE)
    assert rep.stage1_class0_coverage >= 0.76
    assert rep.stage1_class1_coverage >= 0.74
    # Saves substantial Stage-2 queries compared to 100% Stage-2 deployment!
    assert rep.stage2_sensor_savings_fraction > 0.55
    assert rep.net_utility_foveated_net_of_query_cost_per_1k > rep.net_utility_stage1_only_per_1k
    assert (
        rep.net_utility_foveated_net_of_query_cost_per_1k
        > rep.net_utility_full_stage2_everywhere_per_1k
    )


def test_full_credo_lattice_end_to_end_on_gate05() -> None:
    """Verify end-to-end Gate 0.5 CREDO pipeline beats baseline GBDT and SMOTE at certified FPR."""
    cohort = generate_gate05_cohort(n_samples=6000, seed=42, selective_receipts=True)
    res = run_multi_channel_cohort_benchmark(cohort, seed=42)

    base_auprc = res["arms"]["calibrated_gbdt"]["auprc"]
    smote_auprc = res["arms"]["smote_gbdt"]["auprc"]
    credo_auprc = res["arms"]["full_credo_lattice"]["auprc"]

    assert credo_auprc > base_auprc + 0.08
    assert credo_auprc > smote_auprc + 0.08
    assert res["paired_bootstrap_credo_vs_gbdt"]["significant"]
    # Exogenous shock subtype should remain near-zero NP recall while mechanistic subtypes jump!
    sub = res["subtype_np_recall"]
    if "exogenous_shock" in sub and "orthogonal_coincidence" in sub:
        assert (
            sub["orthogonal_coincidence"]["credo_np_recall"]
            > sub["exogenous_shock"]["credo_np_recall"] + 0.15
        )
