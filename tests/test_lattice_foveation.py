"""Tests for Stage O: LIGO time-shift coincidence, OR-of-ANDs Lattice, and Conformal Foveation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imboreceipt.bench.gate05_sandbox import generate_gate05_cohort
from imboreceipt.bench.runner import run_multi_channel_cohort_benchmark
from imboreceipt.contract import DecisionContract
from imboreceipt.lattice import (
    BinaryConformalFoveator,
    ConfirmRescueLattice,
    estimate_tail_coincidence_surrogate,
)


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


def test_kappa0_hybrid_lattice_adapts_to_coupled_interaction_regime() -> None:
    """When kappa_0 is high across coupled channels, hybrid lattice beats pure orthogonal."""
    rng = np.random.default_rng(123)
    n = 5000
    # Shared false-positive spike u_fp elevates x_base, z_1, and z_2 in the same upper tail!
    u_fp = rng.binomial(1, 0.08, size=n) * 2.2
    m1 = rng.normal(0.0, 1.0, size=n)
    m2 = rng.normal(0.0, 1.0, size=n)
    m3 = rng.normal(0.0, 1.0, size=n)

    x_base_sig = m1 + u_fp
    z1_sig = m2 + u_fp
    z2_sig = m3 + 0.8 * u_fp

    # True event requires simultaneous 3-way coupled synergy across X_base, Z_1, and Z_2
    # while u_fp==0 (so joint [X_base, Z_1, Z_2] separates the 3-way synergy from the u_fp spike!)
    y = ((m1 > 0.75) & (m2 > 0.75) & (m3 > 0.95) & (u_fp == 0)).astype(int)

    X_base = pd.DataFrame({"x_sig": x_base_sig, "noise": rng.normal(0, 1, n)})
    Z_1 = pd.DataFrame({"z1_sig": z1_sig})
    Z_2 = pd.DataFrame({"z2_sig": z2_sig})

    s1, s2, s3 = 2700, 3500, 4250
    Xb_tr, Xb_cal, Xb_val, Xb_tst = (
        X_base.iloc[:s1],
        X_base.iloc[s1:s2],
        X_base.iloc[s2:s3],
        X_base.iloc[s3:],
    )
    Zc_tr = {"ch_1": Z_1.iloc[:s1], "ch_2": Z_2.iloc[:s1]}
    Zc_cal = {"ch_1": Z_1.iloc[s1:s2], "ch_2": Z_2.iloc[s1:s2]}
    Zc_val = {"ch_1": Z_1.iloc[s2:s3], "ch_2": Z_2.iloc[s2:s3]}
    Zc_tst = {"ch_1": Z_1.iloc[s3:], "ch_2": Z_2.iloc[s3:]}
    y_tr, y_cal, y_val, y_tst = y[:s1], y[s1:s2], y[s2:s3], y[s3:]

    orth = ConfirmRescueLattice(max_fpr_budget=0.02, mode="pure_orthogonal", seed=123).fit(
        Xb_tr, Zc_tr, y_tr, Xb_cal, Zc_cal, y_cal, Xb_val, Zc_val, y_val
    )
    hyb = ConfirmRescueLattice(max_fpr_budget=0.02, mode="hybrid_kappa0", seed=123).fit(
        Xb_tr, Zc_tr, y_tr, Xb_cal, Zc_cal, y_cal, Xb_val, Zc_val, y_val
    )

    from sklearn.metrics import average_precision_score

    ap_orth = average_precision_score(y_tst, orth.predict_proba(Xb_tst, Zc_tst))
    ap_hyb = average_precision_score(y_tst, hyb.predict_proba(Xb_tst, Zc_tst))

    assert hyb.mean_kappa0_ > 1.5
    assert hyb.orthogonal_lattice_weight_ < 0.85
    assert ap_hyb > ap_orth + 0.03
