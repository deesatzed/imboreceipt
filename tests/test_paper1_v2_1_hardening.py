"""Unit tests verifying Paper 1 v2.1 methodological, leak-isolation, and comparator invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from imboreceipt.bench.fine_grained_cohorts import (
    FineGrainedCohort,
    materialize_leak_free_stress_pool,
)
from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.types import SemanticType
from imboreceipt.discover.selector import CandidateChannelSpec
from imboreceipt.discover.two_phase import (
    discover_competitor_method,
    discover_sequential_conditional_gain,
    sample_two_phase_pilot,
)
from imboreceipt.lattice.covert_afa import (
    CovertGreedyCMIAFA,
    validate_covert_afa_synthetic_oracle,
)
from imboreceipt.lattice.streaming_routing import (
    StreamingSelectiveRouter,
    compute_calibration_summary,
    compute_smooth_ici,
    solve_empirical_quantile_logit_offset,
)


def _make_toy_cohort(n: int = 800, seed: int = 7) -> FineGrainedCohort:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, n).astype(np.float32)
    x1 = rng.normal(0.0, 1.0, n).astype(np.float32)
    za = rng.normal(0.0, 1.0, n).astype(np.float32)
    zb = rng.normal(0.0, 1.0, n).astype(np.float32)
    y = rng.binomial(1, expit(-2.4 + 0.8 * x0 + 1.3 * za + 0.9 * zb))
    Xb = pd.DataFrame({"x0": x0, "x1": x1, "cat": np.where(x0 > 0, "A", "B")})
    Zp = pd.DataFrame({"za": za, "zb": zb})
    cands = [
        CandidateChannelSpec("ch_a", ("za",), 1.0, "mech_a"),
        CandidateChannelSpec("ch_b", ("zb",), 1.0, "mech_b"),
    ]
    contract = DecisionContract(
        task_name="Toy", target_precision=0.35, max_fpr_budget=0.05, cost_fp=1.0, cost_fn=9.0
    )
    return FineGrainedCohort(
        name="Toy",
        display_name="Toy",
        domain="Test",
        provenance="synthetic",
        landmark_description="Test",
        X_base=Xb,
        Z_pool=Zp,
        candidates=cands,
        y=y,
        contract=contract,
        acquisition_budget=2.0,
        per_instance_query_cost=0.03,
        base_semantic_types={
            "x0": SemanticType.CONTINUOUS,
            "x1": SemanticType.CONTINUOUS,
            "cat": SemanticType.NOMINAL,
        },
    )


def test_leak_free_stress_pool_ignores_test_corruptions() -> None:
    """Changing test-partition features by +10^7 must not alter train-partition stress cols."""
    coh1 = _make_toy_cohort(n=500, seed=11)
    coh2 = _make_toy_cohort(n=500, seed=11)
    train_idx = np.arange(0, 300)
    test_idx = np.arange(300, 500)

    # Corrupt test rows massively in coh2
    coh2.X_base.loc[test_idx, "x0"] = 1.0e7
    coh2.Z_pool.loc[test_idx, "za"] = -5.0e6

    Zp1, cands1 = materialize_leak_free_stress_pool(coh1, train_idx=train_idx, seed=99)
    Zp2, cands2 = materialize_leak_free_stress_pool(coh2, train_idx=train_idx, seed=99)

    assert [c.channel_id for c in cands1] == [c.channel_id for c in cands2]
    for col in Zp1.columns:
        if col.startswith(("s2_twin_", "base_echo_", "null_")):
            np.testing.assert_allclose(
                Zp1[col].iloc[train_idx].to_numpy(),
                Zp2[col].iloc[train_idx].to_numpy(),
                rtol=0.0,
                atol=1e-6,
            )


def test_breslow_cain_offset_and_no_lazy_screen_discovery() -> None:
    """Breslow-Cain pilot offset and disabling lazy screen execute cleanly and recover signal."""
    coh = _make_toy_cohort(n=1400, seed=21)
    oof_p = np.clip(expit(-2.4 + 0.8 * coh.X_base["x0"].to_numpy()), 1e-4, 1.0 - 1e-4)
    pilot = sample_two_phase_pilot(coh.y, oof_p, pilot_fraction=0.25, design="neyman", seed=21)

    res_bc = discover_sequential_conditional_gain(
        coh.X_base,
        coh.Z_pool,
        coh.y,
        coh.candidates,
        oof_p,
        pilot,
        max_budget=2.0,
        use_breslow_cain_offset=True,
        allow_lazy_submodular_screen=False,
        seed=21,
    )
    assert "ch_a" in res_bc.selected_ids


def test_chamma_residual_cpi_cancels_collinear_twins_unlike_sequential() -> None:
    """Under near-exact Stage-2 twins Z1 ~ Z2, Chamma residualized CPI cancels both mechanisms."""
    rng = np.random.default_rng(44)
    n = 1600
    x0 = rng.normal(0.0, 1.0, n)
    z1 = rng.normal(0.0, 1.0, n)
    z2 = z1 + rng.normal(0.0, 0.04, n)  # Collinear twin of z1
    z3 = rng.normal(0.0, 1.0, n)
    y = rng.binomial(1, expit(-2.5 + 0.6 * x0 + 1.45 * z1 + 0.85 * z3))

    Xb = pd.DataFrame({"x0": x0.astype(np.float32)})
    Zp = pd.DataFrame(
        {
            "z1": z1.astype(np.float32),
            "z2": z2.astype(np.float32),
            "z3": z3.astype(np.float32),
        }
    )
    cands = [
        CandidateChannelSpec("ch_1", ("z1",), 1.0, "a"),
        CandidateChannelSpec("ch_2_twin", ("z2",), 1.0, "twin"),
        CandidateChannelSpec("ch_3", ("z3",), 1.0, "b"),
    ]
    oof_p = np.clip(expit(-2.5 + 0.6 * x0), 1e-4, 1.0 - 1e-4)
    pilot = sample_two_phase_pilot(y, oof_p, pilot_fraction=0.35, design="neyman", seed=44)

    res_seq = discover_sequential_conditional_gain(
        Xb, Zp, y, cands, oof_p, pilot, max_budget=2.0, allow_lazy_submodular_screen=False, seed=44
    )
    res_chamma = discover_competitor_method(
        "chamma_residual_cpi_pilot", Xb, Zp, y, cands, oof_p, pilot, max_budget=2.0, seed=44
    )
    # Sequential picks ch_3 plus one of {ch_1, ch_2_twin}, never both twins
    assert "ch_3" in res_seq.selected_ids
    assert not ({"ch_1", "ch_2_twin"} <= set(res_seq.selected_ids))
    # Chamma residualized CPI conditions on Z_{-m}, suppressing ch_1 & ch_2_twin scores near zero
    assert res_chamma.initial_scores["ch_3"] > max(
        res_chamma.initial_scores["ch_1"], res_chamma.initial_scores["ch_2_twin"]
    )


def test_empirical_quantile_martingale_voi_nonnegative_and_coherent() -> None:
    """Empirical quantile martingale VOI satisfies E[P2 | P1] == P1 and VOI >= 0 identically."""
    rng = np.random.default_rng(101)
    p1 = rng.uniform(0.005, 0.45, size=400)
    # Heavily right-skewed Stage-2 log-odds updates
    skew_deltas = rng.exponential(scale=0.65, size=(400, 33)) - 0.65
    mu = solve_empirical_quantile_logit_offset(p1, skew_deltas)
    p2_grid = expit(logit(p1)[:, None] + mu[:, None] + skew_deltas)
    np.testing.assert_allclose(np.mean(p2_grid, axis=1), p1, atol=1e-12)

    contract = DecisionContract(
        task_name="Skew", target_precision=0.30, max_fpr_budget=0.05, cost_fp=1.0, cost_fn=12.0
    )
    y_cal = rng.binomial(1, p1)
    p2_cal = np.clip(expit(logit(p1) + rng.exponential(0.6, size=400) - 0.6), 1e-5, 1.0 - 1e-5)
    router = StreamingSelectiveRouter(
        contract,
        per_instance_query_cost=0.03,
        quadrature_mode="empirical_quantile",
        n_emp_quantiles=33,
    ).fit(y_cal, p1, p2_cal)
    voi_vals = router.estimate_voi(p1)
    assert float(np.min(voi_vals)) >= 0.0
    assert float(np.max(voi_vals)) > 0.0


def test_smooth_loess_ici_detects_miscalibration() -> None:
    """Smooth LOESS ICI is small under true probabilities and large under probability distortion."""
    rng = np.random.default_rng(55)
    p_true = rng.uniform(0.02, 0.40, size=1500)
    y = rng.binomial(1, p_true)
    ici_good = compute_smooth_ici(y, p_true)
    ici_bad = compute_smooth_ici(y, np.clip(p_true * 2.0, 0.001, 0.999))
    cal = compute_calibration_summary(y, p_true)
    assert ici_good < 0.035
    assert ici_bad > ici_good + 0.08
    assert np.isclose(cal.ici, ici_good)


def test_covert_afa_zero_mask_boundary_and_oracle_recovery() -> None:
    """CovertGreedyCMIAFA returns Stage-1 at 0% budget and recovers >92% synthetic oracle gain."""
    val = validate_covert_afa_synthetic_oracle(n_samples=4000, seed=2026)
    assert val["oracle_gain_recovery_frac"] > 0.92

    coh = _make_toy_cohort(n=600, seed=33)
    p1 = np.clip(expit(-2.4 + 0.8 * coh.X_base["x0"].to_numpy()), 1e-4, 1.0 - 1e-4)
    afa = CovertGreedyCMIAFA(coh.candidates, coh.contract, 0.03, 2.0, seed=33)
    afa.fit(
        coh.X_base.iloc[:360],
        coh.Z_pool.iloc[:360],
        coh.y[:360],
        p1[:360],
        coh.X_base.iloc[360:480],
        coh.Z_pool.iloc[360:480],
        coh.y[360:480],
        p1[360:480],
    )
    res0 = afa.evaluate(
        coh.X_base.iloc[480:], coh.Z_pool.iloc[480:], coh.y[480:], p1[480:], target_query_rate=0.0
    )
    assert res0.any_query_rate_test == 0.0
    np.testing.assert_allclose(res0.probs_test, p1[480:], atol=1e-7)


def test_storey_bh_conformal_pvalue_algebraic_identity() -> None:
    """Verify p_j^{JC} == pi0_hat * p_j^{null} identically for all ranks and tiedraws."""
    n_cal = 500
    m0 = 440
    n_gt = np.array([0.0, 12.0, 150.0, 439.0])
    n_eq = np.array([1.0, 3.0, 0.0, 1.0])
    u = np.array([0.2, 0.7, 0.5, 0.9])

    p_null = (1.0 + n_gt + u * n_eq) / float(m0 + 1)
    pi0_hat = float(m0 + 1) / float(n_cal + 1)
    p_jc = (1.0 + n_gt + u * n_eq) / float(n_cal + 1)
    np.testing.assert_allclose(pi0_hat * p_null, p_jc, rtol=0.0, atol=1e-15)
