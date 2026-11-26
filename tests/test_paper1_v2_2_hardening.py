"""Unit tests verifying Paper 1 v2.2 methodological, WCS, DB-CMI, and twin-geometry invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.types import SemanticType
from imboreceipt.discover.selector import CandidateChannelSpec
from imboreceipt.discover.two_phase import (
    discover_competitor_method,
    discover_sequential_conditional_gain,
    discover_sequential_split_pilot,
    sample_two_phase_pilot,
)
from imboreceipt.lattice.covert_afa import CovertGreedyCMIAFA
from imboreceipt.lattice.streaming_routing import (
    conformal_fdr_selection,
    weighted_conformal_fdr_selection,
)


def test_wcs_controls_fdr_under_pure_covariate_shift_and_matches_uniform() -> None:
    """Weighted Conformalized Selection (Jin & Candes, 2025) controls FDR under covariate shift."""
    rng = np.random.default_rng(2026)
    n_cal, n_te = 2000, 1500
    # Calibration X ~ N(0, 1); Test X ~ N(+1.1, 1) (pure covariate shift, same P(Y=1 | X)!)
    x_cal = rng.normal(0.0, 1.0, size=n_cal)
    x_te = rng.normal(1.1, 1.0, size=n_te)
    z_cal = rng.normal(0.0, 1.0, size=n_cal)
    z_te = rng.normal(0.0, 1.0, size=n_te)

    p_cal = expit(-2.4 + 1.2 * x_cal + 1.4 * z_cal)
    p_te = expit(-2.4 + 1.2 * x_te + 1.4 * z_te)
    y_cal = rng.binomial(1, p_cal)
    y_te = rng.binomial(1, p_te)

    # Exact covariate likelihood ratio dP_te(x) / dP_cal(x) = exp(1.1 * x - 0.5 * 1.1^2)
    w_cal = np.exp(1.1 * x_cal - 0.5 * 1.21)
    w_te = np.exp(1.1 * x_te - 0.5 * 1.21)

    res_wcs = weighted_conformal_fdr_selection(
        p_cal, y_cal, w_cal, p_te, y_te, w_te, target_fdr=0.35, seed=7
    )
    assert res_wcs.n_selected > 20
    assert res_wcs.realized_fdp <= 0.45

    # Under unit weights on continuous scores, WCS and CfBH match closely
    res_u_wcs = weighted_conformal_fdr_selection(
        p_cal, y_cal, np.ones_like(p_cal), p_te, y_te, np.ones_like(p_te), target_fdr=0.35, seed=7
    )
    res_cfbh = conformal_fdr_selection(p_cal, y_cal, p_te, y_te, target_fdr=0.35, seed=7)
    assert abs(res_u_wcs.selection_rate - res_cfbh.selection_rate) < 0.01


def test_decision_boundary_cmi_weighting_suppresses_far_subthreshold_queries() -> None:
    """DB-CMI concentrates fixed-budget queries closer to tau* than unweighted Covert-AFA."""
    rng = np.random.default_rng(42)
    n = 1200
    xb = rng.normal(0.0, 1.5, size=n)
    z1 = rng.normal(0.0, 1.0, size=n)
    z2 = rng.normal(0.0, 1.0, size=n)
    p_true = expit(-2.2 + 1.3 * xb + 1.2 * z1)
    y = rng.binomial(1, p_true)
    p1 = expit(-2.2 + 1.15 * xb)

    Xb_df = pd.DataFrame({"xb": xb.astype(np.float32)})
    Zp_df = pd.DataFrame({"z1": z1.astype(np.float32), "z2": z2.astype(np.float32)})
    cands = [
        CandidateChannelSpec("ch1", ("z1",), 1.0, "sig"),
        CandidateChannelSpec("ch2", ("z2",), 1.0, "null"),
    ]
    contract = DecisionContract(
        task_name="t", target_precision=0.4, max_fpr_budget=0.05, cost_fp=1.0, cost_fn=9.0
    )
    afa = CovertGreedyCMIAFA(cands, contract, 0.04, 2.0, max_steps_per_instance=2, n_jobs=1, seed=9)
    afa.fit(
        Xb_df.iloc[:600],
        Zp_df.iloc[:600],
        y[:600],
        p1[:600],
        Xb_df.iloc[600:900],
        Zp_df.iloc[600:900],
        y[600:900],
        p1[600:900],
    )
    r_std = afa.evaluate(
        Xb_df.iloc[900:], Zp_df.iloc[900:], y[900:], p1[900:], target_query_rate=0.20
    )
    r_db = afa.evaluate(
        Xb_df.iloc[900:],
        Zp_df.iloc[900:],
        y[900:],
        p1[900:],
        target_query_rate=0.20,
        decision_boundary_weighted=True,
        fee_accounting="proportional_cost",
        ref_bundle_cost=2.0,
    )
    tau = 0.10
    dist_std = float(np.mean(np.abs(p1[900:][r_std.queried_any_test] - tau)))
    dist_db = float(np.mean(np.abs(p1[900:][r_db.queried_any_test] - tau)))
    assert dist_db <= dist_std + 1e-6


def test_symmetric_vs_asymmetric_twin_baseline_stratified_permutation() -> None:
    """Under multi-column twin channels sharing one latent mechanism vs greedy 1-col splits:
    full-pool baseline-stratified permutation (cpi_pilot) avoids 1-col asymmetric twins via greedy
    root concentration, whereas on multi-column twin channels where trees split across both twins,
    off-manifold permutation duplicates both twins while sequential forward gain picks 1 twin + B.
    """
    rng = np.random.default_rng(108)
    n = 6000
    xb = rng.normal(0.0, 1.0, size=n)
    u1 = rng.normal(0.0, 1.0, size=n)
    u2 = rng.normal(0.0, 1.0, size=n)
    z1a = u1 + rng.normal(0.0, 0.15, size=n)
    z1b = u2 + rng.normal(0.0, 0.35, size=n)
    z2a = u1 + rng.normal(0.0, 0.35, size=n)
    z2b = u2 + rng.normal(0.0, 0.15, size=n)
    z3 = rng.normal(0.0, 1.0, size=n)
    y = rng.binomial(1, expit(-2.1 + 0.9 * xb + 2.3 * (u1 + u2) + 1.05 * z3))
    oof_b = expit(-2.1 + 0.9 * xb)

    Xb_df = pd.DataFrame({"xb": xb.astype(np.float32)})
    Z_df = pd.DataFrame(
        {
            "z1a": z1a.astype(np.float32),
            "z1b": z1b.astype(np.float32),
            "z2a": z2a.astype(np.float32),
            "z2b": z2b.astype(np.float32),
            "z3": z3.astype(np.float32),
            "zn": rng.normal(0.0, 1.0, size=n).astype(np.float32),
        }
    )
    cands = [
        CandidateChannelSpec("twin_a1", ("z1a", "z1b"), 1.0, "mech_a"),
        CandidateChannelSpec("twin_a2", ("z2a", "z2b"), 1.0, "mech_a"),
        CandidateChannelSpec("mech_b", ("z3",), 1.0, "mech_b"),
        CandidateChannelSpec("pure_null", ("zn",), 1.0, "pure_noise_null"),
    ]
    sem = {"xb": SemanticType.CONTINUOUS}
    pil = sample_two_phase_pilot(
        y, oof_b, np.ones(n, dtype=bool), pilot_fraction=0.25, design="neyman", seed=108
    )

    r_cpi = discover_competitor_method(
        "cpi_pilot", Xb_df, Z_df, y, cands, oof_b, pil, 2.0, base_semantic_types=sem, seed=108
    )
    r_seq = discover_sequential_conditional_gain(
        Xb_df, Z_df, y, cands, oof_b, pil, 2.0, base_semantic_types=sem, seed=108
    )
    # Multi-column twin channels split predictive weight across both twins in full-pool GBDT
    assert set(r_cpi.selected_ids) == {"twin_a1", "twin_a2"}
    # Sequential forward CMI selects one A-twin + mech_b!
    assert "mech_b" in r_seq.selected_ids
    assert len(set(r_seq.selected_ids) & {"twin_a1", "twin_a2"}) == 1


def test_dimensional_consistency_of_net_benefit_query_cost_across_thresholds() -> None:
    """Verify Delta NB = Delta TP + odds(tau)*Spared_FP - c_query*odds(tau)*q*1000 identically."""
    rng = np.random.default_rng(99)
    n = 2000
    y = rng.binomial(1, 0.12, size=n)
    p1 = rng.uniform(0.01, 0.30, size=n)
    p2 = np.clip(p1 + rng.normal(0.0, 0.05, size=n), 0.001, 0.999)
    q = rng.binomial(1, 0.40, size=n).astype(bool)
    p_r = np.where(q, p2, p1)
    c_q = 0.04

    for tau in (0.06, 1.0 / 9.0, 0.15, 0.20):
        odds_tau = tau / (1.0 - tau)
        cfn = (1.0 - tau) / tau  # with C_FP = 1 numeraire, 1 / C_FN == odds(tau)
        nb_r = (1000.0 / n) * np.sum(
            y * (p_r >= tau) - (1.0 / cfn) * (1 - y) * (p_r >= tau) - (c_q / cfn) * q
        )
        nb_0 = (1000.0 / n) * np.sum(y * (p1 >= tau) - (1.0 / cfn) * (1 - y) * (p1 >= tau))

        dtp = (1000.0 / n) * (np.sum((y == 1) & (p_r >= tau)) - np.sum((y == 1) & (p1 >= tau)))
        spared_fp = (1000.0 / n) * (
            np.sum((y == 0) & (p1 >= tau)) - np.sum((y == 0) & (p_r >= tau))
        )
        fee = c_q * odds_tau * 1000.0 * float(np.mean(q))
        assert abs((nb_r - nb_0) - (dtp + odds_tau * spared_fp - fee)) < 1e-12


def test_discover_sequential_split_pilot_recovers_orthogonal_signals_without_twins() -> None:
    """Multi-step disjoint 50/50 split-pilot (Theorem 1) recovers A+B without twins or nulls."""
    rng = np.random.default_rng(77)
    n = 6000
    xb = rng.normal(0.0, 1.0, size=n)
    za = rng.normal(0.0, 1.0, size=n)
    za_twin = za + rng.normal(0.0, 0.12, size=n)
    zb = rng.normal(0.0, 1.0, size=n)
    y = rng.binomial(1, expit(-2.2 + 0.9 * xb + 1.45 * za + 1.15 * zb))
    oof_b = expit(-2.2 + 0.9 * xb)

    Xb_df = pd.DataFrame({"xb": xb.astype(np.float32)})
    Z_df = pd.DataFrame(
        {
            "za": za.astype(np.float32),
            "za_twin": za_twin.astype(np.float32),
            "zb": zb.astype(np.float32),
            "zn": rng.normal(0.0, 1.0, size=n).astype(np.float32),
        }
    )
    cands = [
        CandidateChannelSpec("ch_a", ("za",), 1.0, "a"),
        CandidateChannelSpec("ch_a_twin", ("za_twin",), 1.0, "a"),
        CandidateChannelSpec("ch_b", ("zb",), 1.0, "b"),
        CandidateChannelSpec("ch_null", ("zn",), 1.0, "null"),
    ]
    pil = sample_two_phase_pilot(
        y, oof_b, np.ones(n, dtype=bool), pilot_fraction=0.25, design="neyman", seed=77
    )
    res_sp = discover_sequential_split_pilot(
        Xb_df, Z_df, y, cands, oof_b, pil, max_budget=2.0, seed=77
    )
    assert "ch_b" in res_sp.selected_ids
    assert len(set(res_sp.selected_ids) & {"ch_a", "ch_a_twin"}) == 1
    assert "ch_null" not in res_sp.selected_ids
