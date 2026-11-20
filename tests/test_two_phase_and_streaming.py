"""Unit tests for Two-Phase Neyman Discovery, Streaming Selective Routing, CfBH, & TOST."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.bootstrap import (
    random_effects_meta_analysis,
    tost_equivalence_and_noninferiority,
)
from imboreceipt.discover.selector import CandidateChannelSpec
from imboreceipt.discover.two_phase import (
    discover_competitor_method,
    discover_sequential_conditional_gain,
    evaluate_split_pilot_step1_inference,
    sample_two_phase_pilot,
    two_phase_variance_decomposition,
)
from imboreceipt.lattice.streaming_routing import (
    StreamingSelectiveRouter,
    calibrated_randomized_threshold_masks,
    compute_calibration_summary,
    conformal_fdr_selection,
    decompose_routing_flips,
    solve_mean_preserving_logit_offset,
)


def test_neyman_pilot_exact_budget_and_ht_sum_even_under_high_prevalence() -> None:
    """Verify constrained Neyman water-filling achieves exact pilot fraction when pi > rho."""
    rng = np.random.default_rng(11)
    n = 4000
    p0 = np.clip(rng.beta(1.5, 4.5, size=n), 1e-4, 1.0 - 1e-4)
    y = rng.binomial(1, p0)
    # Prevalence ~ 25%, target pilot fraction = 10% (naive 100%-case sampling would exceed 25%)
    pilot = sample_two_phase_pilot(y, p0, pilot_fraction=0.10, design="neyman", seed=11)
    assert pilot.n_pilot == 400
    assert abs(pilot.pilot_fraction_of_valid - 0.10) < 1e-9
    # Sum of Horvitz-Thompson weights equals exact Phase-1 valid population N
    assert abs(float(np.sum(pilot.ht_weights)) - float(n)) < 1e-6


def test_two_phase_variance_exact_srs_cancellation_identity() -> None:
    """Under SRS without replacement, V_I + V_II equals exact superpopulation s^2 / n."""
    rng = np.random.default_rng(22)
    n_pop = 2000
    n_smp = 300
    psi = rng.normal(0.08, 0.35, size=n_smp)
    w = np.full(n_smp, float(n_pop) / float(n_smp))
    h = np.zeros(n_smp, dtype=int)
    g_hat, se_tot, se_1, se_2 = two_phase_variance_decomposition(psi, w, h, {0: n_pop}, {0: n_smp})
    assert abs(g_hat - float(np.mean(psi))) < 1e-10
    s2 = float(np.var(psi, ddof=1))
    expected_se = float(np.sqrt(s2 / float(n_smp)))
    assert abs(se_tot - expected_se) < 1e-6
    assert se_2 > se_1 > 0.0


def test_sequential_discovery_stops_on_pure_noise_and_eliminates_stage2_twins() -> None:
    """Verify sequential conditional gain rejects pure-noise nulls AND duplicate Stage-2 twins."""
    rng = np.random.default_rng(33)
    n = 3600
    x0 = rng.normal(0.0, 1.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    z_a = rng.normal(0.0, 1.0, n)
    z_a_twin = z_a + rng.normal(0.0, 0.08, n)
    z_b = rng.normal(0.0, 1.0, n)
    eta = -3.0 + 1.0 * x0 - 0.5 * x1 + 1.25 * z_a + 0.95 * z_b
    p0 = 1.0 / (1.0 + np.exp(-(-3.0 + 1.0 * x0 - 0.5 * x1)))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    X_b = pd.DataFrame({"x0": x0, "x1": x1})
    Z_df = pd.DataFrame(
        {
            "za": z_a,
            "za_twin": z_a_twin,
            "zb": z_b,
            "echo": 1.1 * x0 + rng.normal(0.0, 0.15, n),
            "null1": rng.normal(0.0, 1.0, n),
            "null2": rng.normal(0.0, 1.0, n),
            "null3": rng.normal(0.0, 1.0, n),
        }
    )
    cands = [
        CandidateChannelSpec("true_A", ("za",), 1.0, "sig"),
        CandidateChannelSpec("twin_A", ("za_twin",), 1.0, "twin"),
        CandidateChannelSpec("true_B", ("zb",), 1.0, "sig"),
        CandidateChannelSpec("echo_X", ("echo",), 1.0, "echo"),
        CandidateChannelSpec("null_1", ("null1",), 1.0, "null"),
        CandidateChannelSpec("null_2", ("null2",), 1.0, "null"),
        CandidateChannelSpec("null_3", ("null3",), 1.0, "null"),
    ]

    pilot = sample_two_phase_pilot(y, p0, pilot_fraction=0.25, design="neyman", seed=33)
    # Give budget = 4.0 even though only 2 distinct mechanisms exist -> must stop via Holm LCB!
    res_seq = discover_sequential_conditional_gain(
        X_b, Z_df, y, cands, p0, pilot, max_budget=4.0, sequential=True, seed=33
    )
    sel_seq = set(res_seq.selected_ids)
    # Recovers both distinct mechanisms without buying both twins or pure noise!
    assert ("true_A" in sel_seq) or ("twin_A" in sel_seq)
    assert not (("true_A" in sel_seq) and ("twin_A" in sel_seq))
    assert "true_B" in sel_seq
    assert not any(cid.startswith("null_") for cid in sel_seq)

    # Contrast with non-sequential one-shot CMI under budget=2.0: buys both twins and misses true_B!
    res_one = discover_sequential_conditional_gain(
        X_b, Z_df, y, cands, p0, pilot, max_budget=2.0, sequential=False, seed=33
    )
    assert ("true_A" in res_one.selected_ids) and ("twin_A" in res_one.selected_ids)

    # Check LOCO competitor runs cleanly and exhibits twin cancellation on true_A / twin_A
    res_loco = discover_competitor_method(
        "loco_pilot", X_b, Z_df, y, cands, p0, pilot, max_budget=2.0, seed=33
    )
    assert res_loco.initial_scores["true_B"] > res_loco.initial_scores["true_A"]


def test_streaming_router_and_jin_candes_conformal_fdr() -> None:
    """Verify streaming thresholds are calibrated on D_cal only and CfBH controls realized FDP."""
    rng = np.random.default_rng(44)
    n_cal, n_test = 3000, 4000
    x1_c = rng.normal(0.0, 1.0, n_cal)
    z_c = rng.normal(0.0, 1.0, n_cal)
    p2_c = 1.0 / (1.0 + np.exp(-(-2.6 + 1.1 * x1_c + 1.3 * z_c)))
    p1_c = 1.0 / (1.0 + np.exp(-(-2.6 + 1.1 * x1_c)))
    y_c = rng.binomial(1, p2_c)

    x1_t = rng.normal(0.0, 1.0, n_test)
    z_t = rng.normal(0.0, 1.0, n_test)
    p2_t = 1.0 / (1.0 + np.exp(-(-2.6 + 1.1 * x1_t + 1.3 * z_t)))
    p1_t = 1.0 / (1.0 + np.exp(-(-2.6 + 1.1 * x1_t)))
    y_t = rng.binomial(1, p2_t)

    # Verify mathematical FDR = E[FDP] control around target_fdr = 0.25 across 20 replicates
    fdps = []
    for rep in range(20):
        r_rep = np.random.default_rng(100 + rep)
        xc_r, zc_r = r_rep.normal(0.0, 1.0, n_cal), r_rep.normal(0.0, 1.0, n_cal)
        pc_r = 1.0 / (1.0 + np.exp(-(-2.6 + 1.1 * xc_r + 1.3 * zc_r)))
        yc_r = r_rep.binomial(1, pc_r)
        xt_r, zt_r = r_rep.normal(0.0, 1.0, n_test), r_rep.normal(0.0, 1.0, n_test)
        pt_r = 1.0 / (1.0 + np.exp(-(-2.6 + 1.1 * xt_r + 1.3 * zt_r)))
        yt_r = r_rep.binomial(1, pt_r)
        cf_r = conformal_fdr_selection(pc_r, yc_r, pt_r, yt_r, target_fdr=0.25, seed=rep)
        assert cf_r.n_selected > 10
        fdps.append(cf_r.realized_fdp)
    assert float(np.mean(fdps)) <= 0.28

    contract = DecisionContract(
        task_name="SyntheticStream",
        max_fpr_budget=0.05,
        target_precision=0.75,
        cost_fp=1.0,
        cost_fn=10.0,
    )
    router = StreamingSelectiveRouter(contract, per_instance_query_cost=0.08)
    router.fit(y_c, p1_c, p2_c)
    ev = router.evaluate_budget_frontier(y_t, p1_t, p2_t, query_rates=(0.0, 0.20, 1.0), seed=44)
    ap_0 = ev["curves"]["streaming_voi"][0]["auprc"]
    ap_20 = ev["curves"]["streaming_voi"][1]["auprc"]
    ap_100 = ev["curves"]["streaming_voi"][2]["auprc"]
    assert ap_100 > ap_20 > ap_0
    cal_s = compute_calibration_summary(y_t, p2_t)
    assert cal_s.ece < 0.05
    assert 0.80 <= cal_s.cal_slope <= 1.25


def test_schuirmann_tost_and_random_effects_meta_analysis() -> None:
    """Verify Schuirmann TOST non-inferiority/equivalence and DerSimonian-Laird pooling."""
    rng = np.random.default_rng(55)
    deltas = rng.normal(-0.0005, 0.0020, size=20)
    tost = tost_equivalence_and_noninferiority(deltas, margin=0.005, n_test=1500, n_train=5500)
    assert tost.noninferior_at_05
    assert tost.equivalent_at_05
    assert tost.ci90_low > -0.005 and tost.ci90_high < 0.005

    meta = random_effects_meta_analysis(
        [0.001, -0.002, 0.000, 0.0015], [0.003, 0.004, 0.003, 0.0035]
    )
    assert meta.k_studies == 4
    assert abs(meta.pooled_effect) < 0.005
    assert meta.ci95_low < meta.pooled_effect < meta.ci95_high


def test_competitor_methods_handle_nominal_and_missingness_channels() -> None:
    """Verify CPI, Knockoffs, LOCO, and Global MI handle one-hot nominal and missing columns."""
    rng = np.random.default_rng(66)
    n = 1200
    x0 = rng.normal(0.0, 1.0, n)
    cat_sig = rng.choice(["Normal", "High", "Critical"], size=n, p=[0.7, 0.2, 0.1])
    num_miss = rng.normal(0.0, 1.0, n)
    num_miss[rng.uniform(0.0, 1.0, n) < 0.25] = np.nan
    eta = -2.4 + 0.9 * x0 + 1.4 * (cat_sig == "Critical").astype(float)
    p0 = 1.0 / (1.0 + np.exp(-(-2.4 + 0.9 * x0)))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

    X_b = pd.DataFrame({"x0": x0})
    Z_df = pd.DataFrame({"cat_sig": cat_sig, "num_miss": num_miss})
    cands = [
        CandidateChannelSpec("ch_cat", ("cat_sig",), 1.0, "cat"),
        CandidateChannelSpec("ch_miss", ("num_miss",), 1.0, "miss"),
    ]
    pilot = sample_two_phase_pilot(y, p0, pilot_fraction=0.30, design="neyman", seed=66)
    for m_name in ("global_mi", "cpi_pilot", "loco_pilot", "crt_knockoff_pilot"):
        out = discover_competitor_method(
            m_name, X_b, Z_df, y, cands, p0, pilot, max_budget=1.5, seed=66
        )
        assert out.initial_scores["ch_cat"] > out.initial_scores["ch_miss"]


def test_proposition2_mean_centered_voi_coherence_and_nonnegativity() -> None:
    """Verify Proposition 2: mu*(p1, sigma) preserves E[p2|p1]=p1 to 1e-13 and VOI >= 0 pre-clip."""
    from imboreceipt.lattice.streaming_routing import _GH_NODES, _GH_WEIGHTS

    p1 = np.array([1e-4, 0.005, 0.010, 0.0196, 0.05, 0.20, 0.50, 0.85, 0.995])
    sig = np.array([0.1, 0.5, 1.0, 1.4, 0.8, 1.2, 2.0, 0.9, 1.5])
    mu = solve_mean_preserving_logit_offset(p1, sig)
    probs_grid = expit(logit(p1)[:, None] + mu[:, None] + sig[:, None] * _GH_NODES[None, :])
    mean_p2 = np.sum(_GH_WEIGHTS[None, :] * probs_grid, axis=1)
    assert float(np.max(np.abs(mean_p2 - p1))) < 1e-13

    contract = DecisionContract(
        task_name="ScaniaTest", cost_fp=1.0, cost_fn=50.0, target_precision=0.50
    )
    router = StreamingSelectiveRouter(contract, per_instance_query_cost=0.08, mean_centered=True)
    router.bin_edges_ = np.array([0.0, 1.0])
    router.bin_logit_sd_ = np.array([1.0])
    raw_centered = router._compute_raw_voi(p1, 1.0, 50.0, mean_centered=True)
    raw_uncentered = router._compute_raw_voi(p1, 1.0, 50.0, mean_centered=False)
    assert float(np.min(raw_centered)) >= -1e-12
    # At p1=0.010 under Scania 50:1 costs, uncentered VOI is negative (-0.036) while centered > 0
    assert raw_uncentered[2] < -0.03
    assert raw_centered[2] > 0.08


def test_randomized_tie_breaking_prevents_point_mass_query_explosion() -> None:
    """Verify randomized tie-breaking hits target q=20% even when 92% of scores are tied at 0.0."""
    rng = np.random.default_rng(77)
    n_cal, n_tst = 5000, 5000
    # 92% exact zeros, 8% positive scores -> non-strict >= quantile(0.80)=0 would query 100%!
    sc = np.zeros(n_cal)
    sc[:400] = rng.uniform(0.05, 0.50, 400)
    rng.shuffle(sc)
    st = np.zeros(n_tst)
    st[:400] = rng.uniform(0.05, 0.50, 400)
    rng.shuffle(st)

    u_c = rng.uniform(0.0, 1.0, n_cal)
    u_t = rng.uniform(0.0, 1.0, n_tst)
    m_cal, m_tst, gamma, rho = calibrated_randomized_threshold_masks(sc, st, 0.20, u_c, u_t)
    assert gamma == 0.0
    assert 0.12 < rho < 0.14
    assert abs(float(np.mean(m_cal)) - 0.20) < 0.015
    assert abs(float(np.mean(m_tst)) - 0.20) < 0.015


def test_flip_decomposition_exact_net_benefit_identity() -> None:
    """Verify NB_net(route) - NB_net(Stage 1) == beneficial - harmful - fee_drag identically."""
    rng = np.random.default_rng(88)
    n = 2000
    p1 = np.clip(rng.beta(1.2, 8.0, n), 1e-4, 0.999)
    p2 = np.clip(expit(logit(p1) + rng.normal(0.0, 0.9, n)), 1e-4, 0.999)
    y = rng.binomial(1, p2)
    q_mask = rng.uniform(0.0, 1.0, n) < 0.25
    contract = DecisionContract(
        task_name="Decomp", cost_fp=1.0, cost_fn=12.0, target_precision=0.35
    )
    c_q = 0.04
    decomp = decompose_routing_flips(y, p1, p2, q_mask, contract, c_q)

    tau = 1.0 / 13.0
    nb_s1 = (1000.0 / n) * (
        np.sum((y == 1) & (p1 >= tau)) - (1.0 / 12.0) * np.sum((y == 0) & (p1 >= tau))
    )
    pr = np.where(q_mask, p2, p1)
    nb_rt = (1000.0 / n) * (
        np.sum((y == 1) & (pr >= tau))
        - (1.0 / 12.0) * np.sum((y == 0) & (pr >= tau))
        - (c_q / 12.0) * np.sum(q_mask)
    )
    assert abs((nb_rt - nb_s1) - decomp.net_nb_delta_vs_stage1_per_1k) < 1e-11


def test_split_pilot_step1_inference_runs_and_controls_nulls() -> None:
    """Verify disjoint split-pilot Step-1 inference produces valid SEs and selects signal."""
    rng = np.random.default_rng(99)
    n = 4500
    x0 = rng.normal(0.0, 1.0, n)
    z_sig = rng.normal(0.0, 1.0, n)
    z_nul = rng.normal(0.0, 1.0, n)
    p0 = 1.0 / (1.0 + np.exp(-(-2.8 + 0.9 * x0)))
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(-2.8 + 0.9 * x0 + 1.2 * z_sig))))
    Xb = pd.DataFrame({"x0": x0})
    Zdf = pd.DataFrame({"z_sig": z_sig, "z_nul": z_nul})
    cands = [
        CandidateChannelSpec("sig", ("z_sig",), 1.0, "sig"),
        CandidateChannelSpec("nul", ("z_nul",), 1.0, "null"),
    ]
    pil = sample_two_phase_pilot(y, p0, pilot_fraction=0.25, design="neyman", seed=99)
    out = evaluate_split_pilot_step1_inference(Xb, Zdf, y, cands, p0, pil, seed=99)
    assert "sig" in out["selected_ids"]
    assert "nul" not in out["selected_ids"]
    assert out["ses_total"]["sig"] > 0.0

