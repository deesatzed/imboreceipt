"""Senior Professor Audit Regression Suite: Zero Leakage, Circularity, & Statistical Rigor."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from imboreceipt.bench.datasets import (
    load_tier2_credit_card_fraud,
    load_tier2_taiwan_credit_panel,
)
from imboreceipt.bench.gate05_sandbox import generate_gate05_cohort
from imboreceipt.bench.runner import run_multi_channel_cohort_benchmark
from imboreceipt.core.bootstrap import holm_bonferroni_adjust, nadeau_bengio_ttest
from imboreceipt.core.types import freeze_split
from imboreceipt.discover.selector import (
    CandidateChannelSpec,
    score_channels_baseline_strategy,
    score_channels_credo_reg,
)
from imboreceipt.discover.strata import (
    build_consensus_residual_strata,
    make_leak_free_cv_splits,
    sample_case_cohort_pilot,
)
from imboreceipt.frontier.baselines import PlattLogitCalibrator


def test_zero_presplit_compiler_leakage_under_test_poisoning() -> None:
    """Poisoning test-fold raw trajectories by +1e9 MUST NOT alter train/cal/val features."""
    cohort_a = generate_gate05_cohort(n_samples=1600, seed=77, selective_receipts=False)
    cohort_b = generate_gate05_cohort(n_samples=1600, seed=77, selective_receipts=False)

    df_meta = cohort_a.X_base.copy()
    df_meta["__y_split__"] = cohort_a.y_tilde
    split = freeze_split(df_meta, "__y_split__", seed=77)
    tr_i = np.array(split.train_idx, dtype=int)
    val_i = np.array(split.val_idx, dtype=int)
    tst_i = np.array(split.test_idx, dtype=int)

    assert cohort_b.raw_aux_df is not None and cohort_a.raw_aux_df is not None
    # Massively poison ONLY test-fold raw longitudinal values
    cohort_b.raw_aux_df.iloc[tst_i] = cohort_b.raw_aux_df.iloc[tst_i] * 1e6 + 1e9

    for fac_a, fac_b in zip(
        cohort_a.compiler_factories or [], cohort_b.compiler_factories or [], strict=True
    ):
        ca = fac_a().fit(cohort_a.raw_aux_df.iloc[tr_i])
        cb = fac_b().fit(cohort_b.raw_aux_df.iloc[tr_i])
        va = ca.transform(cohort_a.raw_aux_df.iloc[val_i]).to_numpy()
        vb = cb.transform(cohort_b.raw_aux_df.iloc[val_i]).to_numpy()
        np.testing.assert_allclose(va, vb, rtol=0.0, atol=0.0)


def test_group_and_time_cv_splits_prevent_entity_and_future_leakage() -> None:
    """Internal CV splits must strictly isolate patient groups and prevent future->past training."""
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.15, size=300)
    groups = np.repeat(np.arange(60), 5)

    grp_splits = make_leak_free_cv_splits(y, n_folds=5, groups=groups, seed=11)
    for tr, va in grp_splits:
        overlap = set(groups[tr].tolist()).intersection(set(groups[va].tolist()))
        assert len(overlap) == 0, f"Group leakage detected across CV folds: {overlap}"

    times = np.arange(300, dtype=float)
    t_splits = make_leak_free_cv_splits(y, n_folds=5, time_order=times, seed=11)
    assert len(t_splits) >= 4
    # EVERY returned chronological fold must satisfy max(train_time) < min(val_time)
    for tr, va in t_splits:
        assert np.max(times[tr]) < np.min(times[va]), "Future-to-past temporal CV leakage!"


def test_platt_calibrator_preserves_extreme_tail_auprc_without_tie_clipping() -> None:
    """Monotonic Platt calibration must not collapse extreme (<1e-6 or >1-1e-6) tail rankings."""
    rng = np.random.default_rng(19)
    n = 5000
    y = np.zeros(n, dtype=int)
    y[:15] = 1
    # Extreme fraud-scale probabilities spanning 1e-11 to 1 - 1e-9
    raw = rng.uniform(1e-11, 1e-7, size=n)
    raw[:15] = np.linspace(1.0 - 1e-7, 1.0 - 1e-11, 15)

    cal = PlattLogitCalibrator().fit(raw, y)
    calibrated = cal.predict_proba(raw)
    ap_raw = average_precision_score(y, raw)
    ap_cal = average_precision_score(y, calibrated)
    assert abs(ap_raw - ap_cal) < 1e-12


def test_credo_reg_rejects_redundant_baseline_echo_unlike_global_mi() -> None:
    """CREDO REG must reject collinear baseline-proxy decoys that fool marginal Global MI."""
    rng = np.random.default_rng(29)
    n = 2500
    x1 = rng.normal(0.0, 1.0, size=n)
    z_orth = rng.normal(0.0, 1.0, size=n)
    # Positive iff x1 high (baseline-visible) OR z_orth high (orthogonal mechanism)
    # Make x1 slightly stronger globally so marginal Global MI prefers the x1-clone over z_orth!
    y = (
        (1.5 * x1 + rng.normal(0, 0.5, n) > 2.4) | (1.25 * z_orth + rng.normal(0, 0.5, n) > 2.5)
    ).astype(int)

    X_base = pd.DataFrame({"x1": x1})
    Z_pool = pd.DataFrame(
        {
            "z_redundant_clone": x1 + rng.normal(0.0, 0.08, size=n),
            "z_true_orthogonal": z_orth,
        }
    )
    cands = [
        CandidateChannelSpec("ch_redundant_clone", ("z_redundant_clone",), acquisition_cost=1.0),
        CandidateChannelSpec("ch_true_orthogonal", ("z_true_orthogonal",), acquisition_cost=1.0),
    ]
    from imboreceipt.contract.decision import DecisionContract

    contract = DecisionContract(
        task_name="redundancy_test", target_precision=0.5, max_fpr_budget=0.02
    )
    strata = build_consensus_residual_strata(X_base, y, seed=29)
    pilot = sample_case_cohort_pilot(y, strata.oof_consensus_prob, controls_per_positive=4, seed=29)

    gmi = score_channels_baseline_strategy("global_mi", X_base, Z_pool, y, cands, strata, seed=29)
    ucc = score_channels_baseline_strategy(
        "unmatched_case_control_mi", X_base, Z_pool, y, cands, strata, seed=29
    )
    reg = score_channels_credo_reg(X_base, Z_pool, y, cands, strata, pilot, contract, seed=29)

    assert gmi[0].channel_id == "ch_redundant_clone"
    assert ucc[0].channel_id == "ch_redundant_clone"
    assert reg[0].channel_id == "ch_true_orthogonal"


def test_receipt_repairer_never_poisons_future_positives_without_trap_flags() -> None:
    """ReceiptRepairer must not flip >30d negatives to y=1 when traps_train is None."""
    from imboreceipt.receipt.repair import ReceiptRepairer

    X = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    censor = np.zeros(100, dtype=bool)
    censor[10:15] = True
    future_pos = np.zeros(100, dtype=bool)
    future_pos[15:40] = True

    rep = ReceiptRepairer()
    repaired = rep.fit_transform_train(
        X, y, competing_censor_mask=censor, future_positive_mask=future_pos
    )
    # 5 censored rows removed -> 95 rows remaining; zero pseudo-positives added!
    assert len(repaired.X_expanded) == 95
    assert int(np.sum(repaired.y_expanded == 1)) == 10


def test_theorem3_rare_event_conformal_singleton_paradox_foveation() -> None:
    """Verify CREDO Evidence-Deficit Upper-Tail Foveation beats classical mid-band {0,1}."""
    from imboreceipt.contract.decision import DecisionContract
    from imboreceipt.lattice.foveation import BinaryConformalFoveator

    rng = np.random.default_rng(101)
    n_cal, n_test = 2000, 3000
    y_cal = (rng.uniform(0, 1, n_cal) < 0.02).astype(int)
    y_test = (rng.uniform(0, 1, n_test) < 0.02).astype(int)

    # Stage 1 has moderate precision in the upper tail (~35% precision among top alerts)
    p1_cal = np.where(y_cal == 1, rng.uniform(0.15, 0.55, n_cal), rng.uniform(0.0, 0.05, n_cal))
    p1_test = np.where(y_test == 1, rng.uniform(0.15, 0.55, n_test), rng.uniform(0.0, 0.05, n_test))
    # Add top-tail Stage-1 false positives (which land in conformal singleton {1}!)
    fp_idx = rng.choice(np.flatnonzero(y_test == 0), size=75, replace=False)
    p1_test[fp_idx] = rng.uniform(0.25, 0.58, size=75)

    # Stage 2 cleanly separates true positives from Stage-1 false positives
    p2_test = np.where(y_test == 1, rng.uniform(0.70, 0.98, n_test), rng.uniform(0.0, 0.04, n_test))

    fov = BinaryConformalFoveator(alpha_pos=0.08, alpha_neg=0.05).fit(p1_cal, y_cal)
    contract = DecisionContract(task_name="thm3_test", target_precision=0.80, max_fpr_budget=0.02)
    rep = fov.evaluate_foveation(y_test, p1_test, p2_test, contract)

    assert rep.credo_deficit_foveated_auprc > rep.classical_midband_conformal_auprc + 0.10
    assert (
        rep.net_utility_foveated_net_of_query_cost_per_1k
        >= rep.net_utility_classical_midband_per_1k
    )


def test_taiwan_non_circular_and_ulb_disjoint_test_windows() -> None:
    """Verify Taiwan uses unmodified non-circular target and ULB windows have disjoint test sets."""
    tw = load_tier2_taiwan_credit_panel(seed=42)
    # Cohort filtered strictly on X_base['initial_status_m6'] <= 0
    assert np.all(tw.X_base["initial_status_m6"].to_numpy(dtype=float) <= 0.0)
    assert tw.raw_aux_df is not None and len(tw.compiler_factories or []) == 3

    # Verify ULB 5 chronological windows produce 100% mutually disjoint test timestamps
    test_spans: list[tuple[float, float]] = []
    for w in range(5):
        ulb = load_tier2_credit_card_fraud(max_negatives=None, window_idx=w, n_windows=5, seed=42)
        assert ulb.disjoint_test_blocks is True
        assert ulb.time_order is not None
        df_m = ulb.X_base.copy()
        df_m["__y__"] = ulb.y_tilde
        df_m["__t__"] = ulb.time_order
        sp = freeze_split(df_m, "__y__", time_col="__t__", seed=42)
        tst = np.array(sp.test_idx, dtype=int)
        test_spans.append((float(np.min(ulb.time_order[tst])), float(np.max(ulb.time_order[tst]))))

    for idx in range(len(test_spans) - 1):
        assert test_spans[idx][1] <= test_spans[idx + 1][0], (
            f"Overlapping ULB test windows {idx} and {idx + 1}: {test_spans}"
        )


def test_nadeau_bengio_and_equal_capacity_ablation_arm() -> None:
    """Verify Nadeau-Bengio inflates SE relative to naive SD/sqrt(K) and A5b runs cleanly."""
    deltas = [0.05, 0.06, 0.04, 0.055, 0.045]
    nb_overlap = nadeau_bengio_ttest(deltas, n_test=1500, n_train=5500, disjoint_test_blocks=False)
    nb_disjoint = nadeau_bengio_ttest(deltas, n_test=1500, n_train=5500, disjoint_test_blocks=True)
    assert nb_overlap.corrected_se > nb_disjoint.corrected_se
    assert nb_overlap.p_value_one_sided > nb_disjoint.p_value_one_sided

    adj = holm_bonferroni_adjust([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and adj[2] == 0.06 and adj[1] == 0.06

    g05 = generate_gate05_cohort(n_samples=1500, seed=42, selective_receipts=True)
    res = run_multi_channel_cohort_benchmark(g05, seed=42, n_boot=60)
    assert "monolithic_credo_ensemble" in res["arms"]


def test_wave2_home_credit_compiler_zero_presplit_leakage() -> None:
    """Verify Home Credit 7-table compilers never leak across train/val/test splits."""
    from pathlib import Path

    if not Path("data/cache/tier2_home_credit_7table.pkl").exists():
        return
    from imboreceipt.bench.wave2_datasets import load_wave2_home_credit_relational

    hc_a = load_wave2_home_credit_relational(max_rows=3000, seed=42)
    hc_b = load_wave2_home_credit_relational(max_rows=3000, seed=42)
    assert hc_a.raw_aux_df is not None and hc_b.raw_aux_df is not None
    tr = np.arange(0, 1600)
    va = np.arange(1600, 2200)
    ts = np.arange(2200, 3000)
    hc_b.raw_aux_df.iloc[ts] = hc_b.raw_aux_df.iloc[ts] * 1e6 + 1e8

    za_va = hc_a.materialize_z_splits(tr, va, ts)[1].to_numpy(dtype=float)
    zb_va = hc_b.materialize_z_splits(tr, va, ts)[1].to_numpy(dtype=float)
    np.testing.assert_allclose(np.nan_to_num(za_va), np.nan_to_num(zb_va), rtol=0.0, atol=0.0)
