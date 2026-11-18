"""Tests for Stage D: Case-cohort REG discovery vs global baselines & CSD boundary (P2, P3, P8)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from imboreceipt.bench.gate05_sandbox import generate_gate05_cohort
from imboreceipt.discover import (
    CriticalSlowingDownCompiler,
    adaptive_controls_per_positive,
    build_consensus_residual_strata,
    sample_case_cohort_pilot,
    score_channels_baseline_strategy,
    score_channels_credo_reg,
)


def test_case_cohort_reg_prioritizes_orthogonal_mechanisms_over_confounded_traps() -> None:
    """Verify Predictions 2 & 3 on Gate 0.5: REG rejects confounded FP traps at ~18% cohort cost."""
    cohort = generate_gate05_cohort(n_samples=6000, seed=42, selective_receipts=False)
    tr_idx = np.arange(4000, dtype=int)
    Xb = cohort.X_base.iloc[tr_idx].reset_index(drop=True)
    Zp = cohort.materialize_z_splits(tr_idx)[0]
    y = cohort.y_true[tr_idx]

    strata = build_consensus_residual_strata(
        Xb, y, semantic_types=cohort.base_semantic_types, seed=42
    )
    pilot = sample_case_cohort_pilot(y, strata.oof_consensus_prob, controls_per_positive=4, seed=42)

    # Case-cohort measures ~5 * prevalence (~26%) instead of 100% of cohort!
    assert pilot.fraction_of_cohort_measured < 0.28

    credo_ranked = score_channels_credo_reg(
        Xb, Zp, y, cohort.candidates, strata, pilot, cohort.contract, seed=42
    )
    top3_credo = {r.channel_id for r in credo_ranked[:3]}
    # Top 3 selected by CREDO REG should be the 3 true causal mechanisms, NOT noise decoys!
    assert len(top3_credo & cohort.ground_truth_relevant_channels) >= 2
    assert "ch_decoy_1" not in top3_credo
    assert "ch_decoy_2" not in top3_credo

    # Verify kappa_0 distinguishes orthogonal biomarker from confounded FP trap
    by_id = {r.channel_id: r for r in credo_ranked}
    assert (
        by_id["ch_orth_biomarker"].negative_tail_coincidence
        < by_id["ch_confounded_fp_trap"].negative_tail_coincidence
    )

    # CREDO REG ranks orthogonal biomarker strictly above the confounded FP trap!
    credo_order = [r.channel_id for r in credo_ranked]
    assert credo_order.index("ch_orth_biomarker") < credo_order.index("ch_confounded_fp_trap")

    # Baseline MI also ranks signal above pure noise decoys
    mi_ranked = score_channels_baseline_strategy(
        "global_mi", Xb, Zp, y, cohort.candidates, strata, seed=42
    )
    assert mi_ranked[0].channel_id not in {"ch_decoy_1", "ch_decoy_2"}


def test_prediction8_critical_slowing_down_lifts_transitions_not_memoryless_shocks() -> None:
    """Verify Prediction 8: CSD features lift regime transitions, zero lift on shocks."""
    rng = np.random.default_rng(99)
    n = 3000
    y_transition = rng.binomial(1, 0.06, n)
    y_shock = rng.binomial(1, 0.06, n)

    traj = np.zeros((n, 6), dtype=float)
    base = rng.normal(10.0, 2.0, n)
    for t in range(6):
        phi = np.where(y_transition == 1, 0.2 + 0.14 * t, 0.2)
        sig = np.where(y_transition == 1, 0.6 + 0.5 * t, 0.8)
        traj[:, t] = (
            base + (phi * (traj[:, t - 1] - base) if t > 0 else 0.0) + rng.normal(0, sig, n)
        )

    df_panel = pd.DataFrame({f"v_t{t}": traj[:, t] for t in range(6)})
    csd = (
        CriticalSlowingDownCompiler({"v": [f"v_t{t}" for t in range(6)]})
        .fit(df_panel)
        .transform(df_panel)
    )

    from sklearn.metrics import roc_auc_score

    auc_transition = roc_auc_score(y_transition, csd["csd__v__log_var_inflation"])
    auc_shock = roc_auc_score(y_shock, csd["csd__v__log_var_inflation"])
    assert auc_transition > 0.72
    assert abs(auc_shock - 0.50) < 0.05


def test_proposition1_adaptive_controls_per_positive_regimes() -> None:
    """Verify Proposition 1 prevalence-adaptive matching ratio m(pi) across all 3 regimes."""
    assert adaptive_controls_per_positive(0.1867) == 2
    assert adaptive_controls_per_positive(0.1116) == 2
    assert adaptive_controls_per_positive(0.0350) == 4
    assert adaptive_controls_per_positive(0.0017) == 8

    rng = np.random.default_rng(7)
    # Regime I (pi = 15%): adaptive m=2 cuts pilot size from 5*pi (~75%) to 3*pi (~45%)
    y_mod = np.zeros(1000, dtype=int)
    y_mod[:150] = 1
    scores_mod = rng.uniform(0.0, 1.0, 1000)
    p_m4 = sample_case_cohort_pilot(y_mod, scores_mod, controls_per_positive=4, seed=7)
    p_adapt = sample_case_cohort_pilot(y_mod, scores_mod, controls_per_positive=None, seed=7)
    assert abs(p_m4.fraction_of_cohort_measured - 0.75) < 0.01
    assert abs(p_adapt.fraction_of_cohort_measured - 0.45) < 0.01

