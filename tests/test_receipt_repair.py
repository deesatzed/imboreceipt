"""Tests for Stage R: Receipt auditing, multi-trap capture-recapture repair, and epsilon-audits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.metrics import average_precision_score

from imboreceipt.frontier import fit_single_calibrated_gbdt
from imboreceipt.receipt import (
    ReceiptRepairer,
    audit_receipts,
    chapman_capture_recapture_estimate,
    epsilon_random_audit_metrics,
)


def test_chapman_capture_recapture_recovers_true_positives() -> None:
    rng = np.random.default_rng(101)
    n_true_pos = 400
    # Trap A catches 50%, Trap B catches 40% independently -> union catches 70% (280), misses 120!
    trap_a = np.concatenate([rng.binomial(1, 0.50, n_true_pos), np.zeros(4600, dtype=int)])
    trap_b = np.concatenate([rng.binomial(1, 0.40, n_true_pos), np.zeros(4600, dtype=int)])

    est_total = chapman_capture_recapture_estimate(trap_a, trap_b)
    assert est_total == pytest.approx(n_true_pos, rel=0.12)
    assert est_total > float(np.sum((trap_a | trap_b) > 0)) + 60.0


def test_multi_trap_receipt_repair_beats_naive_selective_labels() -> None:
    """Verify H1/Prediction 6: repairing policy-biased receipts improves out-of-policy ranking."""
    rng = np.random.default_rng(42)
    n = 5000
    x_risk = rng.normal(0, 1, n)
    # Policy confounder: Group B (z_policy < 0) is rarely inspected by primary rules!
    z_policy = rng.normal(0, 1, n)

    true_prob = expit(-3.0 + 1.8 * x_risk - 0.8 * z_policy)
    y_true = rng.binomial(1, true_prob)

    # Trap 1 strongly favors z_policy > 0; Trap 2 is sparse but less policy-skewed
    p_trap1 = expit(0.2 + 2.0 * z_policy)
    p_trap2 = expit(-1.1 - 0.2 * z_policy)
    t1 = (y_true == 1) & (rng.uniform(size=n) < p_trap1)
    t2 = (y_true == 1) & (rng.uniform(size=n) < p_trap2)
    y_tilde = (t1 | t2).astype(int)

    X = pd.DataFrame({"x_risk": x_risk, "z_policy": z_policy})
    traps = pd.DataFrame({"t1": t1.astype(int), "t2": t2.astype(int)})

    audit = audit_receipts(X.iloc[:3500], y_tilde[:3500], traps_df=traps.iloc[:3500])
    assert audit.receipt_limited
    assert audit.estimated_true_prevalence > audit.observed_prevalence * 1.15

    # Compare Naive GBDT on y_tilde vs ReceiptRepairer GBDT evaluated on TRUE y_true holdout!
    naive_arm = fit_single_calibrated_gbdt(
        X.iloc[:3000], y_tilde[:3000], X.iloc[3000:3800], y_tilde[3000:3800], seed=42
    )
    repaired = ReceiptRepairer(seed=42).fit_transform_train(
        X.iloc[:3000], y_tilde[:3000], traps_train=traps.iloc[:3000]
    )
    repaired_arm = fit_single_calibrated_gbdt(
        repaired.X_expanded,
        repaired.y_expanded,
        X.iloc[3000:3800],
        y_tilde[3000:3800],
        sample_weight=repaired.sample_weight,
        seed=42,
    )

    X_test = X.iloc[3800:]
    y_test_true = y_true[3800:]
    auprc_naive = average_precision_score(y_test_true, naive_arm.predict_proba(X_test))
    auprc_repaired = average_precision_score(y_test_true, repaired_arm.predict_proba(X_test))
    assert auprc_repaired > auprc_naive + 0.015


def test_epsilon_random_audit_horvitz_thompson_unbiasedness() -> None:
    rng = np.random.default_rng(77)
    n = 10000
    y_true = rng.binomial(1, 0.04, size=n)
    # Classifier catches 50% of positives with 2% FPR
    alerts = np.where(y_true == 1, rng.binomial(1, 0.50, n), rng.binomial(1, 0.02, n))
    # Without random audit, unalerted positives are never seen -> naive recall looks like 100%!
    y_tilde = (alerts & y_true).astype(int)
    audit_mask = rng.uniform(size=n) < 0.10  # 10% random audit of unalerted pool

    est = epsilon_random_audit_metrics(y_tilde, y_true, audit_mask, alerts)
    true_prec = float(np.sum((alerts == 1) & (y_true == 1)) / np.sum(alerts == 1))
    assert est["precision"] == pytest.approx(true_prec, abs=1e-6)
    assert est["recall"] == pytest.approx(0.50, abs=0.08)
    assert est["estimated_prevalence"] == pytest.approx(0.04, abs=0.01)
