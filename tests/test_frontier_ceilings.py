"""Tests for Stage E: Frontier saturation (Prediction 1) and 4-ceiling bottleneck diagnosis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score

from imboreceipt.contract import DecisionContract
from imboreceipt.core import paired_cluster_bootstrap
from imboreceipt.frontier import (
    BottleneckRegime,
    diagnose_four_ceilings,
    evaluate_certified_frontier,
    train_decoder_saturation_suite,
)
from imboreceipt.receipt import audit_receipts


def test_prediction1_frontier_saturation_and_calibration_degradation() -> None:
    """On an information-limited dataset, SMOTE/Copula do not beat calibrated GBDT beyond CI."""
    rng = np.random.default_rng(12)
    n = 6000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Unobserved driver z_missing carries half the log-odds -> X=(x1, x2) is information-limited!
    z_missing = rng.normal(0, 1.5, n)
    prob = expit(-3.4 + 1.1 * x1 - 0.8 * x2 + 1.5 * z_missing)
    y = rng.binomial(1, prob)

    X = pd.DataFrame({"x1": x1, "x2": x2, "noise": rng.normal(0, 1, n)})
    X_tr, y_tr = X.iloc[:3300], y[:3300]
    X_cal, y_cal = X.iloc[3300:4200], y[3300:4200]
    X_val, y_val = X.iloc[4200:5100], y[4200:5100]
    X_tst, y_tst = X.iloc[5100:], y[5100:]

    contract = DecisionContract(
        task_name="info_limited", target_precision=0.35, max_fpr_budget=0.02
    )
    rcpt = audit_receipts(X_tr, y_tr)
    diag = diagnose_four_ceilings(contract, X_tr, y_tr, X_cal, y_cal, X_val, y_val, rcpt, seed=12)
    assert diag.bottleneck_regime in {
        BottleneckRegime.INFORMATION_LIMITED,
        BottleneckRegime.DECISION_LIMITED,
    }

    arms = train_decoder_saturation_suite(X_tr, y_tr, X_cal, y_cal, seed=12)
    p_gbdt = arms["calibrated_gbdt"].predict_proba(X_tst)
    p_smote_uncal = arms["smote_gbdt_uncal"].predict_proba(X_tst)
    p_smote_cal = arms["smote_gbdt"].predict_proba(X_tst)
    p_copula = arms["latent_copula_gbdt"].predict_proba(X_tst)

    m_gbdt = evaluate_certified_frontier(
        "gbdt", y_val, arms["calibrated_gbdt"].predict_proba(X_val), y_tst, p_gbdt, contract
    )
    m_smote_uncal = evaluate_certified_frontier(
        "smote_uncal",
        y_val,
        arms["smote_gbdt_uncal"].predict_proba(X_val),
        y_tst,
        p_smote_uncal,
        contract,
    )

    # Uncalibrated SMOTE severely distorts calibration error under natural prevalence!
    assert m_smote_uncal.ece > m_gbdt.ece + 0.03

    # Neither SMOTE nor Latent Copula significantly beats calibrated GBDT on AUPRC
    boot_smote = paired_cluster_bootstrap(
        y_tst, p_gbdt, p_smote_cal, average_precision_score, seed=12
    )
    boot_cop = paired_cluster_bootstrap(y_tst, p_gbdt, p_copula, average_precision_score, seed=12)
    assert boot_smote.ci_low <= 0.015
    assert boot_cop.ci_low <= 0.015
