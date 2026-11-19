"""Stage R (Part 1): Auditing whether recorded labels are events or institutional receipts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class ReceiptAuditReport(BaseModel):
    """Diagnostic quantifying selective observation, censoring, and receipt leakage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observed_prevalence: float
    estimated_true_prevalence: float
    hidden_positive_fraction_among_negatives: float
    observation_intensity_auc: float
    future_positive_censoring_rate: float
    competing_event_censoring_rate: float
    chapman_multi_trap_total_positives: float | None = None
    ascertainment_distortion_index: float
    receipt_limited: bool


def _estimate_observation_intensity_auc(
    X: pd.DataFrame,
    y_tilde: np.ndarray,
    receipt_cols: list[str] | None = None,
    seed: int = 42,
) -> float:
    """Measure how strongly missingness patterns + ordering/receipt flags alone predict Y_tilde."""
    indicators: dict[str, np.ndarray] = {}
    for col in X.columns:
        miss = (X[col].isna() | (X[col].astype(str) == "__MISSING__")).to_numpy(dtype=float)
        if 0.005 < miss.mean() < 0.995:
            indicators[f"miss__{col}"] = miss

    if receipt_cols:
        for col in receipt_cols:
            if col in X.columns:
                vals = pd.to_numeric(X[col], errors="coerce")
                med = float(vals.median()) if vals.notna().any() else 0.0
                arr = vals.fillna(med).to_numpy(dtype=float)
                std = float(np.std(arr))
                if std > 1e-9:
                    indicators[f"rcpt__{col}"] = (arr - float(np.mean(arr))) / std

    if not indicators or int(y_tilde.sum()) < 10:
        return 0.5

    Z = np.column_stack(list(indicators.values()))
    n_splits = min(5, int(y_tilde.sum()), int((1 - y_tilde).sum()))
    if n_splits < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(y_tilde.size, dtype=float)
    for tr, va in skf.split(Z, y_tilde):
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(Z[tr], y_tilde[tr])
        oof[va] = clf.predict_proba(Z[va])[:, 1]

    auc = float(roc_auc_score(y_tilde, oof))
    return max(auc, 1.0 - auc)


def chapman_capture_recapture_estimate(trap_a: np.ndarray, trap_b: np.ndarray) -> float:
    """Chapman bias-corrected two-trap capture-recapture abundance estimator.

    ``N_hat = ((n1 + 1) * (n2 + 1)) / (m2 + 1) - 1``
    where ``n1 = sum(trap_a)``, ``n2 = sum(trap_b)``, ``m2 = sum(trap_a & trap_b)``.
    """
    a = np.asarray(trap_a, dtype=int).ravel() > 0
    b = np.asarray(trap_b, dtype=int).ravel() > 0
    n1 = float(np.sum(a))
    n2 = float(np.sum(b))
    m2 = float(np.sum(a & b))
    union_obs = float(np.sum(a | b))
    est = ((n1 + 1.0) * (n2 + 1.0)) / (m2 + 1.0) - 1.0
    return float(max(union_obs, est))


def audit_receipts(
    X_train: pd.DataFrame,
    y_tilde_train: np.ndarray,
    *,
    receipt_cols: list[str] | None = None,
    future_positive_mask: np.ndarray | None = None,
    competing_censor_mask: np.ndarray | None = None,
    traps_df: pd.DataFrame | None = None,
    seed: int = 42,
) -> ReceiptAuditReport:
    """Audit training data for selective labeling, censoring, and observation-policy leakage."""
    y = np.asarray(y_tilde_train, dtype=int).ravel()
    n = max(1, y.size)
    obs_prev = float(y.mean())

    obs_auc = _estimate_observation_intensity_auc(X_train, y, receipt_cols=receipt_cols, seed=seed)

    neg_mask = y == 0
    n_neg = max(1, int(np.sum(neg_mask)))

    fut_rate = 0.0
    if future_positive_mask is not None:
        fp_m = np.asarray(future_positive_mask, dtype=bool).ravel()
        fut_rate = float(np.sum(fp_m & neg_mask) / n_neg)

    comp_rate = 0.0
    if competing_censor_mask is not None:
        cc_m = np.asarray(competing_censor_mask, dtype=bool).ravel()
        comp_rate = float(np.sum(cc_m & neg_mask) / n_neg)

    chapman_total: float | None = None
    est_true_pos = float(np.sum(y))

    if traps_df is not None and traps_df.shape[1] >= 2:
        t0 = traps_df.iloc[:, 0].to_numpy(dtype=int)
        t1 = traps_df.iloc[:, 1].to_numpy(dtype=int)
        chapman_total = chapman_capture_recapture_estimate(t0, t1)
        # Generalize across >2 traps via pairwise median if more traps exist
        if traps_df.shape[1] > 2:
            ests: list[float] = []
            for i in range(traps_df.shape[1]):
                for j in range(i + 1, traps_df.shape[1]):
                    ests.append(
                        chapman_capture_recapture_estimate(
                            traps_df.iloc[:, i].to_numpy(dtype=int),
                            traps_df.iloc[:, j].to_numpy(dtype=int),
                        )
                    )
            chapman_total = float(np.median(ests))
        est_true_pos = max(est_true_pos, chapman_total)

    if future_positive_mask is not None:
        est_true_pos += 0.5 * float(np.sum(np.asarray(future_positive_mask, dtype=bool) & neg_mask))

    effective_denom = max(1.0, float(n) - float(comp_rate * n_neg))
    est_true_prev = float(np.clip(est_true_pos / effective_denom, obs_prev, 0.99))
    hidden_pos_among_neg = float(np.clip((est_true_pos - float(np.sum(y))) / n_neg, 0.0, 0.95))

    # Ascertainment Distortion Index combines missing-positive undercount, censoring,
    # and observation-intensity predictability into [0, 1]
    undercount_ratio = 1.0 - (obs_prev / max(1e-9, est_true_prev))
    intensity_excess = max(0.0, (obs_auc - 0.55) / 0.45)
    censor_component = min(1.0, 2.0 * (fut_rate + comp_rate))
    adi = float(
        np.clip(
            0.50 * undercount_ratio + 0.25 * intensity_excess + 0.25 * censor_component, 0.0, 1.0
        )
    )

    return ReceiptAuditReport(
        observed_prevalence=obs_prev,
        estimated_true_prevalence=est_true_prev,
        hidden_positive_fraction_among_negatives=hidden_pos_among_neg,
        observation_intensity_auc=obs_auc,
        future_positive_censoring_rate=fut_rate,
        competing_event_censoring_rate=comp_rate,
        chapman_multi_trap_total_positives=chapman_total,
        ascertainment_distortion_index=adi,
        receipt_limited=bool(adi >= 0.12 or fut_rate >= 0.03 or comp_rate >= 0.02),
    )
