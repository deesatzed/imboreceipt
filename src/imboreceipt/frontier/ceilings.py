"""Stage E (Part 2): Decision-specific frontiers, 4-ceiling decomposition, and Headroom Index."""

from __future__ import annotations

import warnings
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.optimize import curve_fit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from imboreceipt.contract.decision import (
    DecisionContract,
    expected_net_utility_per_1k,
    gate_c_audit,
)
from imboreceipt.core.types import SemanticType
from imboreceipt.frontier.baselines import TabularPreprocessor, fit_single_calibrated_gbdt
from imboreceipt.receipt.audit import ReceiptAuditReport


class BottleneckRegime(StrEnum):
    """Gate E 4-account bottleneck classification."""

    DECISION_LIMITED = "decision_limited"
    RECEIPT_LIMITED = "receipt_limited"
    ESTIMATION_LIMITED = "estimation_limited"
    INFORMATION_LIMITED = "information_limited"


class CertifiedFrontierMetrics(BaseModel):
    """Out-of-sample decision-frontier metrics using validation-frozen thresholds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm_name: str
    auprc: float
    auroc: float
    brier_score: float
    ece: float
    np_tpr_at_fpr_budget: float
    np_realized_fpr: float
    recall_at_target_precision: float
    realized_precision_at_q_threshold: float
    net_utility_hard_per_1k: float
    net_utility_with_graded_per_1k: float


class FourCeilingDiagnosis(BaseModel):
    """Complete Stage E 4-ceiling decomposition and Headroom Index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_name: str
    prevalence: float
    achieved_auprc: float
    achieved_np_tpr: float
    observed_channel_auprc_ceiling_bound: float
    finite_positive_projected_asymptote: float
    learning_curve_slope_delta: float
    headroom_index: float
    receipt_adi: float
    bottleneck_regime: BottleneckRegime
    recommended_action: str


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error under natural prevalence."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(probs, dtype=float).ravel(), 0.0, 1.0)
    # Quantile + uniform hybrid edges so extreme rare-event tails are represented
    edges = np.unique(
        np.concatenate(
            [np.linspace(0.0, 1.0, n_bins + 1), np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))]
        )
    )
    ece = 0.0
    n = max(1, y.size)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p <= hi if hi == edges[-1] else p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(ece)


def select_neyman_pearson_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    alpha_fpr: float,
) -> float:
    """Select finite-sample order-statistic threshold controlling FPR <= alpha_fpr on val."""
    yv = np.asarray(y_val, dtype=int).ravel()
    sv = np.asarray(scores_val, dtype=float).ravel()
    neg_scores = np.sort(sv[yv == 0])
    m = neg_scores.size
    if m == 0:
        return 1.0
    # Conservative order statistic index (Tong et al. Neyman-Pearson umbrella rank)
    rank_idx = int(np.ceil((1.0 - alpha_fpr) * (m + 1))) - 1
    if rank_idx >= m:
        return float(neg_scores[-1] + 1e-9)
    rank_idx = max(0, rank_idx)
    return float(neg_scores[rank_idx])


def select_precision_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    target_precision: float,
    min_alerts: int = 3,
) -> float:
    """Select lowest validation threshold achieving precision >= target_precision (max recall)."""
    yv = np.asarray(y_val, dtype=int).ravel()
    sv = np.asarray(scores_val, dtype=float).ravel()
    order = np.argsort(-sv, kind="mergesort")
    y_sorted = yv[order]
    s_sorted = sv[order]

    cum_tp = np.cumsum(y_sorted == 1)
    cum_alerts = np.arange(1, yv.size + 1)
    precisions = cum_tp / cum_alerts

    valid = np.flatnonzero((precisions >= target_precision) & (cum_alerts >= min_alerts))
    if valid.size == 0:
        return float(np.max(sv) + 1e-6)
    best_k = int(valid[-1])  # furthest down the sorted list = maximum recall!
    return float(s_sorted[best_k])


def evaluate_certified_frontier(
    arm_name: str,
    y_val: np.ndarray,
    scores_val: np.ndarray,
    y_test: np.ndarray,
    scores_test: np.ndarray,
    contract: DecisionContract,
) -> CertifiedFrontierMetrics:
    """Freeze all operating thresholds on ``val``, then evaluate once on untouched ``test``."""
    yv = np.asarray(y_val, dtype=int).ravel()
    sv = np.asarray(scores_val, dtype=float).ravel()
    yt = np.asarray(y_test, dtype=int).ravel()
    st = np.asarray(scores_test, dtype=float).ravel()

    tau_np = select_neyman_pearson_threshold(yv, sv, contract.max_fpr_budget)
    tau_q = select_precision_threshold(yv, sv, contract.target_precision)

    # Optimal cost thresholds on calibrated probabilities
    p_hard = contract.cost_fp / (contract.cost_fp + contract.cost_fn)
    act_hard = (st >= p_hard).astype(int)
    act_graded: np.ndarray | None = None
    if contract.graded_action_cost_fp is not None:
        eff_fn = contract.graded_action_efficacy * contract.cost_fn
        p_grad = contract.graded_action_cost_fp / (contract.graded_action_cost_fp + eff_fn)
        act_graded = (st >= p_grad).astype(int)

    np_pred = st >= tau_np
    pos_t = max(1, int(np.sum(yt == 1)))
    neg_t = max(1, int(np.sum(yt == 0)))
    np_tpr = float(np.sum(np_pred & (yt == 1)) / pos_t)
    np_fpr = float(np.sum(np_pred & (yt == 0)) / neg_t)

    q_pred = st >= tau_q
    rec_at_q = float(np.sum(q_pred & (yt == 1)) / pos_t)
    alerts_q = int(np.sum(q_pred))
    prec_at_q = float(np.sum(q_pred & (yt == 1)) / alerts_q) if alerts_q > 0 else 0.0

    auprc = float(average_precision_score(yt, st)) if int(yt.sum()) > 0 else 0.0
    auroc = float(roc_auc_score(yt, st)) if (0 < int(yt.sum()) < yt.size) else 0.5

    return CertifiedFrontierMetrics(
        arm_name=arm_name,
        auprc=auprc,
        auroc=auroc,
        brier_score=float(brier_score_loss(yt, np.clip(st, 0.0, 1.0))),
        ece=compute_ece(yt, st),
        np_tpr_at_fpr_budget=np_tpr,
        np_realized_fpr=np_fpr,
        recall_at_target_precision=rec_at_q,
        realized_precision_at_q_threshold=prec_at_q,
        net_utility_hard_per_1k=expected_net_utility_per_1k(yt, act_hard, contract, None),
        net_utility_with_graded_per_1k=expected_net_utility_per_1k(
            yt, act_hard, contract, act_graded
        ),
    )


def estimate_observed_channel_ceiling_bound(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval: np.ndarray,
    achieved_auprc: float,
    semantic_types: dict[str, SemanticType] | None = None,
) -> float:
    """Estimate upper bound on Observed-Channel AUPRC ceiling given measured X.

    Combines Cover-Hart / local kNN collision overlap on standardized feature space
    with achieved model score (respecting the Asymmetry of Proof: true ceiling >= achieved).
    """
    prep = TabularPreprocessor(semantic_types=semantic_types).fit(X_train)
    X_tr_m = prep.transform(X_train)
    X_ev_m = prep.transform(X_eval)
    y_tr_arr = np.asarray(y_train, dtype=int)

    n = X_tr_m.shape[0]
    if n > 8000:
        rng = np.random.default_rng(42)
        pos_i = np.flatnonzero(y_tr_arr == 1)
        neg_i = np.flatnonzero(y_tr_arr == 0)
        keep_pos = rng.choice(pos_i, size=2500, replace=False) if pos_i.size > 2500 else pos_i
        keep_neg = rng.choice(
            neg_i, size=min(neg_i.size, 9500, max(4000, keep_pos.size * 5)), replace=False
        )
        sub = np.concatenate([keep_pos, keep_neg])
        w_pos = float(pos_i.size) / max(1.0, float(keep_pos.size))
        w_neg = float(neg_i.size) / max(1.0, float(keep_neg.size))
        X_ref = X_tr_m[sub]
        y_ref = y_tr_arr[sub]
    else:
        w_pos = 1.0
        w_neg = 1.0
        X_ref = X_tr_m
        y_ref = y_tr_arr

    k = min(21, max(5, int(np.sqrt(X_ref.shape[0]) // 2)))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=4)
    nn.fit(X_ref)
    # Out-of-sample kNN query on X_eval against training reference X_ref
    _, idx = nn.kneighbors(X_ev_m)
    nbr_y = y_ref[idx]
    pos_counts = np.sum(nbr_y == 1, axis=1).astype(float) * w_pos
    neg_counts = np.sum(nbr_y == 0, axis=1).astype(float) * w_neg
    eta_local = pos_counts / np.maximum(1e-9, pos_counts + neg_counts)

    order = np.argsort(-eta_local, kind="mergesort")
    eta_sorted = eta_local[order]
    expected_tp = np.cumsum(eta_sorted)
    expected_alerts = np.arange(1.0, len(eta_sorted) + 1.0)
    total_expected_pos = max(1e-9, float(expected_tp[-1]))
    prec_curve = expected_tp / expected_alerts
    delta_rec = eta_sorted / total_expected_pos
    knn_bayes_auprc = float(np.sum(prec_curve * delta_rec))

    # Bound above achieved score and clip to [achieved, 0.999]
    return float(np.clip(max(achieved_auprc * 1.02, knn_bayes_auprc), achieved_auprc, 0.999))


def estimate_finite_positive_learning_curve(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    semantic_types: dict[str, SemanticType] | None = None,
    seed: int = 42,
) -> tuple[float, float]:
    """Subsample training positives to estimate power-law asymptote ``L_inf`` and slope.

    Returns ``(projected_asymptote_auprc, gain_from_half_to_full_positives)``.
    """
    y_tr = np.asarray(y_train, dtype=int).ravel()
    y_v = np.asarray(y_val, dtype=int).ravel()
    pos_idx = np.flatnonzero(y_tr == 1)
    neg_idx = np.flatnonzero(y_tr == 0)

    if pos_idx.size < 25 or int(y_v.sum()) < 5:
        arm = fit_single_calibrated_gbdt(
            X_train, y_tr, X_cal, y_cal, semantic_types=semantic_types, seed=seed
        )
        full_score = float(average_precision_score(y_v, arm.predict_proba(X_val)))
        return min(0.99, full_score * 1.08), 0.04

    fracs = [0.25, 0.50, 0.75, 1.00]
    rng = np.random.default_rng(seed)
    n_pos_counts: list[float] = []
    scores: list[float] = []

    for f in fracs:
        k_pos = max(8, int(round(pos_idx.size * f)))
        chosen_pos = pos_idx if f == 1.0 else rng.choice(pos_idx, size=k_pos, replace=False)
        sub_idx = np.sort(np.concatenate([chosen_pos, neg_idx]))
        arm = fit_single_calibrated_gbdt(
            X_train.iloc[sub_idx],
            y_tr[sub_idx],
            X_cal,
            y_cal,
            semantic_types=semantic_types,
            seed=seed,
        )
        sc = float(average_precision_score(y_v, arm.predict_proba(X_val)))
        n_pos_counts.append(float(k_pos))
        scores.append(sc)

    slope_delta = float(scores[-1] - scores[1])  # gain from 50% n+ to 100% n+

    def _power_law(n_arr: np.ndarray, l_inf: float, a: float, b: float) -> np.ndarray:
        return l_inf - a * np.power(n_arr, -b)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _power_law,
                np.asarray(n_pos_counts),
                np.asarray(scores),
                p0=[min(0.99, scores[-1] + max(0.02, slope_delta)), 0.5, 0.5],
                bounds=([scores[-1], 0.0, 0.1], [0.999, 10.0, 2.0]),
                maxfev=1000,
            )
            l_inf = float(popt[0])
    except Exception:
        l_inf = float(min(0.999, scores[-1] + max(0.0, 1.5 * slope_delta)))

    return l_inf, slope_delta


def diagnose_four_ceilings(
    contract: DecisionContract,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    receipt_report: ReceiptAuditReport,
    semantic_types: dict[str, SemanticType] | None = None,
    seed: int = 42,
) -> FourCeilingDiagnosis:
    """Run complete Stage E 4-ceiling diagnostic and Gate E bottleneck classification."""
    y_tr = np.asarray(y_train, dtype=int).ravel()
    y_v = np.asarray(y_val, dtype=int).ravel()
    prev = float(y_tr.mean())

    arm = fit_single_calibrated_gbdt(
        X_train, y_tr, X_cal, y_cal, semantic_types=semantic_types, seed=seed
    )
    val_probs = arm.predict_proba(X_val)
    ach_auprc = float(average_precision_score(y_v, val_probs))
    tau_np = select_neyman_pearson_threshold(y_v, val_probs, contract.max_fpr_budget)
    ach_np_tpr = float(np.sum((val_probs >= tau_np) & (y_v == 1)) / max(1, int(y_v.sum())))

    obs_ceil = estimate_observed_channel_ceiling_bound(
        X_train, y_tr, X_val, y_v, achieved_auprc=ach_auprc, semantic_types=semantic_types
    )
    l_inf, slope_delta = estimate_finite_positive_learning_curve(
        X_train, y_tr, X_cal, y_cal, X_val, y_v, semantic_types=semantic_types, seed=seed
    )

    denom = max(1e-6, obs_ceil - prev)
    headroom_h = float(np.clip((ach_auprc - prev) / denom, 0.0, 1.0))

    gate_c = gate_c_audit(contract, prev, n_pos_test=int(y_v.sum()))

    # Gate E 4-way classification hierarchy
    if receipt_report.receipt_limited:
        regime = BottleneckRegime.RECEIPT_LIMITED
        rec = "Repair selective labels / censoring first (Stage R) before channel search."
    elif slope_delta >= 0.035 and (l_inf - ach_auprc) >= 0.04:
        regime = BottleneckRegime.ESTIMATION_LIMITED
        rec = "Learning curve still steep in n+: pool cohorts, precursor transfer, or regularize."
    elif not gate_c.one_stage_plausible and headroom_h >= 0.65:
        regime = BottleneckRegime.DECISION_LIMITED
        rec = "Base-rate LR deficit exceeds 1-stage capacity: use Two-Stage Lattice or Graded Tier."
    else:
        regime = BottleneckRegime.INFORMATION_LIMITED
        rec = "Observed channel saturated: freeze decoder tuning; run Stage D Channel Discovery."

    return FourCeilingDiagnosis(
        task_name=contract.task_name,
        prevalence=prev,
        achieved_auprc=ach_auprc,
        achieved_np_tpr=ach_np_tpr,
        observed_channel_auprc_ceiling_bound=obs_ceil,
        finite_positive_projected_asymptote=l_inf,
        learning_curve_slope_delta=slope_delta,
        headroom_index=headroom_h,
        receipt_adi=receipt_report.ascertainment_distortion_index,
        bottleneck_regime=regime,
        recommended_action=rec,
    )
