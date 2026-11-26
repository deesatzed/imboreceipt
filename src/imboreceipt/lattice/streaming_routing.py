"""Streaming score-threshold selective feature acquisition and Conformal FDR selection.

Implements:
1. Real-time streaming threshold routing across full query-rate frontiers q in [0, 1]
   calibrated strictly on D_cal with prespecified calibration-exact randomized tie-breaking
   (no batch test-set argsort and no hard-zero PR truncation).
2. Coherent Martingale Mean-Centered Logit-Normal Value-of-Information (VOI) routing
   (Proposition 2: unique offset mu*(p_1, sigma) solving E[p_2 | p_1] = p_1 on the 9-node
   Gauss-Hermite measure so pre-clip VOI >= 0 identically), alongside uncentered heuristic
   VOI diagnostics (E0/E1), flip-level Net Benefit decompositions (E2), and autonomous
   Bayes stopping VOI_K(x) > c_query.
3. Conformal Selection with False Discovery Rate (FDR = 1 - PPV) control via smoothed
   conformal p-values and Benjamini-Hochberg (CfBH; Jin & Candes, 2023).
4. Clinical & statistical calibration metrics: ECE, calibration slope, calibration-in-the-large,
   Precision at 50% Recall, and Vickers-Elkin Net Benefit net of query cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, precision_recall_curve

from imboreceipt.contract.decision import DecisionContract

_PROB_EPS = 1e-9

# 9-point Gauss-Hermite quadrature nodes and weights for E_{xi ~ N(0,1)}[g(xi)]
_GH_X_RAW, _GH_W_RAW = np.polynomial.hermite.hermgauss(9)
_GH_NODES = _GH_X_RAW * np.sqrt(2.0)
_GH_WEIGHTS = _GH_W_RAW / np.sqrt(np.pi)


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Probability calibration diagnostics (Van Calster et al., 2019; Austin & Steyerberg, 2019)."""

    ece: float
    brier: float
    cal_intercept: float
    cal_slope: float
    ici: float = 0.0


@dataclass(frozen=True, slots=True)
class ConformalFDRResult:
    """Jin & Candes (2023) Conformal BH (CfBH) positive-selection evaluation."""

    target_fdr: float
    target_ppv: float
    n_selected: int
    selection_rate: float
    realized_fdp: float
    realized_ppv: float
    realized_recall: float


@dataclass(frozen=True, slots=True)
class FlipDecomposition:
    """Exact decomposition of Net Benefit change (per 1,000 arrivals) relative to Stage 1 (E2)."""

    beneficial_flip_nb_gain_per_1k: float
    harmful_flip_nb_loss_per_1k: float
    query_fee_nb_drag_per_1k: float
    net_nb_delta_vs_stage1_per_1k: float
    queried_rate: float
    decision_flip_rate: float
    unchanged_queried_rate: float
    above_tau_unchanged_queried_rate: float


@dataclass(frozen=True, slots=True)
class StreamingBudgetPoint:
    """Evaluation of a streaming routing policy at a single calibration-tuned threshold."""

    policy: str
    nominal_query_rate: float
    realized_query_rate: float
    auprc: float
    prec_at_50_recall: float
    net_benefit_per_1k: float
    total_cost_per_1k: float
    ece: float
    cal_slope: float
    cfbh_recall: float
    cfbh_realized_fdp: float
    beneficial_flip_nb_gain_per_1k: float = 0.0
    harmful_flip_nb_loss_per_1k: float = 0.0
    query_fee_nb_drag_per_1k: float = 0.0
    unchanged_queried_rate: float = 0.0
    above_tau_unchanged_queried_rate: float = 0.0


def solve_mean_preserving_logit_offset(p1: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Solve sum_r omega_r expit(logit(p_1) + mu* + sigma * xi_r) = p_1 (Proposition 2).

    Because f(mu) is strictly increasing from 0 to 1 with derivative
    f'(mu) = sum_r omega_r p_{2,r}(1 - p_{2,r}) > 0, vectorized Newton-Raphson converges
    quadratically to machine precision (< 1e-14) in 8 iterations across all rows simultaneously.
    """
    p = np.clip(np.asarray(p1, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
    sig = np.maximum(0.0, np.asarray(sigma, dtype=float).ravel())
    lp = logit(p)

    # Analytic second-order initialization: -0.5 * sig^2 * (1 - 2p) / (1 + 0.35 * sig^2)
    mu = -0.5 * (sig**2) * (1.0 - 2.0 * p) / (1.0 + 0.35 * (sig**2))
    zero_sig = sig <= 1e-12
    if np.all(zero_sig):
        return np.zeros_like(p)

    nodes = _GH_NODES[None, :]  # (1, 9)
    weights = _GH_WEIGHTS[None, :]  # (1, 9)
    base_arg = lp[:, None] + sig[:, None] * nodes  # (N, 9)

    for _ in range(10):
        probs = expit(base_arg + mu[:, None])
        f_val = np.sum(weights * probs, axis=1) - p
        if float(np.max(np.abs(f_val))) < 1e-14:
            break
        f_der = np.sum(weights * probs * (1.0 - probs), axis=1)
        step = np.clip(f_val / np.maximum(1e-14, f_der), -4.0, 4.0)
        mu = mu - step

    mu[zero_sig] = 0.0
    return mu


def calibrated_randomized_threshold_masks(
    scores_cal: np.ndarray,
    scores_test: np.ndarray,
    target_q: float,
    u_cal: np.ndarray,
    u_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute prespecified calibration-exact randomized tie-broken query masks.

    Guarantees E_U[mean(mask_cal)] == target_q on D_cal even when scores have point-mass ties
    (e.g., clipped zeros or discrete GBDT leaf values). Returns (mask_cal, mask_test, gamma_q,
    tie_prob_at_cutoff).
    """
    sc = np.asarray(scores_cal, dtype=float).ravel()
    st = np.asarray(scores_test, dtype=float).ravel()
    q = float(np.clip(target_q, 0.0, 1.0))
    if q <= 0.0:
        return np.zeros(sc.size, dtype=bool), np.zeros(st.size, dtype=bool), float("inf"), 0.0
    if q >= 1.0:
        return np.ones(sc.size, dtype=bool), np.ones(st.size, dtype=bool), float("-inf"), 1.0

    gamma = float(np.quantile(sc, 1.0 - q))
    frac_gt = float(np.mean(sc > gamma))
    frac_eq = float(np.mean(sc == gamma))
    if frac_eq > 0.0:
        rho_tie = float(np.clip((q - frac_gt) / frac_eq, 0.0, 1.0))
    else:
        rho_tie = 0.0

    m_cal = (sc > gamma) | ((sc == gamma) & (u_cal < rho_tie))
    m_tst = (st > gamma) | ((st == gamma) & (u_test < rho_tie))
    return m_cal, m_tst, gamma, rho_tie


def decompose_routing_flips(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    query_mask: np.ndarray,
    contract: DecisionContract,
    per_instance_query_cost: float,
) -> FlipDecomposition:
    """Decompose Net Benefit delta (route minus Stage 1) into flips and query fees (E2)."""
    yt = np.asarray(y_true, dtype=int).ravel()
    q_m = np.asarray(query_mask, dtype=bool).ravel()
    n_t = max(1.0, float(yt.size))
    c_fp = float(contract.cost_fp)
    c_fn = max(1e-9, float(contract.cost_fn))
    tau_star = c_fp / (c_fp + c_fn)

    p_route = np.where(q_m, p2, p1)
    a1 = p1 >= tau_star
    ar = p_route >= tau_star

    # Beneficial flips: (0->1 on Y=1) saves C_FN; (1->0 on Y=0) saves C_FP
    tp_gained = float(np.sum((~a1) & ar & (yt == 1)))
    fp_avoided = float(np.sum(a1 & (~ar) & (yt == 0)))
    ben_nb_1k = (1000.0 / n_t) * (tp_gained + (c_fp / c_fn) * fp_avoided)

    # Harmful flips: (1->0 on Y=1) loses C_FN; (0->1 on Y=0) adds C_FP
    tp_lost = float(np.sum(a1 & (~ar) & (yt == 1)))
    fp_added = float(np.sum((~a1) & ar & (yt == 0)))
    harm_nb_1k = (1000.0 / n_t) * (tp_lost + (c_fp / c_fn) * fp_added)

    n_q = float(np.sum(q_m))
    fee_drag_1k = (1000.0 / n_t) * (float(per_instance_query_cost) / c_fn) * n_q
    net_delta_1k = ben_nb_1k - harm_nb_1k - fee_drag_1k

    flips = a1 != ar
    unchanged_q = q_m & (~flips)
    above_tau_unchanged_q = q_m & a1 & ar

    return FlipDecomposition(
        beneficial_flip_nb_gain_per_1k=float(ben_nb_1k),
        harmful_flip_nb_loss_per_1k=float(harm_nb_1k),
        query_fee_nb_drag_per_1k=float(fee_drag_1k),
        net_nb_delta_vs_stage1_per_1k=float(net_delta_1k),
        queried_rate=float(n_q / n_t),
        decision_flip_rate=float(np.sum(flips)) / n_t,
        unchanged_queried_rate=float(np.sum(unchanged_q)) / n_t,
        above_tau_unchanged_queried_rate=float(np.sum(above_tau_unchanged_q)) / n_t,
    )


def compute_smooth_ici(y_true: np.ndarray, prob: np.ndarray, n_grid: int = 80) -> float:
    """Compute smooth tricube LOESS Integrated Calibration Index (Austin & Steyerberg, 2019)."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(prob, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
    n = int(y.size)
    if n < 20 or np.unique(y).size < 2:
        return abs(float(np.mean(y) - np.mean(p)))

    lp = logit(p)
    lo_q, hi_q = np.quantile(lp, [0.005, 0.995])
    if hi_q - lo_q < 1e-6:
        return abs(float(np.mean(y) - np.mean(p)))

    grid = np.linspace(float(lo_q), float(hi_q), n_grid)
    span = max(0.35, float(hi_q - lo_q) * 0.28)
    smooth_grid = np.empty(n_grid, dtype=float)

    for g_i, g_val in enumerate(grid):
        dist = np.abs(lp - g_val)
        bw = max(
            span,
            float(
                np.partition(dist, min(n - 1, max(25, int(0.25 * n))))[
                    min(n - 1, max(25, int(0.25 * n)))
                ]
            ),
        )
        u = np.clip(dist / max(1e-8, bw), 0.0, 1.0)
        w = (1.0 - (u**3)) ** 3
        mask = w > 1e-6
        if int(np.sum(mask)) < 10:
            smooth_grid[g_i] = float(np.mean(y))
            continue
        wm = w[mask]
        xm = lp[mask] - g_val
        ym = y[mask]
        sw = float(np.sum(wm))
        sx = float(np.sum(wm * xm))
        sxx = float(np.sum(wm * xm * xm))
        sy = float(np.sum(wm * ym))
        sxy = float(np.sum(wm * xm * ym))
        denom = sw * sxx - sx * sx
        if denom > 1e-8:
            b0 = (sxx * sy - sx * sxy) / denom
        else:
            b0 = sy / max(1e-12, sw)
        smooth_grid[g_i] = float(np.clip(b0, 0.0, 1.0))

    p_hat_smooth = np.interp(np.clip(lp, float(lo_q), float(hi_q)), grid, smooth_grid)
    return float(np.mean(np.abs(p_hat_smooth - p)))


def compute_calibration_summary(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15
) -> CalibrationSummary:
    """Compute smooth LOESS ICI, ECE, Brier score, calibration-in-the-large, and slope."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(prob, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
    brier = float(np.mean((y - p) ** 2))
    ici = compute_smooth_ici(y, p)

    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size <= 2:
        ece = abs(float(np.mean(y) - np.mean(p)))
    else:
        bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, edges.size - 2)
        ece = 0.0
        n_tot = float(y.size)
        for b in range(edges.size - 1):
            m = bins == b
            cnt = int(np.sum(m))
            if cnt > 0:
                ece += (cnt / n_tot) * abs(float(np.mean(y[m]) - np.mean(p[m])))

    lp = logit(p)
    if np.unique(y).size < 2 or float(np.std(lp)) < 1e-8:
        return CalibrationSummary(
            ece=float(ece), brier=brier, cal_intercept=0.0, cal_slope=1.0, ici=ici
        )

    def _nll_int(a_arr: np.ndarray) -> float:
        eta = a_arr[0] + lp
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

    res_int = minimize(_nll_int, x0=np.array([0.0]), method="BFGS")
    cal_int = float(res_int.x[0]) if res_int.success else 0.0

    def _nll_slope(ab: np.ndarray) -> float:
        eta = ab[0] + ab[1] * lp
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 1e-6 * (ab[1] - 1.0) ** 2)

    res_sl = minimize(_nll_slope, x0=np.array([0.0, 1.0]), method="BFGS")
    cal_slope = float(res_sl.x[1]) if res_sl.success else 1.0

    return CalibrationSummary(
        ece=float(ece),
        brier=brier,
        cal_intercept=cal_int,
        cal_slope=cal_slope,
        ici=ici,
    )


def precision_at_target_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.50,
) -> float:
    """Compute maximum precision achieved at recall >= target_recall."""
    y = np.asarray(y_true, dtype=int).ravel()
    if int(np.sum(y)) == 0:
        return 0.0
    prec, rec, _ = precision_recall_curve(y, np.asarray(scores, dtype=float).ravel())
    mask = rec >= target_recall - 1e-9
    return float(np.max(prec[mask])) if np.any(mask) else 0.0


def conformal_fdr_selection(
    cal_scores: np.ndarray,
    y_cal: np.ndarray,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    target_fdr: float = 0.35,
    seed: int = 42,
) -> ConformalFDRResult:
    """Conformal Selection with False Discovery Rate control (CfBH; Jin & Candes, 2023, JMLR)."""
    yc = np.asarray(y_cal, dtype=int).ravel()
    yt = np.asarray(y_test, dtype=int).ravel()
    sc = np.asarray(cal_scores, dtype=float).ravel()
    neg_cal = np.sort(sc[yc == 0])
    st = np.asarray(test_scores, dtype=float).ravel()

    n_cal = int(yc.size)
    m0 = int(neg_cal.size)
    n_test = int(st.size)
    target_ppv = 1.0 - float(target_fdr)
    if n_cal == 0 or m0 == 0 or n_test == 0:
        return ConformalFDRResult(
            target_fdr=float(target_fdr),
            target_ppv=target_ppv,
            n_selected=0,
            selection_rate=0.0,
            realized_fdp=0.0,
            realized_ppv=1.0,
            realized_recall=0.0,
        )

    idx_left = np.searchsorted(neg_cal, st, side="left")
    idx_right = np.searchsorted(neg_cal, st, side="right")
    n_gt = (m0 - idx_right).astype(float)
    n_eq = (idx_right - idx_left).astype(float)
    u_tie = np.random.default_rng(seed).uniform(0.0, 1.0, size=n_test)
    p_conf = (1.0 + n_gt + u_tie * n_eq) / float(1 + n_cal)

    order = np.argsort(p_conf)
    p_sorted = p_conf[order]
    thresh = float(target_fdr) * (np.arange(1, n_test + 1, dtype=float) / float(n_test))
    passing = np.flatnonzero(p_sorted <= thresh)

    if passing.size == 0:
        return ConformalFDRResult(
            target_fdr=float(target_fdr),
            target_ppv=target_ppv,
            n_selected=0,
            selection_rate=0.0,
            realized_fdp=0.0,
            realized_ppv=1.0,
            realized_recall=0.0,
        )

    k_hat = int(passing[-1]) + 1
    sel_mask = np.zeros(n_test, dtype=bool)
    sel_mask[order[:k_hat]] = True

    n_sel = int(np.sum(sel_mask))
    tp = int(np.sum((yt == 1) & sel_mask))
    fp = n_sel - tp
    fdp = float(fp) / float(n_sel)
    ppv = float(tp) / float(n_sel)
    n_pos = max(1, int(np.sum(yt == 1)))
    rec = float(tp) / float(n_pos)
    return ConformalFDRResult(
        target_fdr=float(target_fdr),
        target_ppv=target_ppv,
        n_selected=n_sel,
        selection_rate=float(n_sel) / float(n_test),
        realized_fdp=fdp,
        realized_ppv=ppv,
        realized_recall=rec,
    )


def weighted_conformal_fdr_selection(
    scores_cal: np.ndarray,
    y_cal: np.ndarray,
    weights_cal: np.ndarray,
    scores_test: np.ndarray,
    y_test: np.ndarray,
    weights_test: np.ndarray,
    target_fdr: float,
    seed: int = 42,
) -> ConformalFDRResult:
    """Jin & Candes (2025, Biometrika) Weighted Conformalized Selection (WCS) under covariate shift.

    Computes covariate-shift-weighted conformal p-values using unlabeled likelihood-ratio weights
    ``w(x) = dP_test(x) / dP_cal(x) > 0`` followed by Benjamini-Hochberg step-up selection:

        p_j^WCS = [sum_{i in D_cal: Y_i=0} w_i * 1{S_i > S_test,j}
                   + U_j * (w_test,j + sum_{i in D_cal: Y_i=0} w_i * 1{S_i == S_test,j})]
                  / (w_test,j + sum_{i in D_cal} w_i).
    """
    sc = np.asarray(scores_cal, dtype=float).ravel()
    yc = np.asarray(y_cal, dtype=int).ravel()
    wc = np.maximum(1e-8, np.asarray(weights_cal, dtype=float).ravel())
    st = np.asarray(scores_test, dtype=float).ravel()
    yt = np.asarray(y_test, dtype=int).ravel()
    wt = np.maximum(1e-8, np.asarray(weights_test, dtype=float).ravel())

    n_cal = sc.size
    n_test = st.size
    m0 = int(np.sum(yc == 0))
    target_ppv = 1.0 - float(target_fdr)
    if n_cal == 0 or m0 == 0 or n_test == 0:
        return ConformalFDRResult(
            target_fdr=float(target_fdr),
            target_ppv=target_ppv,
            n_selected=0,
            selection_rate=0.0,
            realized_fdp=0.0,
            realized_ppv=1.0,
            realized_recall=0.0,
        )

    neg_mask = yc == 0
    sc_neg = sc[neg_mask]
    wc_neg = wc[neg_mask]
    ord_neg = np.argsort(sc_neg)
    sc_neg_s = sc_neg[ord_neg]
    wc_neg_s = wc_neg[ord_neg]
    cum_w = np.concatenate([[0.0], np.cumsum(wc_neg_s)])
    tot_w_neg = float(cum_w[-1])
    tot_w_cal = float(np.sum(wc))

    idx_l = np.searchsorted(sc_neg_s, st, side="left")
    idx_r = np.searchsorted(sc_neg_s, st, side="right")
    w_gt = tot_w_neg - cum_w[idx_r]
    w_eq = cum_w[idx_r] - cum_w[idx_l]

    u_rand = np.random.default_rng(seed).uniform(0.0, 1.0, size=n_test)
    p_wcs = (w_gt + u_rand * (wt + w_eq)) / (wt + tot_w_cal)

    order = np.argsort(p_wcs)
    p_sorted = p_wcs[order]
    thresh = float(target_fdr) * (np.arange(1, n_test + 1, dtype=float) / float(n_test))
    passing = np.flatnonzero(p_sorted <= thresh)
    if passing.size == 0:
        return ConformalFDRResult(
            target_fdr=float(target_fdr),
            target_ppv=target_ppv,
            n_selected=0,
            selection_rate=0.0,
            realized_fdp=0.0,
            realized_ppv=1.0,
            realized_recall=0.0,
        )

    k_hat = int(passing[-1]) + 1
    sel_mask = np.zeros(n_test, dtype=bool)
    sel_mask[order[:k_hat]] = True
    n_sel = int(np.sum(sel_mask))
    tp = int(np.sum((yt == 1) & sel_mask))
    fp = n_sel - tp
    n_pos = max(1, int(np.sum(yt == 1)))
    return ConformalFDRResult(
        target_fdr=float(target_fdr),
        target_ppv=target_ppv,
        n_selected=n_sel,
        selection_rate=float(n_sel) / float(n_test),
        realized_fdp=float(fp) / float(n_sel),
        realized_ppv=float(tp) / float(n_sel),
        realized_recall=float(tp) / float(n_pos),
    )


def _bayes_action_loss(p: np.ndarray, cost_fp: float, cost_fn: float) -> np.ndarray:
    return np.minimum(p * cost_fn, (1.0 - p) * cost_fp)


def solve_empirical_quantile_logit_offset(
    p1: np.ndarray, quantile_deltas: np.ndarray
) -> np.ndarray:
    """Solve (1/K) sum_{r=1}^K expit(logit(p_1) + mu* + delta_{b,r}) = p_1 via Newton-Raphson."""
    p = np.clip(np.asarray(p1, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
    deltas = np.asarray(quantile_deltas, dtype=float)  # (N, K)
    if deltas.size == 0 or float(np.max(np.abs(deltas))) < 1e-12:
        return np.zeros_like(p)
    lp = logit(p)
    mu = -np.mean(deltas, axis=1)
    base_arg = lp[:, None] + deltas

    for _ in range(12):
        probs = expit(base_arg + mu[:, None])
        f_val = np.mean(probs, axis=1) - p
        if float(np.max(np.abs(f_val))) < 1e-14:
            break
        f_der = np.mean(probs * (1.0 - probs), axis=1)
        step = np.clip(f_val / np.maximum(1e-14, f_der), -4.0, 4.0)
        mu = mu - step

    return mu


class StreamingSelectiveRouter:
    """Calibrates streaming Stage-2 thresholds on D_cal and evaluates curves on D_test."""

    def __init__(
        self,
        contract: DecisionContract,
        per_instance_query_cost: float = 0.05,
        n_voi_bins: int = 12,
        mean_centered: bool = True,
        quadrature_mode: str = "gauss_hermite",
        n_emp_quantiles: int = 33,
    ) -> None:
        self.contract = contract
        self.query_cost = float(per_instance_query_cost)
        self.n_voi_bins = int(n_voi_bins)
        self.mean_centered = bool(mean_centered)
        self.quadrature_mode = str(quadrature_mode)
        self.n_emp_quantiles = max(5, int(n_emp_quantiles))
        self.bin_edges_: np.ndarray = np.array([0.0, 1.0])
        self.bin_logit_sd_: np.ndarray = np.array([0.5])
        self.bin_emp_quantiles_: np.ndarray = np.zeros((1, self.n_emp_quantiles), dtype=float)
        self.bin_diagnostics_: list[dict[str, float]] = []
        self.p1_cal_: np.ndarray = np.empty(0)
        self.p2_cal_: np.ndarray = np.empty(0)
        self.y_cal_: np.ndarray = np.empty(0, dtype=int)
        self.voi_cal_: np.ndarray = np.empty(0)
        self.voi_raw_cal_: np.ndarray = np.empty(0)

    def fit(
        self, y_cal: np.ndarray, p1_cal: np.ndarray, p2_cal: np.ndarray
    ) -> StreamingSelectiveRouter:
        """Fit conditional Stage-2 log-odds dispersion and streaming quantiles on D_cal."""
        self.y_cal_ = np.asarray(y_cal, dtype=int).ravel()
        self.p1_cal_ = np.clip(np.asarray(p1_cal, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        self.p2_cal_ = np.clip(np.asarray(p2_cal, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)

        delta_logit = logit(self.p2_cal_) - logit(self.p1_cal_)
        q_grid = np.linspace(
            1.0 / (self.n_emp_quantiles + 1),
            self.n_emp_quantiles / (self.n_emp_quantiles + 1),
            self.n_emp_quantiles,
        )
        global_emp_q = (
            np.quantile(delta_logit - float(np.mean(delta_logit)), q_grid)
            if delta_logit.size > 0
            else np.zeros(self.n_emp_quantiles, dtype=float)
        )
        edges = np.unique(np.quantile(self.p1_cal_, np.linspace(0.0, 1.0, self.n_voi_bins + 1)))
        self.bin_diagnostics_ = []
        if edges.size <= 2:
            self.bin_edges_ = np.array([0.0, 1.0])
            sd_g = (
                float(np.std(delta_logit, ddof=1))
                if self.mean_centered and delta_logit.size > 1
                else float(np.sqrt(np.mean(delta_logit**2)))
            )
            self.bin_logit_sd_ = np.array([max(0.02, sd_g)])
            self.bin_emp_quantiles_ = global_emp_q[None, :]
            self.bin_diagnostics_.append(
                {
                    "bin_idx": 0.0,
                    "p1_low": 0.0,
                    "p1_high": 1.0,
                    "n_cal": float(delta_logit.size),
                    "mean_p1": float(np.mean(self.p1_cal_)),
                    "mean_p2": float(np.mean(self.p2_cal_)),
                    "mean_delta_logit": float(np.mean(delta_logit)),
                    "sd_delta_logit": float(self.bin_logit_sd_[0]),
                }
            )
        else:
            self.bin_edges_ = edges
            bins = np.clip(np.digitize(self.p1_cal_, edges[1:-1], right=False), 0, edges.size - 2)
            global_sd = max(
                0.05,
                float(np.std(delta_logit, ddof=1))
                if self.mean_centered and delta_logit.size > 1
                else float(np.std(delta_logit)),
            )
            sds = np.empty(edges.size - 1, dtype=float)
            emp_qs = np.empty((edges.size - 1, self.n_emp_quantiles), dtype=float)
            for b in range(edges.size - 1):
                m = bins == b
                cnt = int(np.sum(m))
                if cnt >= 12:
                    if self.mean_centered:
                        sds[b] = max(0.02, float(np.std(delta_logit[m], ddof=1)))
                    else:
                        sds[b] = max(0.02, float(np.sqrt(np.mean(delta_logit[m] ** 2))))
                    emp_qs[b] = np.quantile(delta_logit[m] - float(np.mean(delta_logit[m])), q_grid)
                else:
                    sds[b] = global_sd
                    emp_qs[b] = global_emp_q
                if cnt > 0:
                    self.bin_diagnostics_.append(
                        {
                            "bin_idx": float(b),
                            "p1_low": float(edges[b]),
                            "p1_high": float(edges[b + 1]),
                            "n_cal": float(cnt),
                            "mean_p1": float(np.mean(self.p1_cal_[m])),
                            "mean_p2": float(np.mean(self.p2_cal_[m])),
                            "mean_delta_logit": float(np.mean(delta_logit[m])),
                            "sd_delta_logit": float(sds[b]),
                        }
                    )
            self.bin_logit_sd_ = sds
            self.bin_emp_quantiles_ = emp_qs

        self.voi_raw_cal_ = self._compute_raw_voi(
            self.p1_cal_, self.contract.cost_fp, self.contract.cost_fn, self.mean_centered
        )
        self.voi_cal_ = np.maximum(0.0, self.voi_raw_cal_)
        return self

    def _lookup_sigma(self, p: np.ndarray) -> np.ndarray:
        if self.bin_edges_.size <= 2:
            return np.full(p.size, self.bin_logit_sd_[0], dtype=float)
        bins = np.clip(
            np.digitize(p, self.bin_edges_[1:-1], right=False), 0, self.bin_edges_.size - 2
        )
        return self.bin_logit_sd_[bins]

    def _lookup_emp_quantiles(self, p: np.ndarray) -> np.ndarray:
        if self.bin_edges_.size <= 2:
            return np.repeat(self.bin_emp_quantiles_[0:1, :], p.size, axis=0)
        bins = np.clip(
            np.digitize(p, self.bin_edges_[1:-1], right=False), 0, self.bin_edges_.size - 2
        )
        return self.bin_emp_quantiles_[bins]

    def _compute_raw_voi(
        self,
        p1: np.ndarray,
        cost_fp: float,
        cost_fn: float,
        mean_centered: bool,
    ) -> np.ndarray:
        p = np.clip(np.asarray(p1, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        lp = logit(p)
        prior_loss = _bayes_action_loss(p, cost_fp, cost_fn)

        if self.quadrature_mode == "empirical_quantile":
            q_deltas = self._lookup_emp_quantiles(p)  # (N, K)
            mu = (
                solve_empirical_quantile_logit_offset(p, q_deltas)
                if mean_centered
                else np.zeros_like(p)
            )
            p_post_mat = expit(lp[:, None] + mu[:, None] + q_deltas)
            post_loss = np.mean(_bayes_action_loss(p_post_mat, cost_fp, cost_fn), axis=1)
            return prior_loss - post_loss

        sig = self._lookup_sigma(p)
        mu = solve_mean_preserving_logit_offset(p, sig) if mean_centered else np.zeros_like(p)
        post_loss = np.zeros(p.size, dtype=float)
        for node, wt in zip(_GH_NODES, _GH_WEIGHTS, strict=True):
            p_post = expit(lp + mu + sig * float(node))
            post_loss += float(wt) * _bayes_action_loss(p_post, cost_fp, cost_fn)
        return prior_loss - post_loss

    def estimate_voi(
        self,
        p1: np.ndarray,
        cost_fp: float | None = None,
        cost_fn: float | None = None,
        mean_centered: bool | None = None,
    ) -> np.ndarray:
        """Compute expected Bayes loss reduction VOI_K(x) via 9-node Gauss-Hermite quadrature."""
        cfp = self.contract.cost_fp if cost_fp is None else float(cost_fp)
        cfn = self.contract.cost_fn if cost_fn is None else float(cost_fn)
        mc = self.mean_centered if mean_centered is None else bool(mean_centered)
        raw = self._compute_raw_voi(p1, cfp, cfn, mean_centered=mc)
        if mc and raw.size > 0 and float(np.min(raw)) < -1e-9:
            raise AssertionError(
                f"Proposition 2 violation: mean-centered VOI had negative value {np.min(raw)}"
            )
        return np.maximum(0.0, raw)

    def _eval_point(
        self,
        policy: str,
        nominal_q: float,
        query_mask_cal: np.ndarray,
        query_mask_test: np.ndarray,
        y_test: np.ndarray,
        p1_test: np.ndarray,
        p2_test: np.ndarray,
    ) -> StreamingBudgetPoint:
        q_cal = np.asarray(query_mask_cal, dtype=bool).ravel()
        q_tst = np.asarray(query_mask_test, dtype=bool).ravel()
        yt = np.asarray(y_test, dtype=int).ravel()

        p_route_cal = np.where(q_cal, self.p2_cal_, self.p1_cal_)
        p_route_tst = np.where(q_tst, p2_test, p1_test)

        ap = float(average_precision_score(yt, p_route_tst))
        p50 = precision_at_target_recall(yt, p_route_tst, target_recall=0.50)
        cal_sum = compute_calibration_summary(yt, p_route_tst)

        tau_star = self.contract.cost_fp / (self.contract.cost_fp + self.contract.cost_fn)
        act_pos = p_route_tst >= tau_star
        fp = float(np.sum((yt == 0) & act_pos))
        fn = float(np.sum((yt == 1) & (~act_pos)))
        n_q = float(np.sum(q_tst))
        n_t = max(1.0, float(yt.size))

        total_cost_1k = (1000.0 / n_t) * (
            self.contract.cost_fp * fp + self.contract.cost_fn * fn + self.query_cost * n_q
        )
        treat_none_cost_1k = (1000.0 / n_t) * self.contract.cost_fn * float(np.sum(yt == 1))
        net_benefit_1k = (treat_none_cost_1k - total_cost_1k) / max(1e-9, self.contract.cost_fn)

        decomp = decompose_routing_flips(
            yt, p1_test, p2_test, q_tst, self.contract, self.query_cost
        )

        target_fdr = max(0.05, min(0.80, 1.0 - self.contract.target_precision))
        cfbh = conformal_fdr_selection(
            p_route_cal, self.y_cal_, p_route_tst, yt, target_fdr=target_fdr
        )

        return StreamingBudgetPoint(
            policy=policy,
            nominal_query_rate=float(nominal_q),
            realized_query_rate=float(n_q / n_t),
            auprc=ap,
            prec_at_50_recall=p50,
            net_benefit_per_1k=float(net_benefit_1k),
            total_cost_per_1k=float(total_cost_1k),
            ece=cal_sum.ece,
            cal_slope=cal_sum.cal_slope,
            cfbh_recall=cfbh.realized_recall,
            cfbh_realized_fdp=cfbh.realized_fdp,
            beneficial_flip_nb_gain_per_1k=decomp.beneficial_flip_nb_gain_per_1k,
            harmful_flip_nb_loss_per_1k=decomp.harmful_flip_nb_loss_per_1k,
            query_fee_nb_drag_per_1k=decomp.query_fee_nb_drag_per_1k,
            unchanged_queried_rate=decomp.unchanged_queried_rate,
            above_tau_unchanged_queried_rate=decomp.above_tau_unchanged_queried_rate,
        )

    def evaluate_contract_sensitivity(
        self,
        y_test: np.ndarray,
        p1_test: np.ndarray,
        p2_test: np.ndarray,
        cost_fn_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
        query_cost_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
        seed: int = 42,
    ) -> dict[str, object]:
        """Evaluate Stage-B robustness across Bayes cost ratios tau* and query fees c_query."""
        yt = np.asarray(y_test, dtype=int).ravel()
        p1_t = np.clip(np.asarray(p1_test, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        p2_t = np.clip(np.asarray(p2_test, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        n_t = max(1.0, float(yt.size))
        rng = np.random.default_rng(seed + 77)
        u_cal = rng.uniform(0.0, 1.0, self.p1_cal_.size)
        u_tst = rng.uniform(0.0, 1.0, p1_t.size)

        def _nb(
            p_arr: np.ndarray, q_mask: np.ndarray, c_fp: float, c_fn: float, c_q: float
        ) -> float:
            tau = c_fp / (c_fp + c_fn)
            act = p_arr >= tau
            tp = float(np.sum((yt == 1) & act))
            fp = float(np.sum((yt == 0) & act))
            nq = float(np.sum(q_mask))
            return (1000.0 / n_t) * (tp - (c_fp / c_fn) * fp - (c_q / c_fn) * nq)

        # 1. Cost-ratio (tau*) sweep at baseline c_query
        tau_sweep: list[dict[str, float]] = []
        z_mask = np.zeros(yt.size, dtype=bool)
        o_mask = np.ones(yt.size, dtype=bool)
        _, rsk20_tst, _, _ = calibrated_randomized_threshold_masks(
            self.p1_cal_, p1_t, 0.20, u_cal, u_tst
        )
        for mult in cost_fn_multipliers:
            cfn_m = self.contract.cost_fn * float(mult)
            cfp_m = self.contract.cost_fp
            tau_m = cfp_m / (cfp_m + cfn_m)
            voi_c_m = self.estimate_voi(
                self.p1_cal_, cost_fp=cfp_m, cost_fn=cfn_m, mean_centered=True
            )
            voi_t_m = self.estimate_voi(p1_t, cost_fp=cfp_m, cost_fn=cfn_m, mean_centered=True)
            _, voi20_tst, _, _ = calibrated_randomized_threshold_masks(
                voi_c_m, voi_t_m, 0.20, u_cal, u_tst
            )
            auto_tst_m = voi_t_m > self.query_cost
            tau_sweep.append(
                {
                    "cost_fn_multiplier": float(mult),
                    "tau_star": float(tau_m),
                    "nb_stage1_0pct": _nb(p1_t, z_mask, cfp_m, cfn_m, self.query_cost),
                    "nb_top_risk_20pct": _nb(
                        np.where(rsk20_tst, p2_t, p1_t), rsk20_tst, cfp_m, cfn_m, self.query_cost
                    ),
                    "nb_voi_20pct": _nb(
                        np.where(voi20_tst, p2_t, p1_t), voi20_tst, cfp_m, cfn_m, self.query_cost
                    ),
                    "nb_voi_autonomous": _nb(
                        np.where(auto_tst_m, p2_t, p1_t), auto_tst_m, cfp_m, cfn_m, self.query_cost
                    ),
                    "auto_realized_q": float(np.mean(auto_tst_m)),
                    "nb_full_100pct": _nb(p2_t, o_mask, cfp_m, cfn_m, self.query_cost),
                }
            )

        # 2. Query-cost (c_query) sweep at baseline contract (C_FP, C_FN)
        voi_c_base = self.estimate_voi(self.p1_cal_, mean_centered=True)
        voi_t_base = self.estimate_voi(p1_t, mean_centered=True)
        _, voi20_base, _, _ = calibrated_randomized_threshold_masks(
            voi_c_base, voi_t_base, 0.20, u_cal, u_tst
        )
        fee_sweep: list[dict[str, float]] = []
        for q_mult in query_cost_multipliers:
            cq = self.query_cost * float(q_mult)
            auto_q_m = voi_t_base > cq
            fee_sweep.append(
                {
                    "query_cost_multiplier": float(q_mult),
                    "c_query": float(cq),
                    "nb_stage1_0pct": _nb(
                        p1_t, z_mask, self.contract.cost_fp, self.contract.cost_fn, cq
                    ),
                    "nb_voi_20pct": _nb(
                        np.where(voi20_base, p2_t, p1_t),
                        voi20_base,
                        self.contract.cost_fp,
                        self.contract.cost_fn,
                        cq,
                    ),
                    "nb_voi_autonomous": _nb(
                        np.where(auto_q_m, p2_t, p1_t),
                        auto_q_m,
                        self.contract.cost_fp,
                        self.contract.cost_fn,
                        cq,
                    ),
                    "auto_realized_q": float(np.mean(auto_q_m)),
                    "nb_full_100pct": _nb(
                        p2_t, o_mask, self.contract.cost_fp, self.contract.cost_fn, cq
                    ),
                }
            )

        return {
            "tau_star_sweep": tau_sweep,
            "query_cost_sweep": fee_sweep,
        }

    def evaluate_budget_frontier(
        self,
        y_test: np.ndarray,
        p1_test: np.ndarray,
        p2_test: np.ndarray,
        query_rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0),
        seed: int = 42,
    ) -> dict[str, object]:
        """Evaluate streaming routing policies across ``query_rates`` + E0/E1/E2 diagnostics."""
        yt = np.asarray(y_test, dtype=int).ravel()
        p1_t = np.clip(np.asarray(p1_test, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        p2_t = np.clip(np.asarray(p2_test, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)

        tau_star = self.contract.cost_fp / (self.contract.cost_fp + self.contract.cost_fn)
        voi_t = self.estimate_voi(p1_t, mean_centered=self.mean_centered)

        # Also compute uncentered heuristic VOI on identical splits for Track-0 E0/E1 audit
        voi_uncentered_cal = self.estimate_voi(self.p1_cal_, mean_centered=False)
        voi_uncentered_tst = self.estimate_voi(p1_t, mean_centered=False)
        raw_uncentered_cal = self._compute_raw_voi(
            self.p1_cal_, self.contract.cost_fp, self.contract.cost_fn, mean_centered=False
        )
        raw_centered_cal = self._compute_raw_voi(
            self.p1_cal_, self.contract.cost_fp, self.contract.cost_fn, mean_centered=True
        )

        unc_cal = -np.abs(self.p1_cal_ - tau_star)
        unc_t = -np.abs(p1_t - tau_star)

        rng = np.random.default_rng(seed)
        u_cal = rng.uniform(0.0, 1.0, self.p1_cal_.size)
        u_tst = rng.uniform(0.0, 1.0, p1_t.size)

        neg_p1 = self.p1_cal_[self.y_cal_ == 0]
        pos_s1 = 1.0 - self.p1_cal_[self.y_cal_ == 1]

        curves: dict[str, list[dict[str, float | str]]] = {
            "streaming_voi": [],
            "streaming_voi_uncentered": [],
            "streaming_top_risk": [],
            "streaming_uncertainty": [],
            "mondrian_ambiguity": [],
            "streaming_random": [],
        }

        for q in query_rates:
            if q <= 0.0:
                z_cal = np.zeros(self.p1_cal_.size, dtype=bool)
                z_tst = np.zeros(p1_t.size, dtype=bool)
                pt0 = self._eval_point("stage1_only", 0.0, z_cal, z_tst, yt, p1_t, p2_t)
                for k in curves:
                    curves[k].append({**asdict(pt0), "policy": k})
                continue
            if q >= 1.0:
                o_cal = np.ones(self.p1_cal_.size, dtype=bool)
                o_tst = np.ones(p1_t.size, dtype=bool)
                pt1 = self._eval_point("full_stage2", 1.0, o_cal, o_tst, yt, p1_t, p2_t)
                for k in curves:
                    curves[k].append({**asdict(pt1), "policy": k})
                continue

            # 1. Mean-Centered Coherent Streaming VOI (Proposition 2 + calibration-exact tie-break)
            m_voi_c, m_voi_t, _, _ = calibrated_randomized_threshold_masks(
                self.voi_cal_, voi_t, q, u_cal, u_tst
            )
            pt_voi = self._eval_point("streaming_voi", q, m_voi_c, m_voi_t, yt, p1_t, p2_t)
            curves["streaming_voi"].append(asdict(pt_voi))

            # 1b. Uncentered Heuristic VOI with randomized tie-break (E1 head-to-head)
            m_uncv_c, m_uncv_t, _, _ = calibrated_randomized_threshold_masks(
                voi_uncentered_cal, voi_uncentered_tst, q, u_cal, u_tst
            )
            pt_uncv = self._eval_point(
                "streaming_voi_uncentered", q, m_uncv_c, m_uncv_t, yt, p1_t, p2_t
            )
            curves["streaming_voi_uncentered"].append(asdict(pt_uncv))

            # 2. Streaming Top-Risk threshold calibrated on D_cal
            m_rsk_c, m_rsk_t, _, _ = calibrated_randomized_threshold_masks(
                self.p1_cal_, p1_t, q, u_cal, u_tst
            )
            pt_rsk = self._eval_point("streaming_top_risk", q, m_rsk_c, m_rsk_t, yt, p1_t, p2_t)
            curves["streaming_top_risk"].append(asdict(pt_rsk))

            # 3. Streaming Decision-Boundary Uncertainty threshold calibrated on D_cal
            m_unc_c, m_unc_t, _, _ = calibrated_randomized_threshold_masks(
                unc_cal, unc_t, q, u_cal, u_tst
            )
            pt_unc = self._eval_point("streaming_uncertainty", q, m_unc_c, m_unc_t, yt, p1_t, p2_t)
            curves["streaming_uncertainty"].append(asdict(pt_unc))

            # 4. Mondrian Conformal Ambiguity {0,1} matched to streaming calibration query rate q
            best_alpha = 0.10
            best_gap = 10.0
            for a_cand in np.linspace(0.01, 0.45, 45):
                qn = float(np.quantile(neg_p1, 1.0 - a_cand)) if neg_p1.size > 0 else 1.0
                qp = float(np.quantile(pos_s1, 1.0 - a_cand)) if pos_s1.size > 0 else 1.0
                amb_c = (self.p1_cal_ <= qn) & ((1.0 - self.p1_cal_) <= qp)
                gap = abs(float(np.mean(amb_c)) - q)
                if gap < best_gap:
                    best_gap = gap
                    best_alpha = float(a_cand)
            qn_b = float(np.quantile(neg_p1, 1.0 - best_alpha)) if neg_p1.size > 0 else 1.0
            qp_b = float(np.quantile(pos_s1, 1.0 - best_alpha)) if pos_s1.size > 0 else 1.0
            amb_cal = (self.p1_cal_ <= qn_b) & ((1.0 - self.p1_cal_) <= qp_b)
            amb_tst = (p1_t <= qn_b) & ((1.0 - p1_t) <= qp_b)
            if float(np.mean(amb_cal)) < q - 0.01:
                rem_q = q - float(np.mean(amb_cal))
                only0_cal = (self.p1_cal_ <= qn_b) & (~amb_cal)
                if np.any(only0_cal):
                    pad_frac = min(0.99, rem_q / max(1e-6, float(np.mean(only0_cal))))
                    pad_c_m, pad_t_m, _, _ = calibrated_randomized_threshold_masks(
                        self.p1_cal_[only0_cal], p1_t, pad_frac, u_cal[only0_cal], u_tst
                    )
                    amb_cal_full = amb_cal.copy()
                    amb_cal_full[only0_cal] = pad_c_m
                    amb_cal = amb_cal | amb_cal_full
                    only0_tst = (p1_t <= qn_b) & (~amb_tst)
                    amb_tst = amb_tst | (only0_tst & pad_t_m)
            pt_amb = self._eval_point("mondrian_ambiguity", q, amb_cal, amb_tst, yt, p1_t, p2_t)
            curves["mondrian_ambiguity"].append(asdict(pt_amb))

            # 5. Streaming Random Bernoulli(q)
            pt_rnd = self._eval_point("streaming_random", q, u_cal < q, u_tst < q, yt, p1_t, p2_t)
            curves["streaming_random"].append(asdict(pt_rnd))

        # Autonomous decision-theoretic VOI rule: query iff VOI_K(x) > query_cost
        auto_cal = self.voi_cal_ > self.query_cost
        auto_tst = voi_t > self.query_cost
        pt_auto = self._eval_point(
            "voi_autonomous",
            float(np.mean(auto_cal)),
            auto_cal,
            auto_tst,
            yt,
            p1_t,
            p2_t,
        )

        # Uncentered autonomous rule for E1 comparison
        auto_unc_cal = voi_uncentered_cal > self.query_cost
        auto_unc_tst = voi_uncentered_tst > self.query_cost
        pt_auto_unc = self._eval_point(
            "voi_uncentered_autonomous",
            float(np.mean(auto_unc_cal)),
            auto_unc_cal,
            auto_unc_tst,
            yt,
            p1_t,
            p2_t,
        )

        # E0/E1 Router Agreement & Tie Diagnostic at q = 0.20
        m_v20_c, m_v20_t, gamma_v20_centered, rho_v20_centered = (
            calibrated_randomized_threshold_masks(self.voi_cal_, voi_t, 0.20, u_cal, u_tst)
        )
        _, m_uv20_t, gamma_v20_uncentered, _ = calibrated_randomized_threshold_masks(
            voi_uncentered_cal, voi_uncentered_tst, 0.20, u_cal, u_tst
        )
        legacy_nontiebroken_v20_t = voi_uncentered_tst >= gamma_v20_uncentered
        jaccard_20 = float(np.sum(m_v20_t & m_uv20_t)) / max(1.0, float(np.sum(m_v20_t | m_uv20_t)))
        disagree_20 = float(np.mean(m_v20_t != m_uv20_t))

        e0_e1_diagnostics = {
            "uncentered_raw_voi_neg_frac_cal": float(np.mean(raw_uncentered_cal < 0.0)),
            "centered_raw_voi_neg_frac_cal": float(np.mean(raw_centered_cal < -1e-12)),
            "uncentered_clipped_zero_frac_cal": float(np.mean(voi_uncentered_cal <= 0.0)),
            "centered_clipped_zero_frac_cal": float(np.mean(self.voi_cal_ <= 0.0)),
            "legacy_nontiebroken_q20_realized_test": float(np.mean(legacy_nontiebroken_v20_t)),
            "centered_tiebroken_q20_realized_test": float(np.mean(m_v20_t)),
            "uncentered_tiebroken_q20_realized_test": float(np.mean(m_uv20_t)),
            "centered_vs_uncentered_q20_disagreement": disagree_20,
            "centered_vs_uncentered_q20_jaccard": jaccard_20,
            "gamma_q20_centered": gamma_v20_centered,
            "rho_tie_q20_centered": rho_v20_centered,
            "gamma_q20_uncentered": gamma_v20_uncentered,
        }

        # Conformal FDR Selection (Jin & Candes, 2023) vs unverified Mondrian {1} singleton
        target_fdr = max(0.05, min(0.80, 1.0 - self.contract.target_precision))
        cfbh_s1 = conformal_fdr_selection(
            self.p1_cal_, self.y_cal_, p1_t, yt, target_fdr=target_fdr
        )
        p_cal_v20 = np.where(m_v20_c, self.p2_cal_, self.p1_cal_)
        p_tst_v20 = np.where(m_v20_t, p2_t, p1_t)
        cfbh_voi20 = conformal_fdr_selection(
            p_cal_v20, self.y_cal_, p_tst_v20, yt, target_fdr=target_fdr
        )
        p_cal_auto = np.where(auto_cal, self.p2_cal_, self.p1_cal_)
        p_tst_auto = np.where(auto_tst, p2_t, p1_t)
        cfbh_auto = conformal_fdr_selection(
            p_cal_auto, self.y_cal_, p_tst_auto, yt, target_fdr=target_fdr
        )
        cfbh_s2 = conformal_fdr_selection(
            self.p2_cal_, self.y_cal_, p2_t, yt, target_fdr=target_fdr
        )

        qn20 = float(np.quantile(neg_p1, 0.80)) if neg_p1.size > 0 else 1.0
        qp20 = float(np.quantile(pos_s1, 0.80)) if pos_s1.size > 0 else 1.0
        sing1_tst = (p1_t > qn20) & ((1.0 - p1_t) <= qp20)
        n_sing1 = int(np.sum(sing1_tst))
        tp_s1 = int(np.sum((yt == 1) & sing1_tst))
        mondrian_sing1_fdp = float(n_sing1 - tp_s1) / float(n_sing1) if n_sing1 > 0 else 0.0
        mondrian_sing1_rec = float(tp_s1) / max(1.0, float(np.sum(yt == 1)))

        return {
            "query_rates": list(query_rates),
            "curves": curves,
            "voi_autonomous": asdict(pt_auto),
            "voi_uncentered_autonomous": asdict(pt_auto_unc),
            "e0_e1_diagnostics": e0_e1_diagnostics,
            "bin_diagnostics": self.bin_diagnostics_,
            "contract_sensitivity": self.evaluate_contract_sensitivity(yt, p1_t, p2_t, seed=seed),
            "conformal_fdr_comparison": {
                "target_fdr": target_fdr,
                "target_ppv": 1.0 - target_fdr,
                "mondrian_singleton1_alpha20_fdp": mondrian_sing1_fdp,
                "mondrian_singleton1_alpha20_recall": mondrian_sing1_rec,
                "cfbh_stage1": asdict(cfbh_s1),
                "cfbh_streaming_voi_20pct": asdict(cfbh_voi20),
                "cfbh_voi_autonomous": asdict(cfbh_auto),
                "cfbh_full_stage2": asdict(cfbh_s2),
            },
            "stage1_calibration": asdict(compute_calibration_summary(yt, p1_t)),
            "stage2_calibration": asdict(compute_calibration_summary(yt, p2_t)),
        }


__all__ = [
    "CalibrationSummary",
    "ConformalFDRResult",
    "FlipDecomposition",
    "StreamingBudgetPoint",
    "StreamingSelectiveRouter",
    "calibrated_randomized_threshold_masks",
    "compute_calibration_summary",
    "compute_smooth_ici",
    "conformal_fdr_selection",
    "decompose_routing_flips",
    "precision_at_target_recall",
    "solve_empirical_quantile_logit_offset",
    "solve_mean_preserving_logit_offset",
    "weighted_conformal_fdr_selection",
]
