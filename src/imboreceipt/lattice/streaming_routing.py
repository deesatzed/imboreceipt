"""Streaming score-threshold selective feature acquisition and Conformal FDR selection.

Implements:
1. Real-time streaming threshold routing across full query-rate frontiers q in [0, 1]
   calibrated strictly on D_cal (no batch test-set argsort and no hard-zero PR truncation).
2. Decision-contract Bayes Value-of-Information (VOI) loss-reduction routing with an
   autonomous cost-optimal stopping rule VOI_K(x) > c_query (Saar-Tsechansky et al., 2009;
   Shim et al., 2018; Li & Oliva, 2021).
3. Conformal Selection with False Discovery Rate (FDR = 1 - PPV) control via conformal
   p-values and Benjamini-Hochberg (CfBH; Jin & Candes, 2023).
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
    """Probability calibration diagnostics (Van Calster et al., 2019)."""

    ece: float
    brier: float
    cal_intercept: float
    cal_slope: float


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


def compute_calibration_summary(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 15
) -> CalibrationSummary:
    """Compute quantile-binned ECE, Brier score, calibration-in-the-large, and calibration slope."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(prob, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
    brier = float(np.mean((y - p) ** 2))

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
        return CalibrationSummary(ece=float(ece), brier=brier, cal_intercept=0.0, cal_slope=1.0)

    # Calibration-in-the-large: intercept alpha with fixed slope = 1 (offset = lp)
    def _nll_int(a_arr: np.ndarray) -> float:
        eta = a_arr[0] + lp
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

    res_int = minimize(_nll_int, x0=np.array([0.0]), method="BFGS")
    cal_int = float(res_int.x[0]) if res_int.success else 0.0

    # Calibration slope: logistic regression of y on logit(p) without regularization
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
    """Conformal Selection with False Discovery Rate control (CfBH; Jin & Candes, 2023, JMLR).

    Implements Jin & Candes (2023, Eq. 6 & Theorem 2.2) smoothed conformal p-values for
    binary positive selection (V(x, 0) = S(x), V(x, 1) = -inf) with deterministic uniform
    tie-breaking so discrete tree-leaf score ties do not collapse discovery power:
      p_j = [#{i in D_cal : Y_i=0, S_i > S_test,j}
             + U_j * (1 + #{i in D_cal : Y_i=0, S_i == S_test,j})] / (1 + |D_cal|),
    followed by Benjamini-Hochberg step-up selection at level ``target_fdr = 1 - target_ppv``.
    """
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

    # Exact counts of calibration negatives strictly greater than and equal to S_test,j
    idx_left = np.searchsorted(neg_cal, st, side="left")
    idx_right = np.searchsorted(neg_cal, st, side="right")
    n_gt = (m0 - idx_right).astype(float)
    n_eq = (idx_right - idx_left).astype(float)
    u_tie = np.random.default_rng(seed).uniform(0.0, 1.0, size=n_test)
    p_conf = (1.0 + n_gt + u_tie * n_eq) / float(1 + n_cal)

    # Benjamini-Hochberg step-up procedure at level target_fdr
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


def _bayes_action_loss(p: np.ndarray, cost_fp: float, cost_fn: float) -> np.ndarray:
    return np.minimum(p * cost_fn, (1.0 - p) * cost_fp)


class StreamingSelectiveRouter:
    """Calibrates streaming Stage-2 thresholds on D_cal and evaluates curves on D_test."""

    def __init__(
        self,
        contract: DecisionContract,
        per_instance_query_cost: float = 0.05,
        n_voi_bins: int = 12,
    ) -> None:
        self.contract = contract
        self.query_cost = float(per_instance_query_cost)
        self.n_voi_bins = int(n_voi_bins)
        self.bin_edges_: np.ndarray = np.array([0.0, 1.0])
        self.bin_logit_sd_: np.ndarray = np.array([0.5])
        self.p1_cal_: np.ndarray = np.empty(0)
        self.p2_cal_: np.ndarray = np.empty(0)
        self.y_cal_: np.ndarray = np.empty(0, dtype=int)
        self.voi_cal_: np.ndarray = np.empty(0)

    def fit(
        self, y_cal: np.ndarray, p1_cal: np.ndarray, p2_cal: np.ndarray
    ) -> StreamingSelectiveRouter:
        """Fit conditional Stage-2 log-odds dispersion and streaming quantiles on D_cal."""
        self.y_cal_ = np.asarray(y_cal, dtype=int).ravel()
        self.p1_cal_ = np.clip(np.asarray(p1_cal, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        self.p2_cal_ = np.clip(np.asarray(p2_cal, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)

        delta_logit = logit(self.p2_cal_) - logit(self.p1_cal_)
        edges = np.unique(np.quantile(self.p1_cal_, np.linspace(0.0, 1.0, self.n_voi_bins + 1)))
        if edges.size <= 2:
            self.bin_edges_ = np.array([0.0, 1.0])
            self.bin_logit_sd_ = np.array([max(0.05, float(np.std(delta_logit)))])
        else:
            self.bin_edges_ = edges
            bins = np.clip(np.digitize(self.p1_cal_, edges[1:-1], right=False), 0, edges.size - 2)
            global_sd = max(0.05, float(np.std(delta_logit)))
            sds = np.empty(edges.size - 1, dtype=float)
            for b in range(edges.size - 1):
                m = bins == b
                if int(np.sum(m)) >= 12:
                    sds[b] = max(0.02, float(np.sqrt(np.mean(delta_logit[m] ** 2))))
                else:
                    sds[b] = global_sd
            self.bin_logit_sd_ = sds

        self.voi_cal_ = self.estimate_voi(self.p1_cal_)
        return self

    def estimate_voi(self, p1: np.ndarray) -> np.ndarray:
        """Compute expected Bayes loss reduction VOI_K(x) via 9-node Gauss-Hermite quadrature."""
        p = np.clip(np.asarray(p1, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        lp = logit(p)
        if self.bin_edges_.size <= 2:
            sig = np.full(p.size, self.bin_logit_sd_[0], dtype=float)
        else:
            bins = np.clip(
                np.digitize(p, self.bin_edges_[1:-1], right=False), 0, self.bin_edges_.size - 2
            )
            sig = self.bin_logit_sd_[bins]

        prior_loss = _bayes_action_loss(p, self.contract.cost_fp, self.contract.cost_fn)
        post_loss = np.zeros(p.size, dtype=float)
        for node, wt in zip(_GH_NODES, _GH_WEIGHTS, strict=True):
            p_post = expit(lp + sig * float(node))
            post_loss += float(wt) * _bayes_action_loss(
                p_post, self.contract.cost_fp, self.contract.cost_fn
            )
        return np.maximum(0.0, prior_loss - post_loss)

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
        # Vickers-Elkin Net Benefit per 1,000 arrivals net of query cost
        net_benefit_1k = (treat_none_cost_1k - total_cost_1k) / max(1e-9, self.contract.cost_fn)

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
        )

    def evaluate_budget_frontier(
        self,
        y_test: np.ndarray,
        p1_test: np.ndarray,
        p2_test: np.ndarray,
        query_rates: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0),
        seed: int = 42,
    ) -> dict[str, object]:
        """Evaluate streaming routing policies across ``query_rates`` + autonomous VOI & CfBH."""
        yt = np.asarray(y_test, dtype=int).ravel()
        p1_t = np.clip(np.asarray(p1_test, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        p2_t = np.clip(np.asarray(p2_test, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)

        tau_star = self.contract.cost_fp / (self.contract.cost_fp + self.contract.cost_fn)
        voi_t = self.estimate_voi(p1_t)
        unc_cal = -np.abs(self.p1_cal_ - tau_star)
        unc_t = -np.abs(p1_t - tau_star)

        rng = np.random.default_rng(seed)
        u_cal = rng.uniform(0.0, 1.0, self.p1_cal_.size)
        u_tst = rng.uniform(0.0, 1.0, p1_t.size)

        # Mondrian conformal calibration nonconformity quantiles on D_cal
        neg_p1 = self.p1_cal_[self.y_cal_ == 0]
        pos_s1 = 1.0 - self.p1_cal_[self.y_cal_ == 1]

        curves: dict[str, list[dict[str, float | str]]] = {
            "streaming_voi": [],
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

            # 1. Streaming VOI threshold calibrated on D_cal
            th_voi = float(np.quantile(self.voi_cal_, 1.0 - q))
            pt_voi = self._eval_point(
                "streaming_voi",
                q,
                self.voi_cal_ >= th_voi,
                voi_t >= th_voi,
                yt,
                p1_t,
                p2_t,
            )
            curves["streaming_voi"].append(asdict(pt_voi))

            # 2. Streaming Top-Risk threshold calibrated on D_cal
            th_rsk = float(np.quantile(self.p1_cal_, 1.0 - q))
            pt_rsk = self._eval_point(
                "streaming_top_risk",
                q,
                self.p1_cal_ >= th_rsk,
                p1_t >= th_rsk,
                yt,
                p1_t,
                p2_t,
            )
            curves["streaming_top_risk"].append(asdict(pt_rsk))

            # 3. Streaming Decision-Boundary Uncertainty threshold calibrated on D_cal
            th_unc = float(np.quantile(unc_cal, 1.0 - q))
            pt_unc = self._eval_point(
                "streaming_uncertainty",
                q,
                unc_cal >= th_unc,
                unc_t >= th_unc,
                yt,
                p1_t,
                p2_t,
            )
            curves["streaming_uncertainty"].append(asdict(pt_unc))

            # 4. Mondrian Conformal Ambiguity {0,1} matched to streaming calibration query rate q
            # Find symmetric alpha on D_cal whose {0,1} size is closest to q
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
            # If {0,1} is smaller than q on D_cal, pad streamingly from {0} closest to qn_b
            if float(np.mean(amb_cal)) < q - 0.01:
                rem_q = q - float(np.mean(amb_cal))
                only0_cal = (self.p1_cal_ <= qn_b) & (~amb_cal)
                if np.any(only0_cal):
                    pad_frac = min(0.99, rem_q / max(1e-6, float(np.mean(only0_cal))))
                    pad_cut = float(np.quantile(self.p1_cal_[only0_cal], 1.0 - pad_frac))
                    amb_cal = amb_cal | (only0_cal & (self.p1_cal_ >= pad_cut))
                    only0_tst = (p1_t <= qn_b) & (~amb_tst)
                    amb_tst = amb_tst | (only0_tst & (p1_t >= pad_cut))
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

        # Conformal FDR Selection (Jin & Candes, 2023) vs unverified Mondrian {1} singleton
        target_fdr = max(0.05, min(0.80, 1.0 - self.contract.target_precision))
        cfbh_s1 = conformal_fdr_selection(
            self.p1_cal_, self.y_cal_, p1_t, yt, target_fdr=target_fdr
        )
        # Routed VOI at q = 0.20 for clean fixed-rate CfBH comparison
        th_voi20 = float(np.quantile(self.voi_cal_, 0.80))
        p_cal_v20 = np.where(self.voi_cal_ >= th_voi20, self.p2_cal_, self.p1_cal_)
        p_tst_v20 = np.where(voi_t >= th_voi20, p2_t, p1_t)
        cfbh_voi20 = conformal_fdr_selection(
            p_cal_v20, self.y_cal_, p_tst_v20, yt, target_fdr=target_fdr
        )
        cfbh_s2 = conformal_fdr_selection(
            self.p2_cal_, self.y_cal_, p2_t, yt, target_fdr=target_fdr
        )

        # Unverified Mondrian {1} singleton at alpha_0 = 0.20, alpha_1 = 0.20 for direct contrast
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
            "conformal_fdr_comparison": {
                "target_fdr": target_fdr,
                "target_ppv": 1.0 - target_fdr,
                "mondrian_singleton1_alpha20_fdp": mondrian_sing1_fdp,
                "mondrian_singleton1_alpha20_recall": mondrian_sing1_rec,
                "cfbh_stage1": asdict(cfbh_s1),
                "cfbh_streaming_voi_20pct": asdict(cfbh_voi20),
                "cfbh_full_stage2": asdict(cfbh_s2),
            },
            "stage1_calibration": asdict(compute_calibration_summary(yt, p1_t)),
            "stage2_calibration": asdict(compute_calibration_summary(yt, p2_t)),
        }


__all__ = [
    "CalibrationSummary",
    "ConformalFDRResult",
    "StreamingBudgetPoint",
    "StreamingSelectiveRouter",
    "compute_calibration_summary",
    "conformal_fdr_selection",
    "precision_at_target_recall",
]
