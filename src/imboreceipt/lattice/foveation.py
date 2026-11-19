"""Stage O (Part 3): Mondrian Conformal Prediction & Equal-Budget Selective Stage-2 Routing."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

from imboreceipt.contract.decision import DecisionContract, expected_net_utility_per_1k

_EPS = 1e-6


class RoutingPolicyMetrics(BaseModel):
    """Comprehensive evaluation of a single Stage-2 selective acquisition policy at budget K."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_name: str
    query_rate: float
    auprc: float
    raw_ap_retention: float
    incremental_ap_gain_retention: float
    precision: float
    recall: float
    fpr: float
    net_utility_per_1k: float
    unqueried_positive_fraction: float
    unqueried_missed_positive_fraction: float
    additional_tp_per_100_queries: float


class ConformalFoveationReport(BaseModel):
    """Comparison of 1-stage, 6 equal-budget Stage-2 routing policies, and 100% Stage-2."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha_neg: float
    alpha_pos: float
    stage1_class0_coverage: float
    stage1_class1_coverage: float
    stage1_class1_coverage_gap: float = 0.0
    stage1_singleton_pos_rate: float = 0.0
    stage1_singleton_pos_ppv: float = 0.0
    stage1_passive_abstention_rate: float
    unbudgeted_conformal_01_query_rate: float = 0.0
    unbudgeted_conformal_01_auprc: float = 0.0
    foveated_stage2_query_rate: float
    stage2_sensor_savings_fraction: float
    post_foveation_unresolved_rate: float
    foveated_precision: float
    foveated_recall: float
    foveated_fpr: float
    foveated_unqueried_positive_fraction: float = 0.0
    foveated_unqueried_missed_positive_fraction: float = 0.0
    stage1_only_auprc: float = 0.0
    literal_conformal_01_auprc: float = 0.0
    classical_midband_conformal_auprc: float = 0.0
    top_risk_k_auprc: float = 0.0
    uncertainty_k_auprc: float = 0.0
    voi_k_auprc: float = 0.0
    random_routing_auprc: float = 0.0
    credo_deficit_foveated_auprc: float = 0.0
    full_stage2_auprc: float = 0.0
    raw_ap_retention: float = 0.0
    incremental_ap_gain_retention: float = 0.0
    net_utility_stage1_only_per_1k: float
    net_utility_literal_conformal_01_per_1k: float = 0.0
    net_utility_classical_midband_per_1k: float = 0.0
    net_utility_top_risk_k_per_1k: float = 0.0
    net_utility_uncertainty_k_per_1k: float = 0.0
    net_utility_voi_k_per_1k: float = 0.0
    net_utility_random_k_per_1k: float = 0.0
    net_utility_foveated_net_of_query_cost_per_1k: float
    net_utility_full_stage2_everywhere_per_1k: float
    policies: dict[str, RoutingPolicyMetrics] = {}


def _bayes_min_loss(probs: np.ndarray, contract: DecisionContract) -> np.ndarray:
    """Minimum expected decision loss per instance under calibrated posterior probability ``p``.

    Loss relative to zero-loss oracle:
    - Abstain (a=0): loss ``p * cost_fn``
    - Graded action (a=g, if enabled): loss ``p * (1 - eff) * cost_fn + (1 - p) * cost_fp_g``
    - Hard action (a=1): loss ``(1 - p) * cost_fp``
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    loss_abstain = p * contract.cost_fn
    loss_hard = (1.0 - p) * contract.cost_fp
    best = np.minimum(loss_abstain, loss_hard)
    if contract.graded_action_cost_fp is not None:
        rem_fn = (1.0 - contract.graded_action_efficacy) * contract.cost_fn
        loss_grad = p * rem_fn + (1.0 - p) * contract.graded_action_cost_fp
        best = np.minimum(best, loss_grad)
    return best


def estimate_expected_voi(
    stage1_test_probs: np.ndarray,
    contract: DecisionContract,
    *,
    stage1_val_probs: np.ndarray | None = None,
    stage2_val_probs: np.ndarray | None = None,
    n_bins: int = 5,
) -> np.ndarray:
    """Estimate per-instance expected decision loss reduction V(x) from acquiring Stage 2.

    Uses validation-fold Stage-1 -> Stage-2 log-odds updates stratified by Stage-1 risk quantiles
    (zero test leakage) to evaluate:
        V(x) = min_a E[L(a, Y) | x] - E_{Z | x}[ min_a E[L(a, Y) | x, Z] ].
    """
    p1_tst = np.clip(np.asarray(stage1_test_probs, dtype=float).ravel(), _EPS, 1.0 - _EPS)
    l1_tst = logit(p1_tst)
    prior_loss = _bayes_min_loss(p1_tst, contract)

    if (
        stage1_val_probs is not None
        and stage2_val_probs is not None
        and stage1_val_probs.size >= 20
        and stage1_val_probs.size == stage2_val_probs.size
    ):
        p1_v = np.clip(np.asarray(stage1_val_probs, dtype=float).ravel(), _EPS, 1.0 - _EPS)
        p2_v = np.clip(np.asarray(stage2_val_probs, dtype=float).ravel(), _EPS, 1.0 - _EPS)
        deltas_v = logit(p2_v) - logit(p1_v)
        edges = np.unique(np.quantile(p1_v, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size <= 2:
            bin_v = np.zeros(p1_v.size, dtype=int)
            bin_t = np.zeros(p1_tst.size, dtype=int)
            n_eff = 1
        else:
            bin_v = np.searchsorted(edges[1:-1], p1_v, side="right")
            bin_t = np.searchsorted(edges[1:-1], p1_tst, side="right")
            n_eff = int(edges.size - 1)

        post_loss = np.empty_like(prior_loss)
        rng = np.random.default_rng(42)
        for b in range(n_eff):
            mask_t = bin_t == b
            if not np.any(mask_t):
                continue
            d_b = deltas_v[bin_v == b]
            if d_b.size == 0:
                d_b = deltas_v
            if d_b.size > 64:
                d_b = rng.choice(d_b, size=64, replace=False)
            # Shape: (n_test_in_bin, n_draws)
            p_post = expit(l1_tst[mask_t, None] + d_b[None, :])
            losses_mat = _bayes_min_loss(p_post, contract)
            post_loss[mask_t] = np.mean(losses_mat, axis=1)
    else:
        # Parametric Gauss-Hermite / quantile grid fallback when validation Stage-1 probs absent
        gh_shifts = np.linspace(-2.0, 2.0, 15)
        gh_weights = np.exp(-0.5 * (gh_shifts / 1.1) ** 2)
        gh_weights /= np.sum(gh_weights)
        p_post = expit(l1_tst[:, None] + gh_shifts[None, :])
        losses_mat = _bayes_min_loss(p_post, contract)
        post_loss = losses_mat @ gh_weights

    return np.maximum(0.0, prior_loss - post_loss)


def build_literal_conformal_01_mask(
    p1: np.ndarray,
    in_0: np.ndarray,
    in_1: np.ndarray,
    q_neg: float,
    k_queries: int,
) -> np.ndarray:
    """Build the literal Mondrian conformal ambiguity routing mask at exact budget ``k_queries``.

    Budget-matching specification:
    1. Candidate priority 1: Ambiguous conformal sets ``~(only_1 | only_0)`` (i.e. ``{0, 1}`` or
       ``{}``). If ``|ambiguous| > k_queries``, select the ``k_queries`` records inside
       ``ambiguous`` closest to ``p = 0.50``.
    2. Candidate priority 2: If ``|ambiguous| < k_queries``, query ALL of ``ambiguous`` and pad
       the remaining ``k_queries - |ambiguous|`` slots from negative singletons ``only_0`` closest
       to the upper negative conformal boundary ``q_neg`` (highest ``p1`` inside ``only_0``).
    3. Positive singletons ``only_1`` stay unqueried unless ``|ambiguous| + |only_0| < K``,
       and when ``only_1`` is empty, ZERO top-risk records are skipped artificially.
    """
    n = p1.size
    mask = np.zeros(n, dtype=bool)
    if k_queries <= 0 or n == 0:
        return mask
    k = min(n, int(k_queries))

    only_1 = in_1 & (~in_0)
    only_0 = in_0 & (~in_1)
    amb_idx = np.flatnonzero(~(only_1 | only_0))

    if amb_idx.size >= k:
        # Tie-break inside ambiguous set by proximity to 0.50
        dist_mid = np.abs(p1[amb_idx] - 0.50)
        chosen = amb_idx[np.argsort(dist_mid, kind="mergesort")[:k]]
        mask[chosen] = True
        return mask

    mask[amb_idx] = True
    rem = k - amb_idx.size
    only0_idx = np.flatnonzero(only_0)
    if only0_idx.size >= rem:
        # Pad from {0} closest to q_neg boundary (highest p1 among {0})
        dist_qneg = np.abs(q_neg - p1[only0_idx])
        chosen_pad = only0_idx[np.argsort(dist_qneg, kind="mergesort")[:rem]]
        mask[chosen_pad] = True
        return mask

    mask[only0_idx] = True
    rem -= only0_idx.size
    if rem > 0:
        only1_idx = np.flatnonzero(only_1)
        # Only reached if k_queries exceeds |ambiguous| + |only_0|; pick lowest p1 in {1}
        chosen_last = only1_idx[np.argsort(p1[only1_idx], kind="mergesort")[:rem]]
        mask[chosen_last] = True
    return mask


class BinaryConformalFoveator:
    """Finite-sample class-conditional conformal predictor with selective Stage-2 routing.

    Compares six equal-budget Stage-2 routing rules at identical query count ``K``:
    - ``credo_foveation``: Queries ``(in_1 | ambiguous) & (p1 < max(q_target, 0.85))``.
    - ``literal_conformal_01``: Queries literal ``{0, 1}`` ambiguity region (padded from ``{0}``
      closest to ``q_neg_``, never skipping high-risk records when ``{1}`` is empty).
    - ``top_risk_k``: Queries the ``K`` highest Stage-1 risk scores ``p1``.
    - ``uncertainty_k``: Queries the ``K`` highest Stage-1 binary Shannon entropies ``H(p1)``.
    - ``voi_k``: Queries the ``K`` largest calibrated Value-of-Information gains ``V(x)``.
    - ``random_k``: Queries ``K`` records uniformly at random.
    """

    def __init__(
        self,
        *,
        alpha_neg: float = 0.20,
        alpha_pos: float = 0.20,
        stage2_query_cost_per_case: float = 0.15,
    ) -> None:
        if not (0.0 < alpha_neg < 1.0 and 0.0 < alpha_pos < 1.0):
            raise ValueError("Conformal miscoverage alphas must lie in (0, 1).")
        self.alpha_neg = alpha_neg
        self.alpha_pos = alpha_pos
        self.stage2_query_cost_per_case = stage2_query_cost_per_case
        self.q_neg_: float = 1.0
        self.q_pos_: float = 1.0
        self._fitted = False

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        n = scores.size
        if n == 0:
            return float("inf")
        rank = math.ceil((n + 1) * (1.0 - alpha))
        if rank > n:
            return float("inf")
        sorted_s = np.sort(scores)
        return float(sorted_s[max(0, rank - 1)])

    def fit(self, stage1_cal_probs: np.ndarray, y_cal: np.ndarray) -> BinaryConformalFoveator:
        """Calibrate finite-sample class-conditional nonconformity thresholds on cal fold."""
        p = np.asarray(stage1_cal_probs, dtype=float).ravel()
        y = np.asarray(y_cal, dtype=int).ravel()

        s0 = p[y == 0]
        s1 = 1.0 - p[y == 1]
        self.q_neg_ = self._conformal_quantile(s0, self.alpha_neg)
        self.q_pos_ = self._conformal_quantile(s1, self.alpha_pos)
        self._fitted = True
        return self

    def predict_stage1_sets(self, stage1_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return boolean membership ``(in_class_0, in_class_1)`` for each instance."""
        if not self._fitted:
            raise RuntimeError("BinaryConformalFoveator must be fit before prediction.")
        p = np.asarray(stage1_probs, dtype=float).ravel()
        in_0 = p <= self.q_neg_
        in_1 = (1.0 - p) <= self.q_pos_
        return in_0, in_1

    def evaluate_foveation(
        self,
        y_test: np.ndarray,
        stage1_test_probs: np.ndarray,
        stage2_test_probs: np.ndarray,
        contract: DecisionContract,
        *,
        stage1_val_probs: np.ndarray | None = None,
        stage2_val_probs: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        fixed_query_rate: float | None = None,
    ) -> ConformalFoveationReport:
        """Evaluate Stage-1, 6 equal-budget Stage-2 routing policies, and 100% Stage-2."""
        yt = np.asarray(y_test, dtype=int).ravel()
        p1 = np.asarray(stage1_test_probs, dtype=float).ravel()
        p2 = np.asarray(stage2_test_probs, dtype=float).ravel()
        n = max(1, yt.size)
        has_pos = int(np.sum(yt == 1)) > 0
        pos_tot = max(1, int(np.sum(yt == 1)))
        neg_tot = max(1, int(np.sum(yt == 0)))

        in_0, in_1 = self.predict_stage1_sets(p1)
        only_1 = in_1 & (~in_0)
        only_0 = in_0 & (~in_1)
        ambiguous = ~(only_1 | only_0)

        cov_0 = float(np.mean(in_0[yt == 0])) if np.any(yt == 0) else 1.0
        cov_1 = float(np.mean(in_1[yt == 1])) if np.any(yt == 1) else 1.0
        abstention_rate = float(np.mean(ambiguous))
        singleton_pos_rate = float(np.mean(only_1))
        singleton_pos_ppv = float(np.mean(yt[only_1])) if np.any(only_1) else 0.0

        # Decision thresholds from cost contract (or precision target on validation if stricter)
        p_hard = contract.cost_fp / (contract.cost_fp + contract.cost_fn)
        if stage2_val_probs is not None and y_val is not None:
            from imboreceipt.frontier.ceilings import select_precision_threshold

            tau_q = select_precision_threshold(y_val, stage2_val_probs, contract.target_precision)
            tau_2_hard = min(max(p_hard, 0.15), tau_q) if tau_q <= 1.0 else p_hard
        else:
            tau_2_hard = p_hard

        p_graded = (
            contract.graded_action_cost_fp
            / (contract.graded_action_cost_fp + contract.graded_action_efficacy * contract.cost_fn)
            if contract.graded_action_cost_fp is not None
            else p_hard
        )

        # Stage-1 only baseline (with identical graded-action contract)
        s1_hard = (p1 >= tau_2_hard).astype(int)
        s1_graded = ((s1_hard == 0) & (p1 >= p_graded)).astype(int)
        util_s1 = expected_net_utility_per_1k(yt, s1_hard, contract, s1_graded)
        tp_s1 = int(np.sum((s1_hard == 1) & (yt == 1)))
        ap_s1 = float(average_precision_score(yt, p1)) if has_pos else 0.0

        # Full 100% Stage-2 reference
        full_s2_hard = (p2 >= tau_2_hard).astype(int)
        full_s2_graded = ((full_s2_hard == 0) & (p2 >= p_graded)).astype(int)
        util_full_s2 = float(
            expected_net_utility_per_1k(yt, full_s2_hard, contract, full_s2_graded)
            - 1000.0 * self.stage2_query_cost_per_case
        )
        ap_s2 = float(average_precision_score(yt, p2)) if has_pos else 0.0

        # Unbudgeted literal {0,1} conformal routing (for diagnostic transparency)
        unbud_probs = np.where(ambiguous, p2, p1)
        unbud_ap = float(average_precision_score(yt, unbud_probs)) if has_pos else 0.0

        # 1. CREDO Evidence-Deficit Upper-Tail Foveation determines query budget K
        #    (or respects explicit external budget cap fixed_query_rate when supplied)
        deficit_positive = p1 < max(contract.target_precision, 0.85)
        raw_credo_cand = (in_1 | ambiguous) & deficit_positive
        if not np.any(raw_credo_cand):
            raw_credo_cand = in_1 | ambiguous

        if fixed_query_rate is not None:
            k_queries = max(1, min(yt.size, int(round(float(fixed_query_rate) * yt.size))))
            credo_mask = np.zeros(yt.size, dtype=bool)
            cand_idx = np.flatnonzero(raw_credo_cand)
            if cand_idx.size >= k_queries:
                # Upper-tail priority within deficit-positive conformal candidates
                chosen_c = cand_idx[np.argsort(-p1[cand_idx], kind="mergesort")[:k_queries]]
                credo_mask[chosen_c] = True
            else:
                credo_mask[cand_idx] = True
                rem_c = k_queries - cand_idx.size
                other_idx = np.flatnonzero(~credo_mask)
                chosen_pad = other_idx[np.argsort(-p1[other_idx], kind="mergesort")[:rem_c]]
                credo_mask[chosen_pad] = True
        else:
            credo_mask = raw_credo_cand
            k_queries = int(np.sum(credo_mask))

        query_rate = float(k_queries / n)
        savings_frac = 1.0 - query_rate

        # 2. Literal Conformal {0,1} at identical budget K (no artificial 30% skip floor!)
        lit_conf_mask = build_literal_conformal_01_mask(p1, in_0, in_1, self.q_neg_, k_queries)

        # 3. Highest Baseline Risk Top-K
        top_risk_mask = np.zeros(yt.size, dtype=bool)
        if k_queries > 0:
            top_risk_mask[np.argsort(-p1, kind="mergesort")[:k_queries]] = True

        # 4. Highest Binary Shannon Entropy Uncertainty-K
        p_clip = np.clip(p1, _EPS, 1.0 - _EPS)
        entropy = -(p_clip * np.log(p_clip) + (1.0 - p_clip) * np.log(1.0 - p_clip))
        unc_mask = np.zeros(yt.size, dtype=bool)
        if k_queries > 0:
            unc_mask[np.argsort(-entropy, kind="mergesort")[:k_queries]] = True

        # 5. Expected Decision-Loss Reduction Value-of-Information VOI-K
        voi_scores = estimate_expected_voi(
            p1,
            contract,
            stage1_val_probs=stage1_val_probs,
            stage2_val_probs=stage2_val_probs,
        )
        voi_mask = np.zeros(yt.size, dtype=bool)
        if k_queries > 0:
            # Break ties in VOI by p1 descending
            voi_order = np.lexsort((-p1, -voi_scores))
            voi_mask[voi_order[:k_queries]] = True

        # 6. Equal-Budget Uniform Random-K
        rng_r = np.random.default_rng(42)
        rand_mask = np.zeros(yt.size, dtype=bool)
        if k_queries > 0:
            rand_mask[rng_r.choice(yt.size, size=min(yt.size, k_queries), replace=False)] = True

        def _eval_policy(name: str, q_mask: np.ndarray) -> RoutingPolicyMetrics:
            q_rate = float(np.mean(q_mask))
            q_cnt = int(np.sum(q_mask))
            routed_p = np.where(q_mask, p2, p1)
            hard = (routed_p >= tau_2_hard).astype(int)
            if name == "credo_foveation":
                # Unqueried Mondrian negative singletons {0} are certified negative;
                # unqueried positive singletons {1} (above deficit target) act positively.
                hard[only_0 & (~q_mask)] = 0
                hard[only_1 & (~q_mask)] = 1
            elif name == "literal_conformal_01":
                # Literal conformal {0,1} abstains -> queries {0,1}, acts 0 on {0} and 1 on {1}
                hard[only_0 & (~q_mask)] = 0
                hard[only_1 & (~q_mask)] = 1
            graded = ((hard == 0) & (routed_p >= p_graded)).astype(int)

            q_cost_1k = 1000.0 * q_rate * self.stage2_query_cost_per_case
            net_u = float(expected_net_utility_per_1k(yt, hard, contract, graded) - q_cost_1k)
            ap = float(average_precision_score(yt, routed_p)) if has_pos else 0.0
            raw_ret = float(ap / max(1e-9, ap_s2)) if ap_s2 > 0 else 1.0
            denom_inc = ap_s2 - ap_s1
            inc_ret = float((ap - ap_s1) / denom_inc) if abs(denom_inc) > 1e-5 else 1.0

            tp = int(np.sum((hard == 1) & (yt == 1)))
            fp = int(np.sum((hard == 1) & (yt == 0)))
            unq_pos = float(np.sum((~q_mask) & (yt == 1)) / pos_tot) if has_pos else 0.0
            unq_missed = (
                float(np.sum((~q_mask) & (hard == 0) & (yt == 1)) / pos_tot) if has_pos else 0.0
            )
            add_tp_100 = float(100.0 * (tp - tp_s1) / max(1, q_cnt)) if q_cnt > 0 else 0.0

            return RoutingPolicyMetrics(
                policy_name=name,
                query_rate=q_rate,
                auprc=ap,
                raw_ap_retention=raw_ret,
                incremental_ap_gain_retention=inc_ret,
                precision=float(tp / max(1, tp + fp)),
                recall=float(tp / pos_tot),
                fpr=float(fp / neg_tot),
                net_utility_per_1k=net_u,
                unqueried_positive_fraction=unq_pos,
                unqueried_missed_positive_fraction=unq_missed,
                additional_tp_per_100_queries=add_tp_100,
            )

        policy_map: dict[str, RoutingPolicyMetrics] = {
            "credo_foveation": _eval_policy("credo_foveation", credo_mask),
            "literal_conformal_01": _eval_policy("literal_conformal_01", lit_conf_mask),
            "top_risk_k": _eval_policy("top_risk_k", top_risk_mask),
            "uncertainty_k": _eval_policy("uncertainty_k", unc_mask),
            "voi_k": _eval_policy("voi_k", voi_mask),
            "random_k": _eval_policy("random_k", rand_mask),
        }

        fov_m = policy_map["credo_foveation"]
        lit_m = policy_map["literal_conformal_01"]
        top_m = policy_map["top_risk_k"]
        unc_m = policy_map["uncertainty_k"]
        voi_m = policy_map["voi_k"]
        rnd_m = policy_map["random_k"]

        fov_probs = np.where(credo_mask, p2, p1)
        fov_hard = (fov_probs >= tau_2_hard).astype(int)
        fov_graded = ((fov_hard == 0) & (fov_probs >= p_graded)).astype(int)
        unresolved_after = (
            credo_mask & (fov_hard == 0) & (fov_graded == 0) & (fov_probs > p_graded * 0.5)
        )

        return ConformalFoveationReport(
            alpha_neg=self.alpha_neg,
            alpha_pos=self.alpha_pos,
            stage1_class0_coverage=cov_0,
            stage1_class1_coverage=cov_1,
            stage1_class1_coverage_gap=max(0.0, (1.0 - self.alpha_pos) - cov_1),
            stage1_singleton_pos_rate=singleton_pos_rate,
            stage1_singleton_pos_ppv=singleton_pos_ppv,
            stage1_passive_abstention_rate=abstention_rate,
            unbudgeted_conformal_01_query_rate=abstention_rate,
            unbudgeted_conformal_01_auprc=unbud_ap,
            foveated_stage2_query_rate=query_rate,
            stage2_sensor_savings_fraction=savings_frac,
            post_foveation_unresolved_rate=float(np.sum(unresolved_after) / n),
            foveated_precision=fov_m.precision,
            foveated_recall=fov_m.recall,
            foveated_fpr=fov_m.fpr,
            foveated_unqueried_positive_fraction=fov_m.unqueried_positive_fraction,
            foveated_unqueried_missed_positive_fraction=fov_m.unqueried_missed_positive_fraction,
            stage1_only_auprc=ap_s1,
            literal_conformal_01_auprc=lit_m.auprc,
            classical_midband_conformal_auprc=lit_m.auprc,
            top_risk_k_auprc=top_m.auprc,
            uncertainty_k_auprc=unc_m.auprc,
            voi_k_auprc=voi_m.auprc,
            random_routing_auprc=rnd_m.auprc,
            credo_deficit_foveated_auprc=fov_m.auprc,
            full_stage2_auprc=ap_s2,
            raw_ap_retention=fov_m.raw_ap_retention,
            incremental_ap_gain_retention=fov_m.incremental_ap_gain_retention,
            net_utility_stage1_only_per_1k=util_s1,
            net_utility_literal_conformal_01_per_1k=lit_m.net_utility_per_1k,
            net_utility_classical_midband_per_1k=lit_m.net_utility_per_1k,
            net_utility_top_risk_k_per_1k=top_m.net_utility_per_1k,
            net_utility_uncertainty_k_per_1k=unc_m.net_utility_per_1k,
            net_utility_voi_k_per_1k=voi_m.net_utility_per_1k,
            net_utility_random_k_per_1k=rnd_m.net_utility_per_1k,
            net_utility_foveated_net_of_query_cost_per_1k=fov_m.net_utility_per_1k,
            net_utility_full_stage2_everywhere_per_1k=util_full_s2,
            policies=policy_map,
        )
