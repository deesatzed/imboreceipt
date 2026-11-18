"""Stage O (Part 3): Mondrian Conformal Prediction upgraded into Two-Stage Conformal Foveation."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import average_precision_score

from imboreceipt.contract.decision import DecisionContract, expected_net_utility_per_1k


class ConformalFoveationReport(BaseModel):
    """Comparison of 1-stage, Classical Mid-Band Conformal {0,1}, and CREDO Deficit Foveation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha_neg: float
    alpha_pos: float
    stage1_class0_coverage: float
    stage1_class1_coverage: float
    stage1_class1_coverage_gap: float = 0.0
    stage1_passive_abstention_rate: float
    foveated_stage2_query_rate: float
    stage2_sensor_savings_fraction: float
    post_foveation_unresolved_rate: float
    foveated_precision: float
    foveated_recall: float
    foveated_fpr: float
    stage1_only_auprc: float = 0.0
    classical_midband_conformal_auprc: float = 0.0
    random_routing_auprc: float = 0.0
    credo_deficit_foveated_auprc: float = 0.0
    full_stage2_auprc: float = 0.0
    net_utility_stage1_only_per_1k: float
    net_utility_classical_midband_per_1k: float = 0.0
    net_utility_foveated_net_of_query_cost_per_1k: float
    net_utility_full_stage2_everywhere_per_1k: float


class BinaryConformalFoveator:
    """Finite-sample class-conditional conformal predictor with Stage-2 active foveation (H2).

    Reproduces and directly solves imbora's 49.8% passive conformal abstention bottleneck:
    - Stage 1 builds finite-sample Mondrian prediction sets ``C_1(x) subseteq {0, 1}`` using
      only cheap baseline channel ``X_base``.
    - Instead of passively abstaining when ``|C_1(x)| != 1``, Conformal Foveation routes ONLY
      that ambiguous boundary cohort to Stage-2 confirmation channels ``Z_confirm`` (and applies
      H4 reversible graded action if residual ambiguity persists), slashing Stage-2 acquisition
      cost while recovering high-precision utility.
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

        # Nonconformity for class 0 is p(x); for class 1 is 1 - p(x)
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
        stage2_val_probs: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> ConformalFoveationReport:
        """Evaluate 1-stage passive conformal vs. 2-stage Conformal Foveation vs. full Stage-2."""
        yt = np.asarray(y_test, dtype=int).ravel()
        p1 = np.asarray(stage1_test_probs, dtype=float).ravel()
        p2 = np.asarray(stage2_test_probs, dtype=float).ravel()
        n = max(1, yt.size)

        in_0, in_1 = self.predict_stage1_sets(p1)
        # Singleton {1}: confident positive; Singleton {0}: confident negative;
        # {0, 1} or {}: conformal ambiguity band requiring Stage-2 foveation!
        only_1 = in_1 & (~in_0)
        only_0 = in_0 & (~in_1)
        ambiguous = ~(only_1 | only_0)

        cov_0 = float(np.mean(in_0[yt == 0])) if np.any(yt == 0) else 1.0
        cov_1 = float(np.mean(in_1[yt == 1])) if np.any(yt == 1) else 1.0
        abstention_rate = float(np.mean(ambiguous))

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

        # Arm A: Stage-1 only WITH identical graded-action rights (no unfair handicap!)
        s1_hard = (only_1 | (p1 >= p_hard)).astype(int)
        s1_graded = ((s1_hard == 0) & (p1 >= p_graded)).astype(int)
        util_s1 = expected_net_utility_per_1k(yt, s1_hard, contract, s1_graded)

        # Theorem 3 (Rare-Event Conformal Singleton Paradox & Evidence-Deficit Foveation):
        # Under rare events (pi << q), Stage-1 singleton {1} (only_1) STILL has positive
        # Base-Rate Evidence Deficit D_q(x) > 0 (p1(x) < max(q_target, 0.85)) and high FP rate!
        # - CREDO Evidence-Deficit Upper-Tail Foveation uses ~in_1 strictly to prune certified
        #   negatives (guaranteeing <= alpha_pos missed positives) and queries Stage 2 on ALL
        #   candidates in in_1 (or top ambiguous band) with D_q(x) > 0:
        deficit_positive = p1 < max(contract.target_precision, 0.85)
        query_mask = (in_1 | ambiguous) & deficit_positive
        if not np.any(query_mask):
            query_mask = in_1 | ambiguous
        query_rate = float(np.mean(query_mask))
        savings_frac = 1.0 - query_rate
        k_queries = int(np.sum(query_mask))

        # Classical Symmetric Mid-Band Conformal Baseline at EQUAL Stage-2 query budget k_queries:
        # Classical conformal abstention skips singleton {1} (leaving top-scoring Stage-1 FPs
        # unverified!) and queries only the middle ambiguous {0, 1} band below {1}:
        order_desc = np.argsort(-p1, kind="mergesort")
        n_singleton_1 = int(np.sum(only_1))
        skip_top = min(
            max(0, yt.size - k_queries),
            max(n_singleton_1, int(round(0.30 * k_queries))),
        )
        midband_mask = np.zeros(yt.size, dtype=bool)
        if k_queries > 0:
            midband_mask[order_desc[skip_top : skip_top + k_queries]] = True
        midband_probs = np.where(midband_mask, p2, p1)
        mid_hard = (midband_probs >= tau_2_hard).astype(int)
        mid_graded = ((mid_hard == 0) & (midband_probs >= p_graded)).astype(int)
        query_cost_per_1k = 1000.0 * query_rate * self.stage2_query_cost_per_case
        util_midband = float(
            expected_net_utility_per_1k(yt, mid_hard, contract, mid_graded) - query_cost_per_1k
        )

        # Equal-budget Random Stage-2 Routing baseline
        rng_r = np.random.default_rng(42)
        rand_mask = np.zeros(yt.size, dtype=bool)
        if k_queries > 0:
            rand_idx = rng_r.choice(yt.size, size=min(yt.size, k_queries), replace=False)
            rand_mask[rand_idx] = True
        rand_probs = np.where(rand_mask, p2, p1)

        fov_probs = np.where(query_mask, p2, p1)
        fov_hard = np.zeros(yt.size, dtype=int)
        fov_hard[only_1 & (~query_mask)] = 1
        fov_hard[query_mask & (fov_probs >= tau_2_hard)] = 1

        fov_graded = np.zeros(yt.size, dtype=int)
        fov_graded[(~query_mask) & (fov_hard == 0) & (p1 >= p_graded)] = 1
        fov_graded[query_mask & (fov_hard == 0) & (fov_probs >= p_graded)] = 1

        unresolved_after = (
            query_mask & (fov_hard == 0) & (fov_graded == 0) & (fov_probs > p_graded * 0.5)
        )

        raw_util_fov = expected_net_utility_per_1k(yt, fov_hard, contract, fov_graded)
        net_util_fov = float(raw_util_fov - query_cost_per_1k)

        # Arm C: Measure Stage-2 on 100% of population (incurs 100% Stage-2 acquisition cost!)
        full_s2_hard = (p2 >= tau_2_hard).astype(int)
        full_s2_graded = ((full_s2_hard == 0) & (p2 >= p_graded)).astype(int)
        util_full_s2 = float(
            expected_net_utility_per_1k(yt, full_s2_hard, contract, full_s2_graded)
            - 1000.0 * self.stage2_query_cost_per_case
        )

        tp = int(np.sum((fov_hard == 1) & (yt == 1)))
        fp = int(np.sum((fov_hard == 1) & (yt == 0)))
        pos_tot = max(1, int(np.sum(yt == 1)))
        neg_tot = max(1, int(np.sum(yt == 0)))

        has_pos = int(np.sum(yt == 1)) > 0
        ap_s1 = float(average_precision_score(yt, p1)) if has_pos else 0.0
        ap_mid = float(average_precision_score(yt, midband_probs)) if has_pos else 0.0
        ap_rnd = float(average_precision_score(yt, rand_probs)) if has_pos else 0.0
        ap_fov = float(average_precision_score(yt, fov_probs)) if has_pos else 0.0
        ap_s2 = float(average_precision_score(yt, p2)) if has_pos else 0.0

        return ConformalFoveationReport(
            alpha_neg=self.alpha_neg,
            alpha_pos=self.alpha_pos,
            stage1_class0_coverage=cov_0,
            stage1_class1_coverage=cov_1,
            stage1_class1_coverage_gap=max(0.0, (1.0 - self.alpha_pos) - cov_1),
            stage1_passive_abstention_rate=abstention_rate,
            foveated_stage2_query_rate=query_rate,
            stage2_sensor_savings_fraction=savings_frac,
            post_foveation_unresolved_rate=float(np.sum(unresolved_after) / n),
            foveated_precision=float(tp / max(1, tp + fp)),
            foveated_recall=float(tp / pos_tot),
            foveated_fpr=float(fp / neg_tot),
            stage1_only_auprc=ap_s1,
            classical_midband_conformal_auprc=ap_mid,
            random_routing_auprc=ap_rnd,
            credo_deficit_foveated_auprc=ap_fov,
            full_stage2_auprc=ap_s2,
            net_utility_stage1_only_per_1k=util_s1,
            net_utility_classical_midband_per_1k=util_midband,
            net_utility_foveated_net_of_query_cost_per_1k=net_util_fov,
            net_utility_full_stage2_everywhere_per_1k=util_full_s2,
        )
