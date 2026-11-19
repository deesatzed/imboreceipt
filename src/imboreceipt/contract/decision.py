"""Stage C: Decision contracts, base-rate evidence deficit, and MDE thermometer sizing."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import logit
from scipy.stats import norm


class DecisionContract(BaseModel):
    """Immutable specification of operational rare-event decision constraints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_name: str
    target_precision: float = Field(default=0.30, gt=0.0, lt=1.0)
    max_fpr_budget: float = Field(default=0.01, gt=0.0, lt=1.0)
    cost_fp: float = Field(default=1.0, gt=0.0)
    cost_fn: float = Field(default=25.0, gt=0.0)
    graded_action_cost_fp: float | None = Field(default=0.25, gt=0.0)
    graded_action_efficacy: float = Field(default=0.75, gt=0.0, le=1.0)
    min_recall_resolution: float = Field(default=0.05, gt=0.0, lt=0.5)


class GateCAuditResult(BaseModel):
    """Pre-modeling feasibility and statistical resolution audit (Gate C)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prevalence: float
    target_precision: float
    required_lr: float
    required_log_lr_nats: float
    required_specificity_at_tpr80: float
    achieved_ppv_at_tpr80_fpr01: float
    bayes_optimal_threshold: float
    graded_action_threshold: float | None
    n_pos_test: int
    recall_ci_half_width_at_80: float
    required_n_pos_for_resolution: int
    thermometer_adequate: bool
    one_stage_plausible: bool
    gate_recommendation: str


def required_likelihood_ratio(prevalence: float, target_precision: float) -> float:
    """Compute required likelihood ratio ``(q/(1-q)) / (pi/(1-pi))``."""
    if not (0.0 < prevalence < 1.0):
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    if not (0.0 < target_precision < 1.0):
        raise ValueError(f"target_precision must be in (0, 1), got {target_precision}")
    prior_odds = prevalence / (1.0 - prevalence)
    target_odds = target_precision / (1.0 - target_precision)
    return float(target_odds / prior_odds)


def achieved_ppv(prevalence: float, tpr: float, fpr: float) -> float:
    """Compute Positive Predictive Value (precision) from prevalence, TPR, and FPR."""
    num = prevalence * tpr
    den = num + (1.0 - prevalence) * fpr
    if den <= 0.0:
        return 0.0
    return float(num / den)


def required_specificity(prevalence: float, tpr: float, target_precision: float) -> float:
    """Compute specificity ``1 - FPR`` needed to reach ``target_precision`` at ``tpr``."""
    req_lr = required_likelihood_ratio(prevalence, target_precision)
    max_fpr = tpr / req_lr
    return float(np.clip(1.0 - max_fpr, 0.0, 1.0))


def log_likelihood_ratio_from_probs(
    probs: np.ndarray,
    prevalence: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert calibrated posterior probabilities ``P(Y=1|X=x)`` to ``log Lambda_X(x)``."""
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    pi = float(np.clip(prevalence, eps, 1.0 - eps))
    return np.asarray(logit(p) - logit(pi), dtype=float)


def instance_evidence_deficit(
    probs: np.ndarray,
    prevalence: float,
    target_precision: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute CREDO Base-Rate Evidence Deficit ``D_q(x)`` in log-odds (nats).

    ``D_q(x) = max(0, logit(q) - logit(pi) - log Lambda_X(x))``
    represents the exact additional conditional log-likelihood ratio a confirmatory
    channel must supply before acting on instance ``x`` is justified at precision ``q``.
    """
    log_lr = log_likelihood_ratio_from_probs(probs, prevalence, eps=eps)
    q = float(np.clip(target_precision, eps, 1.0 - eps))
    pi = float(np.clip(prevalence, eps, 1.0 - eps))
    req_log_lr = float(logit(q) - logit(pi))
    return np.maximum(0.0, req_log_lr - log_lr)


def wilson_interval(
    successes: int,
    total: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute exact Wilson (1927) score confidence interval ``(low, high)`` for a binomial rate."""
    if total <= 0:
        return (0.0, 1.0)
    k = int(np.clip(successes, 0, total))
    n = float(total)
    p_hat = float(k) / n
    z = float(norm.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def wilson_halfwidth(
    successes: int,
    total: int,
    alpha: float = 0.05,
) -> float:
    """Return half-width ``(high - low) / 2`` of the Wilson score interval at observed ``k / n``."""
    lo, hi = wilson_interval(successes, total, alpha=alpha)
    return float(0.5 * (hi - lo))


def minimum_detectable_recall_halfwidth(
    n_pos_test: int,
    reference_recall: float = 0.80,
    alpha: float = 0.05,
) -> float:
    """Illustrative normal-approximation planning half-width at assumed ``reference_recall``."""
    if n_pos_test <= 0:
        return 1.0
    z = float(norm.ppf(1.0 - alpha / 2.0))
    se = math.sqrt(reference_recall * (1.0 - reference_recall) / n_pos_test)
    return float(min(1.0, z * se))


def required_positives_for_resolution(
    half_width: float = 0.05,
    reference_recall: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Compute independent test positives required to resolve recall within ``+-half_width``."""
    z = float(norm.ppf(1.0 - alpha / 2.0))
    n_req = (z * z * reference_recall * (1.0 - reference_recall)) / (half_width * half_width)
    return int(math.ceil(n_req))


def expected_net_utility_per_1k(
    y_true: np.ndarray,
    hard_actions: np.ndarray,
    contract: DecisionContract,
    graded_actions: np.ndarray | None = None,
) -> float:
    """Compute decision utility per 1,000 instances relative to an always-abstain baseline.

    - Catching a true positive with hard action saves ``cost_fn``.
    - False positive with hard action incurs ``-cost_fp``.
    - Smoke-detector graded action (H4, when ``hard_actions == 0`` and ``graded_actions == 1``)
      saves ``graded_action_efficacy * cost_fn`` on TPs while incurring ``-graded_action_cost_fp``
      on FPs.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    act = np.asarray(hard_actions, dtype=int).ravel()
    n = max(1, y.size)

    tp_hard = np.sum((y == 1) & (act == 1))
    fp_hard = np.sum((y == 0) & (act == 1))
    utility = float(tp_hard * contract.cost_fn - fp_hard * contract.cost_fp)

    if graded_actions is not None and contract.graded_action_cost_fp is not None:
        g_act = np.asarray(graded_actions, dtype=int).ravel() & (act == 0)
        tp_graded = np.sum((y == 1) & (g_act == 1))
        fp_graded = np.sum((y == 0) & (g_act == 1))
        utility += float(
            tp_graded * (contract.graded_action_efficacy * contract.cost_fn)
            - fp_graded * contract.graded_action_cost_fp
        )

    return float(1000.0 * utility / n)


def gate_c_audit(
    contract: DecisionContract,
    prevalence: float,
    n_pos_test: int,
) -> GateCAuditResult:
    """Execute Gate C pre-modeling decision & statistical thermometer audit."""
    req_lr = required_likelihood_ratio(prevalence, contract.target_precision)
    req_spec = required_specificity(prevalence, 0.80, contract.target_precision)
    ref_ppv = achieved_ppv(prevalence, 0.80, 0.01)
    p_star = contract.cost_fp / (contract.cost_fp + contract.cost_fn)

    p_graded: float | None = None
    if contract.graded_action_cost_fp is not None:
        eff_benefit = contract.graded_action_efficacy * contract.cost_fn
        p_graded = contract.graded_action_cost_fp / (contract.graded_action_cost_fp + eff_benefit)

    hw = minimum_detectable_recall_halfwidth(n_pos_test, reference_recall=0.80)
    req_n = required_positives_for_resolution(
        half_width=contract.min_recall_resolution, reference_recall=0.80
    )
    therm_ok = n_pos_test >= req_n
    # A single tabular channel rarely achieves LR > 250 cleanly without confirmation
    one_stage_ok = req_lr <= 250.0

    if not one_stage_ok:
        rec = "TWO_STAGE_LATTICE_OR_GRADED_ACTION_REQUIRED"
    elif not therm_ok:
        rec = "PROCEED_WITH_CLUSTER_BOOTSTRAP_THERMOMETER_WARNING"
    else:
        rec = "PROCEED_STANDARD"

    return GateCAuditResult(
        prevalence=float(prevalence),
        target_precision=contract.target_precision,
        required_lr=req_lr,
        required_log_lr_nats=float(math.log(req_lr)),
        required_specificity_at_tpr80=req_spec,
        achieved_ppv_at_tpr80_fpr01=ref_ppv,
        bayes_optimal_threshold=float(p_star),
        graded_action_threshold=p_graded,
        n_pos_test=int(n_pos_test),
        recall_ci_half_width_at_80=hw,
        required_n_pos_for_resolution=req_n,
        thermometer_adequate=therm_ok,
        one_stage_plausible=one_stage_ok,
        gate_recommendation=rec,
    )
