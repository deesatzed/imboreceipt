"""Decision contract and base-rate evidence-deficit calculus."""

from imboreceipt.contract.decision import (
    DecisionContract,
    GateCAuditResult,
    achieved_ppv,
    expected_net_utility_per_1k,
    gate_c_audit,
    instance_evidence_deficit,
    log_likelihood_ratio_from_probs,
    minimum_detectable_recall_halfwidth,
    required_likelihood_ratio,
    required_positives_for_resolution,
    required_specificity,
)

__all__ = [
    "DecisionContract",
    "GateCAuditResult",
    "achieved_ppv",
    "expected_net_utility_per_1k",
    "gate_c_audit",
    "instance_evidence_deficit",
    "log_likelihood_ratio_from_probs",
    "minimum_detectable_recall_halfwidth",
    "required_likelihood_ratio",
    "required_positives_for_resolution",
    "required_specificity",
]
