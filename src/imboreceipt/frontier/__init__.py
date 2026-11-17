"""Frontier evaluation, baseline saturation suites, and 4-ceiling bottleneck diagnosis."""

from imboreceipt.frontier.baselines import (
    FittedDecoderArm,
    PlattLogitCalibrator,
    TabularPreprocessor,
    fit_single_calibrated_gbdt,
    train_decoder_saturation_suite,
)
from imboreceipt.frontier.ceilings import (
    BottleneckRegime,
    CertifiedFrontierMetrics,
    FourCeilingDiagnosis,
    compute_ece,
    diagnose_four_ceilings,
    estimate_finite_positive_learning_curve,
    estimate_observed_channel_ceiling_bound,
    evaluate_certified_frontier,
    select_neyman_pearson_threshold,
    select_precision_threshold,
)

__all__ = [
    "BottleneckRegime",
    "CertifiedFrontierMetrics",
    "FittedDecoderArm",
    "FourCeilingDiagnosis",
    "PlattLogitCalibrator",
    "TabularPreprocessor",
    "compute_ece",
    "diagnose_four_ceilings",
    "estimate_finite_positive_learning_curve",
    "estimate_observed_channel_ceiling_bound",
    "evaluate_certified_frontier",
    "fit_single_calibrated_gbdt",
    "select_neyman_pearson_threshold",
    "select_precision_threshold",
    "train_decoder_saturation_suite",
]
