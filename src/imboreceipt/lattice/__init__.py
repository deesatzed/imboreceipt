"""Orthogonal OR-of-ANDs confirmation/rescue lattices, time-shift FPR, and conformal foveation."""

from imboreceipt.lattice.coincidence import (
    TailCoincidenceReport,
    estimate_tail_coincidence_surrogate,
)
from imboreceipt.lattice.covert_afa import (
    CovertAFAResult,
    CovertGreedyCMIAFA,
    validate_covert_afa_synthetic_oracle,
)
from imboreceipt.lattice.foveation import (
    BinaryConformalFoveator,
    ConformalFoveationReport,
)
from imboreceipt.lattice.or_of_ands import (
    ConfirmRescueLattice,
    MechanismBranch,
)
from imboreceipt.lattice.streaming_routing import (
    CalibrationSummary,
    ConformalFDRResult,
    FlipDecomposition,
    StreamingBudgetPoint,
    StreamingSelectiveRouter,
    calibrated_randomized_threshold_masks,
    compute_calibration_summary,
    compute_smooth_ici,
    conformal_fdr_selection,
    decompose_routing_flips,
    precision_at_target_recall,
    solve_empirical_quantile_logit_offset,
    solve_mean_preserving_logit_offset,
    weighted_conformal_fdr_selection,
)

__all__ = [
    "BinaryConformalFoveator",
    "CalibrationSummary",
    "ConfirmRescueLattice",
    "ConformalFDRResult",
    "ConformalFoveationReport",
    "CovertAFAResult",
    "CovertGreedyCMIAFA",
    "FlipDecomposition",
    "MechanismBranch",
    "StreamingBudgetPoint",
    "StreamingSelectiveRouter",
    "TailCoincidenceReport",
    "calibrated_randomized_threshold_masks",
    "compute_calibration_summary",
    "compute_smooth_ici",
    "conformal_fdr_selection",
    "decompose_routing_flips",
    "estimate_tail_coincidence_surrogate",
    "precision_at_target_recall",
    "solve_empirical_quantile_logit_offset",
    "solve_mean_preserving_logit_offset",
    "validate_covert_afa_synthetic_oracle",
    "weighted_conformal_fdr_selection",
]
