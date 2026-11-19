"""Orthogonal OR-of-ANDs confirmation/rescue lattices, time-shift FPR, and conformal foveation."""

from imboreceipt.lattice.coincidence import (
    TailCoincidenceReport,
    estimate_tail_coincidence_surrogate,
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
    StreamingSelectiveRouter,
    compute_calibration_summary,
    conformal_fdr_selection,
    precision_at_target_recall,
)

__all__ = [
    "BinaryConformalFoveator",
    "CalibrationSummary",
    "ConfirmRescueLattice",
    "ConformalFDRResult",
    "ConformalFoveationReport",
    "MechanismBranch",
    "StreamingSelectiveRouter",
    "TailCoincidenceReport",
    "compute_calibration_summary",
    "conformal_fdr_selection",
    "estimate_tail_coincidence_surrogate",
    "precision_at_target_recall",
]
