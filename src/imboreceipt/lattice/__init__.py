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

__all__ = [
    "BinaryConformalFoveator",
    "ConfirmRescueLattice",
    "ConformalFoveationReport",
    "MechanismBranch",
    "TailCoincidenceReport",
    "estimate_tail_coincidence_surrogate",
]
