"""Residual channel discovery, mechanistic compilers, and case-cohort REG selection."""

from imboreceipt.discover.compilers import (
    ConservationMismatchCompiler,
    ConservationRule,
    CriticalSlowingDownCompiler,
    MissingLineSilenceCompiler,
    TypedFeatureDAGCompiler,
)
from imboreceipt.discover.selector import (
    CandidateChannelSpec,
    ChannelScoreRecord,
    benjamini_hochberg_mask,
    score_channels_baseline_strategy,
    score_channels_conditional_gain,
    score_channels_credo_reg,
    select_channels_within_budget,
)
from imboreceipt.discover.strata import (
    CaseCohortPilot,
    ConsensusResidualStrata,
    adaptive_controls_per_positive,
    build_consensus_residual_strata,
    make_leak_free_cv_splits,
    sample_case_cohort_pilot,
)

__all__ = [
    "CandidateChannelSpec",
    "CaseCohortPilot",
    "ChannelScoreRecord",
    "ConsensusResidualStrata",
    "ConservationMismatchCompiler",
    "ConservationRule",
    "CriticalSlowingDownCompiler",
    "MissingLineSilenceCompiler",
    "TypedFeatureDAGCompiler",
    "adaptive_controls_per_positive",
    "benjamini_hochberg_mask",
    "build_consensus_residual_strata",
    "make_leak_free_cv_splits",
    "sample_case_cohort_pilot",
    "score_channels_baseline_strategy",
    "score_channels_conditional_gain",
    "score_channels_credo_reg",
    "select_channels_within_budget",
]
