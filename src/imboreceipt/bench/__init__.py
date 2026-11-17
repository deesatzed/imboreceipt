"""Benchmarking suite: Gate 0.5 Sandbox, Tier-1 OpenML Saturation, and Tier-2 CREDO-Bench."""

from imboreceipt.bench.datasets import (
    TIER1_PUBLIC_SPECS,
    Tier1DatasetSpec,
    load_tier1_dataset,
    load_tier2_ai4i_maintenance,
    load_tier2_credit_card_fraud,
    load_tier2_diabetes130,
    load_tier2_taiwan_credit_panel,
)
from imboreceipt.bench.gate05_sandbox import MultiChannelCohort, generate_gate05_cohort
from imboreceipt.bench.runner import (
    run_gate05_suite,
    run_multi_channel_cohort_benchmark,
    run_tier1_saturation_suite,
    run_tier2_credo_bench,
)

__all__ = [
    "MultiChannelCohort",
    "TIER1_PUBLIC_SPECS",
    "Tier1DatasetSpec",
    "generate_gate05_cohort",
    "load_tier1_dataset",
    "load_tier2_ai4i_maintenance",
    "load_tier2_credit_card_fraud",
    "load_tier2_diabetes130",
    "load_tier2_taiwan_credit_panel",
    "run_gate05_suite",
    "run_multi_channel_cohort_benchmark",
    "run_tier1_saturation_suite",
    "run_tier2_credo_bench",
]
