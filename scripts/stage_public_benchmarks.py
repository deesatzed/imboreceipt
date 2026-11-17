"""Stage and SHA-256 cache all Tier-1 and Tier-2 public OpenML/imblearn datasets locally."""

from __future__ import annotations

from pathlib import Path

from imboreceipt.bench.datasets import (
    DEFAULT_CACHE_DIR,
    TIER1_PUBLIC_SPECS,
    load_tier1_dataset,
    load_tier2_ai4i_maintenance,
    load_tier2_credit_card_fraud,
    load_tier2_diabetes130,
    load_tier2_taiwan_credit_panel,
)


def stage_all(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    print("Staging Tier-1 Canonical OpenML / imblearn datasets...")
    for spec in TIER1_PUBLIC_SPECS:
        X, y, sha = load_tier1_dataset(spec, cache_dir=cache_dir)
        print(
            f"  [OK] {spec.key:16s} rows={len(X):6d} cols={X.shape[1]:3d} "
            f"pos={int(y.sum()):5d} sha256={sha[:12]}"
        )

    print("Staging Tier-2 CREDO-Bench Multi-Channel UCI/OpenML cohorts...")
    for loader_fn in (
        load_tier2_ai4i_maintenance,
        load_tier2_taiwan_credit_panel,
        load_tier2_diabetes130,
        load_tier2_credit_card_fraud,
    ):
        cohort = loader_fn(cache_dir=cache_dir)
        print(
            f"  [OK] {cohort.name:34s} rows={len(cohort.X_base):6d} "
            f"pos={int(cohort.y_true.sum()):5d} channels={len(cohort.candidates)}"
        )


if __name__ == "__main__":
    stage_all()
