"""Command-line interface for imboreceipt (CREDO & CREDO-Bench)."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from imboreceipt.bench.datasets import DEFAULT_CACHE_DIR
from imboreceipt.bench.runner import (
    run_gate05_suite,
    run_tier1_saturation_suite,
    run_tier2_credo_bench,
)

console = Console()


@click.group()
def main() -> None:
    """imboreceipt: CREDO & CREDO-Bench rare-event benchmark CLI."""


@main.command("bench-gate05")
@click.option("--out", type=click.Path(path_type=Path), default=Path("artifacts/bench/gate05"))
@click.option("--seed", type=int, default=42)
def cmd_bench_gate05(out: Path, seed: int) -> None:
    """Run Gate 0.5 multi-mechanism + selective-receipt ground-truth benchmark."""
    res = run_gate05_suite(out_dir=out, seed=seed)
    tbl = Table(title=f"Gate 0.5 Ground-Truth Benchmark ({res['cohort_name']})")
    tbl.add_column("Arm", style="cyan")
    tbl.add_column("Test AUPRC", justify="right")
    tbl.add_column("NP TPR@FPR", justify="right")
    tbl.add_column("Recall@Target q", justify="right")
    tbl.add_column("ECE", justify="right")

    for arm_name, m in res["arms"].items():
        tbl.add_row(
            arm_name,
            f"{m['auprc']:.4f}",
            f"{m['np_tpr_at_fpr_budget']:.4f}",
            f"{m['recall_at_target_precision']:.4f}",
            f"{m['ece']:.4f}",
        )
    console.print(tbl)
    console.print(f"[green]Saved Gate 0.5 receipt to {out / 'gate05_results.json'}[/green]")


@main.command("bench-tier1")
@click.option("--out", type=click.Path(path_type=Path), default=Path("artifacts/bench/tier1"))
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE_DIR)
@click.option("--seed", type=int, default=42)
def cmd_bench_tier1(out: Path, cache_dir: Path, seed: int) -> None:
    """Run Tier-1 Canonical OpenML/imblearn Imbalanced Saturation & Thermometer Suite."""
    results = run_tier1_saturation_suite(out_dir=out, cache_dir=cache_dir, seed=seed)
    tbl = Table(title="Tier-1 Public Imbalanced Benchmarks: Ceiling Diagnosis & SMOTE Delta")
    tbl.add_column("Dataset", style="cyan")
    tbl.add_column("n+ Test", justify="right")
    tbl.add_column("MDE +-95%", justify="right")
    tbl.add_column("Regime", style="yellow")
    tbl.add_column("GBDT AUPRC", justify="right")
    tbl.add_column("SMOTE Delta (95% CI)", justify="right")

    for r in results:
        d = r["decoder_deltas_vs_calibrated_gbdt"]["smote_gbdt"]
        tbl.add_row(
            r["display_name"],
            str(r["n_pos_test"]),
            f"±{r['gate_c']['recall_ci_half_width_at_80']:.3f}",
            r["four_ceilings"]["bottleneck_regime"],
            f"{r['arms']['calibrated_gbdt']['auprc']:.4f}",
            f"{d['delta_auprc']:+.4f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]",
        )
    console.print(tbl)
    console.print(f"[green]Saved Tier-1 artifacts & plot to {out}[/green]")


@main.command("bench-tier2")
@click.option("--out", type=click.Path(path_type=Path), default=Path("artifacts/bench/tier2"))
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE_DIR)
@click.option("--seed", type=int, default=42)
def cmd_bench_tier2(out: Path, cache_dir: Path, seed: int) -> None:
    """Run Tier-2 CREDO-Bench Multi-Channel & Receipt Public Cohort Suite."""
    results = run_tier2_credo_bench(out_dir=out, cache_dir=cache_dir, seed=seed)
    tbl = Table(title="Tier-2 CREDO-Bench: Public Multi-Channel & Receipt Benchmarks")
    tbl.add_column("Cohort", style="cyan")
    tbl.add_column("Base GBDT AUPRC", justify="right")
    tbl.add_column("SMOTE AUPRC", justify="right")
    tbl.add_column("CREDO Lattice AUPRC", justify="right", style="bold green")
    tbl.add_column("NP TPR (Base->CREDO)", justify="right")
    tbl.add_column("Stage-2 Sensor Saved", justify="right")

    for r in results:
        b = r["arms"]["calibrated_gbdt"]
        s = r["arms"]["smote_gbdt"]
        c = r["arms"]["full_credo_lattice"]
        fov = r["conformal_foveation"]
        tbl.add_row(
            r["cohort_name"],
            f"{b['auprc']:.4f}",
            f"{s['auprc']:.4f}",
            f"{c['auprc']:.4f}",
            f"{b['np_tpr_at_fpr_budget']:.3f} -> {c['np_tpr_at_fpr_budget']:.3f}",
            f"{100.0 * fov['stage2_sensor_savings_fraction']:.1f}%",
        )
    console.print(tbl)
    console.print(f"[green]Saved Tier-2 CREDO-Bench artifacts & plot to {out}[/green]")


if __name__ == "__main__":
    main()
