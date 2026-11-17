"""Command-line interface for imboreceipt (CREDO & CREDO-Bench)."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from imboreceipt.bench.datasets import DEFAULT_CACHE_DIR
from imboreceipt.bench.runner import (
    run_gate05_suite,
    run_multiseed_paper_campaign,
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
@click.option("--uncapped/--capped", default=False)
def cmd_bench_tier2(out: Path, cache_dir: Path, seed: int, uncapped: bool) -> None:
    """Run Tier-2 CREDO-Bench Multi-Channel & Receipt Public Cohort Suite."""
    results = run_tier2_credo_bench(out_dir=out, cache_dir=cache_dir, seed=seed, uncapped=uncapped)
    tbl = Table(title="Tier-2 CREDO-Bench: Public Multi-Channel & Receipt Benchmarks")
    tbl.add_column("Cohort", style="cyan")
    tbl.add_column("Base GBDT", justify="right")
    tbl.add_column("REG Mono", justify="right")
    tbl.add_column("Orth Lattice", justify="right")
    tbl.add_column("Full CREDO (Hybrid)", justify="right", style="bold green")
    tbl.add_column("kappa_0", justify="right")

    for r in results:
        b = r["arms"]["calibrated_gbdt"]
        m = r["arms"]["monolithic_credo_reg"]
        o = r["arms"]["pure_orthogonal_lattice"]
        c = r["arms"]["full_credo_lattice"]
        tbl.add_row(
            r["cohort_name"],
            f"{b['auprc']:.4f}",
            f"{m['auprc']:.4f}",
            f"{o['auprc']:.4f}",
            f"{c['auprc']:.4f}",
            f"{r['mean_kappa0']:.2f} (w={r['orthogonal_lattice_weight']:.2f})",
        )
    console.print(tbl)
    console.print(f"[green]Saved Tier-2 CREDO-Bench artifacts & plot to {out}[/green]")


@main.command("bench-paper-full")
@click.option("--out", type=click.Path(path_type=Path), default=Path("artifacts/bench/paper_full"))
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE_DIR)
@click.option("--n-seeds", type=int, default=5)
@click.option("--uncapped/--capped", default=True)
def cmd_bench_paper_full(out: Path, cache_dir: Path, n_seeds: int, uncapped: bool) -> None:
    """Run full multi-seed Paper A campaign across Tier 0, Tier 1, and Uncapped Tier 2."""
    seeds = tuple(42 + i for i in range(n_seeds))
    summary = run_multiseed_paper_campaign(
        out_dir=out, cache_dir=cache_dir, seeds=seeds, uncapped=uncapped
    )
    tbl = Table(
        title=f"Paper A Multi-Seed ({n_seeds}x, uncapped={uncapped}) Knockout Ablation Summary"
    )
    tbl.add_column("Cohort", style="cyan")
    tbl.add_column("Base AUPRC", justify="right")
    tbl.add_column("Global MI", justify="right")
    tbl.add_column("Pure Orth", justify="right")
    tbl.add_column("Full CREDO", justify="right", style="bold green")
    tbl.add_column("kappa_0 -> w_orth", justify="right")
    tbl.add_column("Wilcoxon p", justify="right")

    for c in summary["multi_channel_summary"]:
        ab = c["ablation_ladder"]
        tbl.add_row(
            c["cohort_name"],
            f"{ab['calibrated_gbdt']['auprc']['mean']:.4f}±{ab['calibrated_gbdt']['auprc']['sd']:.3f}",
            f"{ab['monolithic_global_mi']['auprc']['mean']:.4f}±{ab['monolithic_global_mi']['auprc']['sd']:.3f}",
            f"{ab['pure_orthogonal_lattice']['auprc']['mean']:.4f}±{ab['pure_orthogonal_lattice']['auprc']['sd']:.3f}",
            f"{ab['full_credo_lattice']['auprc']['mean']:.4f}±{ab['full_credo_lattice']['auprc']['sd']:.3f}",
            f"{c['mean_kappa0']['mean']:.2f} -> {c['orthogonal_lattice_weight']['mean']:.2f}",
            f"{c['wilcoxon_credo_vs_base_p']:.4f}",
        )
    console.print(tbl)
    console.print(f"[green]Saved multi-seed Paper A campaign artifacts to {out}[/green]")


if __name__ == "__main__":
    main()
