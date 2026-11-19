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
    run_reconstructed_paper_campaign,
    run_tier1_saturation_suite,
    run_tier2_credo_bench,
    run_wave2_paper_campaign,
)
from imboreceipt.bench.two_phase_campaign import (
    aggregate_two_phase_campaign,
    run_single_cohort_split,
    run_statistical_theory_simulations,
    run_supplement_fixed_table_tost,
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


@main.command("bench-wave2")
@click.option("--out", type=click.Path(path_type=Path), default=Path("artifacts/bench/wave2_full"))
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE_DIR)
@click.option("--n-seeds", type=int, default=5)
@click.option("--uncapped/--capped", default=True)
def cmd_bench_wave2(out: Path, cache_dir: Path, n_seeds: int, uncapped: bool) -> None:
    """Run Wave-2 Flagship Expansion (Scania 76k, Home Credit 307k, IEEE-CIS 590k, BAF 1M)."""
    seeds = tuple(42 + i for i in range(n_seeds))
    summary = run_wave2_paper_campaign(
        out_dir=out, cache_dir=cache_dir, seeds=seeds, uncapped=uncapped
    )
    tbl = Table(title=f"Unified 9-Cohort CREDO-Bench Summary ({n_seeds}x, uncapped={uncapped})")
    tbl.add_column("Cohort", style="cyan")
    tbl.add_column("Base AUPRC", justify="right")
    tbl.add_column("Global MI", justify="right")
    tbl.add_column("6-Ens (A5b)", justify="right")
    tbl.add_column("Full CREDO", justify="right", style="bold green")
    tbl.add_column("Pilot %", justify="right")
    tbl.add_column("Holm FWER p", justify="right")

    for c in summary["multi_channel_summary"]:
        ab = c["ablation_ladder"]
        tbl.add_row(
            c["cohort_name"],
            f"{ab['calibrated_gbdt']['auprc']['mean']:.4f}",
            f"{ab['monolithic_global_mi']['auprc']['mean']:.4f}",
            f"{ab['monolithic_credo_ensemble']['auprc']['mean']:.4f}",
            f"{ab['full_credo_lattice']['auprc']['mean']:.4f}",
            f"{100.0 * c['pilot_fraction_measured']['mean']:.1f}%",
            f"{c['holm_bonferroni_fwer_p']:.5f}",
        )
    console.print(tbl)
    console.print(f"[green]Saved unified 9-cohort Wave-2 campaign artifacts to {out}[/green]")


@main.command("bench-reconstructed")
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("results/reconstructed"),
)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE_DIR)
@click.option("--n-seeds", type=int, default=5)
@click.option("--uncapped/--capped", default=True)
def cmd_bench_reconstructed(out: Path, cache_dir: Path, n_seeds: int, uncapped: bool) -> None:
    """Run reconstructed single-factor CREDO evaluation (Native + Proxy Stress + 7 Sims)."""
    seeds = tuple(42 + i for i in range(n_seeds))
    summary = run_reconstructed_paper_campaign(
        out_dir=out, cache_dir=cache_dir, seeds=seeds, uncapped=uncapped
    )
    tbl = Table(
        title=f"Reconstructed Native-Channel CREDO Summary ({n_seeds}x, uncapped={uncapped})"
    )
    tbl.add_column("Cohort", style="cyan")
    tbl.add_column("Base A0", justify="right")
    tbl.add_column("GlobMI A4", justify="right")
    tbl.add_column("UnmCMI A4u", justify="right")
    tbl.add_column("StratCMI A5", justify="right", style="bold green")
    tbl.add_column("AllNative Ref", justify="right")
    tbl.add_column("Holm p(A5>A0)", justify="right")

    for c in summary["native_channel_summary"]:
        ab = c["ablation_ladder"]
        tbl.add_row(
            c["cohort_name"],
            f"{ab['calibrated_gbdt']['auprc']['mean']:.4f}",
            f"{ab['monolithic_global_mi']['auprc']['mean']:.4f}",
            f"{ab['monolithic_unmatched_pilot_cmi']['auprc']['mean']:.4f}",
            f"{ab['monolithic_matched_pilot_cmi']['auprc']['mean']:.4f}",
            f"{ab['unrestricted_all_native_gbdt']['auprc']['mean']:.4f}",
            f"{c['holm_prim_vs_base_auprc_p']:.5f}",
        )
    console.print(tbl)
    console.print(f"[green]Saved reconstructed campaign receipts & manifests to {out}[/green]")


@main.command("bench-two-phase")
@click.option("--out", type=click.Path(path_type=Path), default=Path("results/two_phase"))
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE_DIR)
@click.option("--n-splits", type=int, default=20)
@click.option("--run-sims/--skip-sims", default=True)
@click.option("--run-supplement/--skip-supplement", default=True)
def cmd_bench_two_phase(
    out: Path,
    cache_dir: Path,
    n_splits: int,
    run_sims: bool,
    run_supplement: bool,
) -> None:
    """Run Two-Phase Stratified Pilot Discovery & Streaming Selective Acquisition campaign."""
    import json

    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    if run_sims:
        sim_out = run_statistical_theory_simulations()
        (out / "theory_simulations.json").write_text(json.dumps(sim_out, indent=2))

    if run_supplement:
        supp_out = run_supplement_fixed_table_tost(n_splits=n_splits, cache_dir=cache_dir)
        (out / "supplement_fixed_table_tost.json").write_text(json.dumps(supp_out, indent=2))

    cohorts = (
        "UCI_Diabetes_Discharge",
        "Home_Credit_7Table",
        "IEEE_CIS_Fraud",
        "Scania_APS_Fleet",
        "NeurIPS_BAF",
        "UCI_Taiwan_Credit",
    )
    for c_key in cohorts:
        for s_idx in range(n_splits):
            p = ckpt / f"{c_key}_split{s_idx:02d}.json"
            if not p.exists():
                r = run_single_cohort_split(
                    c_key, split_idx=s_idx, n_splits=n_splits, cache_dir=cache_dir
                )
                p.write_text(json.dumps(r, indent=2))

    summary = aggregate_two_phase_campaign(out, cohort_keys=cohorts, n_splits=n_splits)
    tbl = Table(title=f"Two-Phase Pilot Discovery & TOST Non-Inferiority ({n_splits} Splits)")
    tbl.add_column("Cohort", style="cyan")
    tbl.add_column("Base AUPRC", justify="right")
    tbl.add_column("Neyman 15%", justify="right", style="bold green")
    tbl.add_column("Full 100%", justify="right")
    tbl.add_column("Delta (90% CI)", justify="right")
    tbl.add_column("TOST NI p", justify="right")

    for c in summary["cohorts"]:
        ns = c["native_summary"]
        tost = c["tost_neyman15_vs_full_auprc"]
        tbl.add_row(
            c["cohort_key"],
            f"{ns['base_only']['auprc']['mean']:.4f}",
            f"{ns['neyman_15pct_seq']['auprc']['mean']:.4f}",
            f"{ns['full_100pct_seq']['auprc']['mean']:.4f}",
            f"{tost['mean_delta']:+.4f} [{tost['ci90_low']:+.4f}, {tost['ci90_high']:+.4f}]",
            f"{tost['p_noninferiority']:.4g}",
        )
    console.print(tbl)
    console.print(f"[green]Saved two-phase summary & figures to {out}[/green]")


if __name__ == "__main__":
    main()
