#!/usr/bin/env python3
"""Shard runner for the R=20 Two-Phase Discovery & Streaming Selective Acquisition campaign."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from imboreceipt.bench.datasets import DEFAULT_CACHE_DIR
from imboreceipt.bench.two_phase_campaign import (
    aggregate_two_phase_campaign,
    run_single_cohort_split,
    run_statistical_theory_simulations,
    run_supplement_fixed_table_tost,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Two-Phase campaign shard.")
    parser.add_argument(
        "--mode",
        choices=("sim", "supplement", "cohort", "sensitivity", "aggregate"),
        required=True,
    )
    parser.add_argument("--out", type=Path, default=Path("results/two_phase"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cohort", type=str, default="UCI_Diabetes_Discharge")
    parser.add_argument("--start-split", type=int, default=0)
    parser.add_argument("--end-split", type=int, default=20)
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.mode == "sim":
        print("[sim] Running statistical theory simulations (Sim A, Sim B, Sim C)...", flush=True)
        res = run_statistical_theory_simulations(n_rep_coverage=80, n_rep_mechanisms=40, seed=100)
        (args.out / "theory_simulations.json").write_text(json.dumps(res, indent=2))
        print(f"[sim] Done in {time.time() - t0:.1f}s.", flush=True)
        return

    if args.mode == "supplement":
        print(f"[supplement] Running R={args.n_splits} fixed-table TOST suite...", flush=True)
        res = run_supplement_fixed_table_tost(n_splits=args.n_splits, cache_dir=args.cache_dir)
        (args.out / "supplement_fixed_table_tost.json").write_text(json.dumps(res, indent=2))
        print(f"[supplement] Done in {time.time() - t0:.1f}s.", flush=True)
        return

    if args.mode == "cohort":
        for s_idx in range(args.start_split, args.end_split):
            out_p = ckpt_dir / f"{args.cohort}_split{s_idx:02d}.json"
            if out_p.exists() and not args.force:
                print(f"[{args.cohort}] Split {s_idx:02d} already exists, skipping.", flush=True)
                continue
            ts = time.time()
            r = run_single_cohort_split(
                args.cohort,
                split_idx=s_idx,
                n_splits=args.n_splits,
                cache_dir=args.cache_dir,
            )
            out_p.write_text(json.dumps(r, indent=2))
            print(
                f"[{args.cohort}] Split {s_idx:02d}/{args.n_splits - 1:02d} "
                f"in {time.time() - ts:.1f}s "
                f"(base={r['native_arms']['base_only']['auprc']:.4f}, "
                f"ney15={r['native_arms']['neyman_15pct_seq']['auprc']:.4f}, "
                f"full100={r['native_arms']['full_100pct_seq']['auprc']:.4f})",
                flush=True,
            )
        return

    if args.mode == "sensitivity":
        sens_keys = ("UCI_Diabetes_RaceBlind", "UCI_Taiwan_BaselineCurrent")
        for sk in sens_keys:
            for s_idx in range(args.start_split, args.end_split):
                out_p = ckpt_dir / f"{sk}_split{s_idx:02d}.json"
                if out_p.exists() and not args.force:
                    continue
                ts = time.time()
                r = run_single_cohort_split(
                    sk, split_idx=s_idx, n_splits=args.n_splits, cache_dir=args.cache_dir
                )
                out_p.write_text(json.dumps(r, indent=2))
                print(
                    f"[sens:{sk}] Split {s_idx:02d} finished in {time.time() - ts:.1f}s", flush=True
                )
        return

    if args.mode == "aggregate":
        summary = aggregate_two_phase_campaign(args.out, n_splits=args.n_splits)
        # Also aggregate sensitivity cohorts if present
        sens_summary = aggregate_two_phase_campaign(
            args.out,
            cohort_keys=("UCI_Diabetes_RaceBlind", "UCI_Taiwan_BaselineCurrent"),
            n_splits=args.n_splits,
        )
        # Re-render primary 6-cohort summary and attach sensitivity summary cleanly
        summary = aggregate_two_phase_campaign(args.out, n_splits=args.n_splits)
        summary["sensitivity_cohorts"] = sens_summary.get("cohorts", [])
        (args.out / "two_phase_campaign_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[aggregate] Emitted summary and publication figures to {args.out}", flush=True)


if __name__ == "__main__":
    main()
