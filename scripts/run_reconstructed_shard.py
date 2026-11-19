"""Parallel shard runner for the reconstructed CREDO evaluation campaign."""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

from imboreceipt.bench.adversarial_sims import run_adversarial_simulation_suite
from imboreceipt.bench.datasets import (
    DEFAULT_CACHE_DIR,
    load_tier2_ai4i_maintenance,
    load_tier2_credit_card_fraud,
    load_tier2_diabetes130,
    load_tier2_taiwan_credit_panel,
)
from imboreceipt.bench.gate05_sandbox import generate_gate05_cohort
from imboreceipt.bench.runner import (
    run_multi_channel_cohort_benchmark,
    run_tier1_saturation_suite,
)
from imboreceipt.bench.wave2_datasets import (
    load_wave2_home_credit_relational,
    load_wave2_ieee_cis_fraud,
    load_wave2_neurips_baf,
    load_wave2_scania_aps,
)

SEEDS = (42, 43, 44, 45, 46)
OUT_DIR = Path("results/reconstructed")


def _build_cohort(key: str, w_idx: int, seed: int, with_proxies: bool):
    n_win = len(SEEDS)
    if key == "Gate05_Synthetic_MultiMechanism":
        return generate_gate05_cohort(n_samples=12000, seed=seed, selective_receipts=True)
    if key == "UCI_AI4I_2020_Maintenance":
        return load_tier2_ai4i_maintenance(
            cache_dir=DEFAULT_CACHE_DIR, seed=seed, include_synthetic_proxies=with_proxies
        )
    if key == "UCI_Taiwan_Credit_6m_Panel":
        return load_tier2_taiwan_credit_panel(
            cache_dir=DEFAULT_CACHE_DIR, seed=seed, include_synthetic_proxies=with_proxies
        )
    if key == "UCI_Diabetes_130US_Hospitals":
        return load_tier2_diabetes130(
            cache_dir=DEFAULT_CACHE_DIR,
            max_patients=None,
            seed=seed,
            include_synthetic_proxies=with_proxies,
        )
    if key == "ULB_CreditCard_Fraud_Chronological":
        return load_tier2_credit_card_fraud(
            cache_dir=DEFAULT_CACHE_DIR,
            max_negatives=None,
            window_idx=w_idx,
            n_windows=n_win,
            seed=seed,
            include_synthetic_proxies=with_proxies,
        )
    if key == "Scania_APS_76k_Fleet":
        return load_wave2_scania_aps(
            cache_dir=DEFAULT_CACHE_DIR, seed=seed, include_synthetic_proxies=with_proxies
        )
    if key == "Home_Credit_307k_7Table_Relational":
        return load_wave2_home_credit_relational(
            cache_dir=DEFAULT_CACHE_DIR,
            max_rows=None,
            seed=seed,
            include_synthetic_proxies=with_proxies,
        )
    if key == "IEEE_CIS_590k_MultiTable":
        return load_wave2_ieee_cis_fraud(
            cache_dir=DEFAULT_CACHE_DIR,
            window_idx=w_idx,
            n_windows=n_win,
            max_rows=None,
            seed=seed,
            include_synthetic_proxies=with_proxies,
        )
    if key == "NeurIPS_BAF_1M_Chronological":
        return load_wave2_neurips_baf(
            cache_dir=DEFAULT_CACHE_DIR,
            window_idx=w_idx,
            n_windows=n_win,
            max_rows=None,
            seed=seed,
            include_synthetic_proxies=with_proxies,
        )
    raise ValueError(f"Unknown cohort key: {key}")


def run_shard(shard_id: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_native = OUT_DIR / "checkpoints_native"
    ckpt_proxy = OUT_DIR / "checkpoints_proxy_stress"
    ckpt_native.mkdir(parents=True, exist_ok=True)
    ckpt_proxy.mkdir(parents=True, exist_ok=True)

    if shard_id == "shard1":
        print("[Shard 1] Running 7-scenario adversarial sims + Tier-1 suite...", flush=True)
        adv_json = OUT_DIR / "adversarial_simulations.json"
        if not adv_json.exists():
            adv = run_adversarial_simulation_suite(seeds=SEEDS)
            adv_json.write_text(json.dumps(adv, indent=2))
            print("[Shard 1] Completed adversarial_simulations.json", flush=True)
        for s in SEEDS:
            t1_dir = OUT_DIR / "seeds" / f"tier1_seed{s}"
            if not (t1_dir / "tier1_results.json").exists():
                run_tier1_saturation_suite(t1_dir, cache_dir=DEFAULT_CACHE_DIR, seed=s)
                print(f"[Shard 1] Completed Tier-1 seed={s}", flush=True)
                gc.collect()

    elif shard_id == "shard2":
        keys = [
            "Gate05_Synthetic_MultiMechanism",
            "UCI_AI4I_2020_Maintenance",
            "UCI_Taiwan_Credit_6m_Panel",
            "UCI_Diabetes_130US_Hospitals",
            "ULB_CreditCard_Fraud_Chronological",
            "Scania_APS_76k_Fleet",
        ]
        for k in keys:
            for w_i, s in enumerate(SEEDS):
                cp = ckpt_native / f"{k}_seed{s}.json"
                if not cp.exists():
                    coh = _build_cohort(k, w_i, s, with_proxies=False)
                    res = run_multi_channel_cohort_benchmark(
                        coh, seed=s, n_boot=200, include_complex_fusion=True
                    )
                    cp.write_text(json.dumps(res, indent=2))
                    print(
                        f"[Shard 2 Native] {k} s={s}: "
                        f"A0={res['arms']['calibrated_gbdt']['auprc']:.4f} "
                        f"A5={res['arms']['monolithic_matched_pilot_cmi']['auprc']:.4f} "
                        f"Ref={res['arms']['unrestricted_all_native_gbdt']['auprc']:.4f}",
                        flush=True,
                    )
                    del coh
                    gc.collect()
                if k != "Gate05_Synthetic_MultiMechanism":
                    cp_p = ckpt_proxy / f"{k}_seed{s}.json"
                    if not cp_p.exists():
                        coh_p = _build_cohort(k, w_i, s, with_proxies=True)
                        res_p = run_multi_channel_cohort_benchmark(
                            coh_p, seed=s, n_boot=100, include_complex_fusion=False
                        )
                        cp_p.write_text(json.dumps(res_p, indent=2))
                        print(f"[Shard 2 Proxy] {k} s={s} done", flush=True)
                        del coh_p
                        gc.collect()

    elif shard_id == "shard3":
        keys = [
            "Home_Credit_307k_7Table_Relational",
            "IEEE_CIS_590k_MultiTable",
        ]
        for k in keys:
            for w_i, s in enumerate(SEEDS):
                cp = ckpt_native / f"{k}_seed{s}.json"
                if not cp.exists():
                    coh = _build_cohort(k, w_i, s, with_proxies=False)
                    res = run_multi_channel_cohort_benchmark(
                        coh, seed=s, n_boot=200, include_complex_fusion=True
                    )
                    cp.write_text(json.dumps(res, indent=2))
                    print(
                        f"[Shard 3 Native] {k} s={s}: "
                        f"A0={res['arms']['calibrated_gbdt']['auprc']:.4f} "
                        f"A5={res['arms']['monolithic_matched_pilot_cmi']['auprc']:.4f} "
                        f"Ref={res['arms']['unrestricted_all_native_gbdt']['auprc']:.4f}",
                        flush=True,
                    )
                    del coh
                    gc.collect()
                cp_p = ckpt_proxy / f"{k}_seed{s}.json"
                if not cp_p.exists():
                    coh_p = _build_cohort(k, w_i, s, with_proxies=True)
                    res_p = run_multi_channel_cohort_benchmark(
                        coh_p, seed=s, n_boot=100, include_complex_fusion=False
                    )
                    cp_p.write_text(json.dumps(res_p, indent=2))
                    print(f"[Shard 3 Proxy] {k} s={s} done", flush=True)
                    del coh_p
                    gc.collect()

    elif shard_id == "shard4":
        k = "NeurIPS_BAF_1M_Chronological"
        for w_i, s in enumerate(SEEDS):
            cp = ckpt_native / f"{k}_seed{s}.json"
            if not cp.exists():
                coh = _build_cohort(k, w_i, s, with_proxies=False)
                res = run_multi_channel_cohort_benchmark(
                    coh, seed=s, n_boot=200, include_complex_fusion=True
                )
                cp.write_text(json.dumps(res, indent=2))
                print(
                    f"[Shard 4 Native] {k} s={s}: "
                    f"A0={res['arms']['calibrated_gbdt']['auprc']:.4f} "
                    f"A5={res['arms']['monolithic_matched_pilot_cmi']['auprc']:.4f} "
                    f"Ref={res['arms']['unrestricted_all_native_gbdt']['auprc']:.4f}",
                    flush=True,
                )
                del coh
                gc.collect()
                cp_p = ckpt_proxy / f"{k}_seed{s}.json"
                if not cp_p.exists():
                    coh_p = _build_cohort(k, w_i, s, with_proxies=True)
                    res_p = run_multi_channel_cohort_benchmark(
                        coh_p, seed=s, n_boot=100, include_complex_fusion=False
                    )
                    cp_p.write_text(json.dumps(res_p, indent=2))
                    print(f"[Shard 4 Proxy] {k} s={s} done", flush=True)
                    del coh_p
                    gc.collect()

    elif shard_id == "shard5":
        # Reverse-order helper on Scania + ULB using atomic .tmp -> .json rename
        for k in ("Scania_APS_76k_Fleet", "ULB_CreditCard_Fraud_Chronological"):
            for w_i, s in reversed(list(enumerate(SEEDS))):
                cp = ckpt_native / f"{k}_seed{s}.json"
                if not cp.exists():
                    coh = _build_cohort(k, w_i, s, with_proxies=False)
                    res = run_multi_channel_cohort_benchmark(
                        coh, seed=s, n_boot=200, include_complex_fusion=True
                    )
                    tmp = ckpt_native / f".tmp_{k}_seed{s}_s5.json"
                    tmp.write_text(json.dumps(res, indent=2))
                    if not cp.exists():
                        tmp.replace(cp)
                    else:
                        tmp.unlink(missing_ok=True)
                    print(f"[Shard 5 Native] {k} s={s} done", flush=True)
                    del coh
                    gc.collect()
                cp_p = ckpt_proxy / f"{k}_seed{s}.json"
                if not cp_p.exists():
                    coh_p = _build_cohort(k, w_i, s, with_proxies=True)
                    res_p = run_multi_channel_cohort_benchmark(
                        coh_p, seed=s, n_boot=100, include_complex_fusion=False
                    )
                    tmp_p = ckpt_proxy / f".tmp_{k}_seed{s}_s5.json"
                    tmp_p.write_text(json.dumps(res_p, indent=2))
                    if not cp_p.exists():
                        tmp_p.replace(cp_p)
                    else:
                        tmp_p.unlink(missing_ok=True)
                    print(f"[Shard 5 Proxy] {k} s={s} done", flush=True)
                    del coh_p
                    gc.collect()

    elif shard_id == "shard6":
        k = "IEEE_CIS_590k_MultiTable"
        for w_i, s in reversed(list(enumerate(SEEDS))):
            cp = ckpt_native / f"{k}_seed{s}.json"
            if not cp.exists():
                coh = _build_cohort(k, w_i, s, with_proxies=False)
                res = run_multi_channel_cohort_benchmark(
                    coh, seed=s, n_boot=200, include_complex_fusion=True
                )
                tmp = ckpt_native / f".tmp_{k}_seed{s}_s6.json"
                tmp.write_text(json.dumps(res, indent=2))
                if not cp.exists():
                    tmp.replace(cp)
                else:
                    tmp.unlink(missing_ok=True)
                print(f"[Shard 6 Native] {k} s={s} done", flush=True)
                del coh
                gc.collect()
            cp_p = ckpt_proxy / f"{k}_seed{s}.json"
            if not cp_p.exists():
                coh_p = _build_cohort(k, w_i, s, with_proxies=True)
                res_p = run_multi_channel_cohort_benchmark(
                    coh_p, seed=s, n_boot=100, include_complex_fusion=False
                )
                tmp_p = ckpt_proxy / f".tmp_{k}_seed{s}_s6.json"
                tmp_p.write_text(json.dumps(res_p, indent=2))
                if not cp_p.exists():
                    tmp_p.replace(cp_p)
                else:
                    tmp_p.unlink(missing_ok=True)
                print(f"[Shard 6 Proxy] {k} s={s} done", flush=True)
                del coh_p
                gc.collect()
    else:
        raise ValueError(f"Unknown shard_id: {shard_id}")


if __name__ == "__main__":
    run_shard(sys.argv[1])
