#!/usr/bin/env python3
"""Verify and regenerate Manuscript Tables 1, 3, 4, 5, and 6 from committed v2.2 JSON checkpoints.

Usage:
    python scripts/verify_v2_2_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS_DIR = ROOT / "results" / "two_phase" / "v2_2_blocks"

COHORT_ORDER = [
    "UCI_Diabetes_Discharge",
    "Home_Credit_7Table",
    "IEEE_CIS_Fraud",
    "Scania_APS_Fleet",
    "NeurIPS_BAF",
    "UCI_Taiwan_Credit",
]


def load_block(filename: str) -> dict:
    path = BLOCKS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_all_tables(verbose: bool = True) -> dict[str, bool]:
    b1 = load_block("block1_v2_2_weak_signal_and_twins.json")
    b2 = load_block("block2_v2_2_full_120split_leakfree_stress.json")
    b3 = load_block("block3_v2_2_stageb_comparators_and_feesweep.json")
    b4 = load_block("block4_v2_2_cfbh_shift_and_wcs.json")
    b5 = load_block("block5_v2_2_raceblind_diabetes_scenario.json")

    checks: dict[str, bool] = {}

    # --- Table 1: Sim A1, A2, B, C1, C2 ---
    a1 = b1["weak_sim_a1_1000rep"]
    c1 = b1["sim_c1_1col_asymmetric_twin_200rep"]
    c2 = b1["sim_c2_multicol_symmetric_twin_200rep"]
    checks["tab1_sim_a1_split_cov"] = (
        abs(a1["split_pilot_50_50_holm"]["ci95_cov_data_adaptive"]["rate"] - 0.956) < 1e-6
    )
    checks["tab1_sim_a1_power_split_vs_cv"] = (
        abs(a1["split_pilot_50_50_holm"]["power"]["rate"] - 0.590) < 1e-6
        and abs(a1["cv_4fold_holm"]["power"]["rate"] - 0.920) < 1e-6
    )
    checks["tab1_sim_c1_vs_c2_twins"] = (
        abs(c1["baseline_strat_perm"]["dup_twin_rate"]["rate"] - 0.005) < 1e-6
        and abs(c2["c2_fullpool_cpi_pilot"]["dup_twin_rate"]["rate"] - 0.990) < 1e-6
        and abs(c2["c2_chamma_cpi"]["both_mech_recovered"]["rate"] - 0.980) < 1e-6
        and abs(c2["c2_seq_cv"]["both_mech_recovered"]["rate"] - 0.690) < 1e-6
        and abs(c2["c2_seq_split_pilot"]["both_mech_recovered"]["rate"] - 0.240) < 1e-6
        and abs(c2["c2_seq_split_pilot"]["null_fwer"]["rate"] - 0.015) < 1e-6
    )

    # --- Table 3: Full 120-Split Leak-Free Stress Campaign ---
    stress = b2["leak_free_120split_stress"]
    total_splits = sum(int(stress[c]["n_splits"]) for c in COHORT_ORDER)
    total_lazy_match = sum(int(stress[c]["lazy_exact_match_count"]) for c in COHORT_ORDER)
    checks["tab3_120_splits_and_lazy_match"] = total_splits == 120 and total_lazy_match == 120

    seq_zero_waste = True
    for c in COHORT_ORDER:
        for arm_name in (
            "proposed_ht_holm",
            "no_lazy_ht_holm",
            "split_pilot_50_50_holm",
            "breslow_cain_holm",
        ):
            arm = stress[c]["arms"][arm_name]
            if arm["null_pct"] != 0.0 or arm["twin_pct"] != 0.0 or arm["echo_pct"] != 0.0:
                seq_zero_waste = False
    checks["tab3_seq_arms_zero_waste_all_120"] = seq_zero_waste
    checks["tab3_baf_leakfree_reconciles"] = (
        abs(stress["NeurIPS_BAF"]["arms"]["proposed_ht_holm"]["auprc_mean"] - 0.103149) < 1e-4
    )

    # --- Table 4: Canonical Holdouts & Fee Sweep ---
    can = b3["canonical_holdouts"]
    checks["tab4_all_6_cohorts_present"] = all(c in can for c in COHORT_ORDER)
    ieee_bu_diff = (
        can["IEEE_CIS_Fraud"]["arms_default_cq004"]["gh_voi_auto"]["nb"]
        - can["IEEE_CIS_Fraud"]["arms_default_cq004"]["bu_matched_q_auto"]["nb"]
    )
    checks["tab4_ieee_voi_vs_bu"] = abs(ieee_bu_diff - 0.659) < 0.01

    # --- Block 4: R=100 WCS vs Burn-In Shift ---
    baf_shift = b4["cfbh_wcs_burnin_100rep"]["NeurIPS_BAF"]["methods"]
    checks["fig3c_baf_100rep_shift"] = (
        baf_shift["in_era_exchangeable_cfbh"]["mean_fdr"] <= 0.75
        and baf_shift["unlabeled_covariate_shift_wcs"]["mean_fdr"] > 0.75
        and baf_shift["labeled_burnin_2500_cfbh"]["mean_fdr"] <= 0.75
    )

    # --- Tables 5 & 6: Race-Blind UCI Diabetes Scenario ---
    triad = b5["outer_19999_r20_triad_race_blind"]
    exch = {round(row["tau"], 3): row for row in b5["referral_exchange_rates_race_blind"]}
    sub = b5["subgroup_equity_and_calibration_race_blind"]
    checks["tab5_outer_split_reconciles"] = (
        triad["n_test_encounters"] == 19999
        and abs(exch[0.111]["delta_nb_vs_full100"] - 0.1508) < 0.01
        and abs(exch[0.15]["tp_per_1k_voi"] - 37.80) < 0.05
        and abs(exch[0.2]["tp_per_1k_voi"] - 12.80) < 0.05
    )
    checks["tab6_cochran_q_reconciles"] = (
        abs(sub["race"]["p_het_vs_full100"] - 0.0242) < 0.001
        and abs(sub["gender"]["p_het_vs_full100"] - 0.0409) < 0.001
    )

    if verbose:
        print("=== Paper 1 (v2.2.1) Checkpoint-to-Table Verification ===")
        print(
            f"Table 3 total leak-free splits: {total_splits}/120 "
            f"(lazy exact match: {total_lazy_match}/120)"
        )
        for c in COHORT_ORDER:
            a_def = stress[c]["arms"]["proposed_ht_holm"]
            a_os = stress[c]["arms"]["oneshot_cmi"]
            a_cpi = stress[c]["arms"]["cpi_pilot"]
            print(
                f"  {c:24s} | Seq-CMI={a_def['auprc_mean']:.4f}±{a_def['auprc_se']:.4f} "
                f"({a_def['null_pct']:.0f}/{a_def['twin_pct']:.0f}/{a_def['echo_pct']:.0f}%) | "
                f"OneShot twin={a_os['twin_pct']:.1f}% | cpi_pilot twin={a_cpi['twin_pct']:.1f}%"
            )
        print("All invariant checks:", checks)

    if not all(checks.values()):
        failed = [k for k, ok in checks.items() if not ok]
        raise AssertionError(f"Table verification checks failed: {failed}")
    return checks


if __name__ == "__main__":
    verify_all_tables(verbose=True)
