"""CREDO-Bench experiment runners, statistical hypothesis tests, and publication figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/imboreceipt_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from imboreceipt.bench.datasets import (
    DEFAULT_CACHE_DIR,
    TIER1_PUBLIC_SPECS,
    load_tier1_dataset,
    load_tier2_ai4i_maintenance,
    load_tier2_credit_card_fraud,
    load_tier2_diabetes130,
    load_tier2_taiwan_credit_panel,
)
from imboreceipt.bench.gate05_sandbox import MultiChannelCohort, generate_gate05_cohort
from imboreceipt.contract.decision import DecisionContract, gate_c_audit
from imboreceipt.core.bootstrap import paired_cluster_bootstrap
from imboreceipt.core.types import freeze_split, infer_semantic_types
from imboreceipt.discover.selector import (
    score_channels_baseline_strategy,
    score_channels_credo_reg,
    select_channels_within_budget,
)
from imboreceipt.discover.strata import build_consensus_residual_strata, sample_case_cohort_pilot
from imboreceipt.frontier.baselines import (
    fit_single_calibrated_gbdt,
    train_decoder_saturation_suite,
)
from imboreceipt.frontier.ceilings import (
    diagnose_four_ceilings,
    evaluate_certified_frontier,
    select_neyman_pearson_threshold,
)
from imboreceipt.lattice.foveation import BinaryConformalFoveator
from imboreceipt.lattice.or_of_ands import ConfirmRescueLattice
from imboreceipt.receipt.audit import audit_receipts
from imboreceipt.receipt.repair import ReceiptRepairer


def run_multi_channel_cohort_benchmark(
    cohort: MultiChannelCohort,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the full 9-arm CREDO-Bench evaluation on a MultiChannelCohort."""
    df_meta = cohort.X_base.copy()
    df_meta["__y_split__"] = cohort.y_tilde
    group_col: str | None = None
    time_col: str | None = None
    if cohort.group_ids is not None:
        df_meta["__grp__"] = cohort.group_ids
        group_col = "__grp__"
    elif cohort.time_order is not None:
        df_meta["__time__"] = cohort.time_order
        time_col = "__time__"

    split = freeze_split(
        df_meta,
        "__y_split__",
        group_col=group_col,
        time_col=time_col,
        seed=seed,
    )
    tr_i = np.array(split.train_idx, dtype=int)
    cal_i = np.array(split.cal_idx, dtype=int)
    val_i = np.array(split.val_idx, dtype=int)
    tst_i = np.array(split.test_idx, dtype=int)

    Xb_tr, Xb_cal = cohort.X_base.iloc[tr_i], cohort.X_base.iloc[cal_i]
    Xb_val, Xb_tst = cohort.X_base.iloc[val_i], cohort.X_base.iloc[tst_i]
    Zp_tr, Zp_cal = cohort.Z_pool.iloc[tr_i], cohort.Z_pool.iloc[cal_i]
    Zp_val, Zp_tst = cohort.Z_pool.iloc[val_i], cohort.Z_pool.iloc[tst_i]

    yt_tr = cohort.y_tilde[tr_i]
    yt_cal = cohort.y_tilde[cal_i]
    # Evaluation targets: true event Y on val/test (de-censored when competing censoring exists)
    eval_val_mask = np.ones(val_i.size, dtype=bool)
    eval_tst_mask = np.ones(tst_i.size, dtype=bool)
    if cohort.competing_censor_mask is not None:
        eval_val_mask = ~(cohort.competing_censor_mask[val_i] & (cohort.y_true[val_i] == 0))
        eval_tst_mask = ~(cohort.competing_censor_mask[tst_i] & (cohort.y_true[tst_i] == 0))

    Xb_val_e = Xb_val.iloc[eval_val_mask]
    Zp_val_e = Zp_val.iloc[eval_val_mask]
    y_val_e = cohort.y_true[val_i][eval_val_mask]

    Xb_tst_e = Xb_tst.iloc[eval_tst_mask]
    Zp_tst_e = Zp_tst.iloc[eval_tst_mask]
    y_tst_e = cohort.y_true[tst_i][eval_tst_mask]
    grp_tst_e = cohort.group_ids[tst_i][eval_tst_mask] if cohort.group_ids is not None else None

    sem_types = cohort.base_semantic_types or infer_semantic_types(Xb_tr)

    # Stage C & R: Contract + Receipt Audit & Repair
    gate_c = gate_c_audit(
        cohort.contract,
        prevalence=max(1e-5, float(yt_tr.mean())),
        n_pos_test=int(y_tst_e.sum()),
    )
    rcpt_audit = audit_receipts(
        Xb_tr,
        yt_tr,
        receipt_cols=cohort.receipt_cols,
        future_positive_mask=(
            cohort.future_positive_mask[tr_i] if cohort.future_positive_mask is not None else None
        ),
        competing_censor_mask=(
            cohort.competing_censor_mask[tr_i] if cohort.competing_censor_mask is not None else None
        ),
        traps_df=cohort.traps_df.iloc[tr_i] if cohort.traps_df is not None else None,
        seed=seed,
    )

    repairer = ReceiptRepairer(seed=seed)
    repaired = repairer.fit_transform_train(
        Xb_tr,
        yt_tr,
        traps_train=cohort.traps_df.iloc[tr_i] if cohort.traps_df is not None else None,
        future_positive_mask=(
            cohort.future_positive_mask[tr_i] if cohort.future_positive_mask is not None else None
        ),
        competing_censor_mask=(
            cohort.competing_censor_mask[tr_i] if cohort.competing_censor_mask is not None else None
        ),
    )

    # Stage E: 4-Ceiling Diagnosis & Decoder Saturation Arms on fixed X_base
    ceil_diag = diagnose_four_ceilings(
        cohort.contract,
        Xb_tr,
        yt_tr,
        Xb_cal,
        yt_cal,
        Xb_val_e,
        y_val_e,
        rcpt_audit,
        semantic_types=sem_types,
        seed=seed,
    )
    decoder_arms = train_decoder_saturation_suite(
        Xb_tr, yt_tr, Xb_cal, yt_cal, semantic_types=sem_types, seed=seed
    )

    arm_metrics: dict[str, dict[str, Any]] = {}
    arm_test_probs: dict[str, np.ndarray] = {}

    for name in (
        "calibrated_gbdt",
        "smote_gbdt",
        "cost_sensitive_gbdt",
        "latent_copula_gbdt",
        "soft_avg_ensemble",
    ):
        arm = decoder_arms[name]
        vp = arm.predict_proba(Xb_val_e)
        tp = arm.predict_proba(Xb_tst_e)
        arm_test_probs[name] = tp
        arm_metrics[name] = evaluate_certified_frontier(
            name, y_val_e, vp, y_tst_e, tp, cohort.contract
        ).model_dump()

    # Arm: Receipt-Repaired Baseline on X_base alone
    rcpt_only_arm = fit_single_calibrated_gbdt(
        repaired.X_expanded,
        repaired.y_expanded,
        Xb_cal,
        yt_cal,
        semantic_types=sem_types,
        sample_weight=repaired.sample_weight,
        arm_name="receipt_repaired_xbase",
        seed=seed,
    )
    vp_rcpt = rcpt_only_arm.predict_proba(Xb_val_e)
    tp_rcpt = rcpt_only_arm.predict_proba(Xb_tst_e)
    arm_test_probs["receipt_repaired_xbase"] = tp_rcpt
    arm_metrics["receipt_repaired_xbase"] = evaluate_certified_frontier(
        "receipt_repaired_xbase", y_val_e, vp_rcpt, y_tst_e, tp_rcpt, cohort.contract
    ).model_dump()

    # Stage D: Residual Strata + Case-Cohort Pilot + Channel Selection Strategies
    strata = build_consensus_residual_strata(Xb_tr, yt_tr, semantic_types=sem_types, seed=seed)
    pilot = sample_case_cohort_pilot(
        yt_tr, strata.oof_consensus_prob, controls_per_positive=4, seed=seed
    )
    spec_map = {c.channel_id: c for c in cohort.candidates}

    credo_scores = score_channels_credo_reg(
        Xb_tr, Zp_tr, yt_tr, cohort.candidates, strata, pilot, cohort.contract, seed=seed
    )
    chosen_credo = select_channels_within_budget(
        credo_scores, spec_map, cohort.acquisition_budget, require_fdr=True
    )

    discovery_comparisons: dict[str, dict[str, Any]] = {}
    for strat in ("credo_reg", "global_mi", "global_shap_gain", "global_residual", "random"):
        if strat == "credo_reg":
            ranked = credo_scores
            chosen = chosen_credo
        else:
            ranked = score_channels_baseline_strategy(
                strat, Xb_tr, Zp_tr, yt_tr, cohort.candidates, strata, seed=seed
            )
            chosen = select_channels_within_budget(
                ranked, spec_map, cohort.acquisition_budget, require_fdr=False
            )

        chosen_ids = [c.channel_id for c in chosen]
        relevant_hits = sum(1 for cid in chosen_ids if cid in cohort.ground_truth_relevant_channels)
        precision_at_budget = float(relevant_hits / max(1, len(chosen_ids)))

        z_cols = [col for c in chosen for col in c.columns]
        XZ_tr = pd.concat(
            [Xb_tr.reset_index(drop=True), Zp_tr[z_cols].reset_index(drop=True)], axis=1
        )
        XZ_cal = pd.concat(
            [Xb_cal.reset_index(drop=True), Zp_cal[z_cols].reset_index(drop=True)], axis=1
        )
        XZ_val = pd.concat(
            [Xb_val_e.reset_index(drop=True), Zp_val_e[z_cols].reset_index(drop=True)], axis=1
        )
        XZ_tst = pd.concat(
            [Xb_tst_e.reset_index(drop=True), Zp_tst_e[z_cols].reset_index(drop=True)], axis=1
        )

        mono_arm = fit_single_calibrated_gbdt(
            XZ_tr, yt_tr, XZ_cal, yt_cal, arm_name=f"monolithic_{strat}", seed=seed
        )
        vp_m = mono_arm.predict_proba(XZ_val)
        tp_m = mono_arm.predict_proba(XZ_tst)
        f_m = evaluate_certified_frontier(
            f"monolithic_{strat}", y_val_e, vp_m, y_tst_e, tp_m, cohort.contract
        )
        discovery_comparisons[strat] = {
            "chosen_channels": chosen_ids,
            "hidden_channel_precision": precision_at_budget,
            "pilot_fraction_measured": (
                pilot.fraction_of_cohort_measured if strat == "credo_reg" else 1.0
            ),
            "metrics": f_m.model_dump(),
        }
        if strat in {"global_mi", "global_residual"}:
            arm_test_probs[f"monolithic_{strat}"] = tp_m
            arm_metrics[f"monolithic_{strat}"] = f_m.model_dump()

    # Stage O: Full CREDO (Receipt Repair + Deficit-Selected Channels + OR-of-ANDs Lattice)
    Xb_tr_lat = repaired.X_expanded
    y_tr_lat = repaired.y_expanded
    sw_lat = repaired.sample_weight
    Z_tr_lattice = {
        c.channel_id: Zp_tr.iloc[repaired.source_indices][list(c.columns)].reset_index(drop=True)
        for c in chosen_credo
    }

    Z_cal_lattice = {c.channel_id: Zp_cal[list(c.columns)] for c in chosen_credo}
    Z_val_lattice = {c.channel_id: Zp_val_e[list(c.columns)] for c in chosen_credo}
    Z_tst_lattice = {c.channel_id: Zp_tst_e[list(c.columns)] for c in chosen_credo}

    lattice = ConfirmRescueLattice(max_fpr_budget=cohort.contract.max_fpr_budget, seed=seed)
    lattice.fit(
        Xb_tr_lat,
        Z_tr_lattice,
        y_tr_lat,
        Xb_cal,
        Z_cal_lattice,
        yt_cal,
        Xb_val_e,
        Z_val_lattice,
        y_val_e,
        base_semantic_types=sem_types,
        sample_weight_train=sw_lat,
    )

    vp_credo = lattice.predict_proba(Xb_val_e, Z_val_lattice)
    tp_credo = lattice.predict_proba(Xb_tst_e, Z_tst_lattice)
    bool_credo_tst = lattice.predict_boolean_lattice(Xb_tst_e, Z_tst_lattice)

    credo_frontier = evaluate_certified_frontier(
        "full_credo_lattice", y_val_e, vp_credo, y_tst_e, tp_credo, cohort.contract
    ).model_dump()
    credo_frontier["boolean_lattice_tpr"] = float(
        np.sum((bool_credo_tst == 1) & (y_tst_e == 1)) / max(1, int(y_tst_e.sum()))
    )
    credo_frontier["boolean_lattice_fpr"] = float(
        np.sum((bool_credo_tst == 1) & (y_tst_e == 0)) / max(1, int((y_tst_e == 0).sum()))
    )
    arm_test_probs["full_credo_lattice"] = tp_credo
    arm_metrics["full_credo_lattice"] = credo_frontier

    # Conformal Foveation Evaluation (Stage 1 X_base -> Stage 2 CREDO Lattice)
    s1_cal_p = decoder_arms["calibrated_gbdt"].predict_proba(Xb_cal)
    foveator = BinaryConformalFoveator(
        alpha_neg=0.20, alpha_pos=0.20, stage2_query_cost_per_case=0.15
    )
    foveator.fit(s1_cal_p, yt_cal)
    fov_report = foveator.evaluate_foveation(
        y_tst_e,
        arm_test_probs["calibrated_gbdt"],
        tp_credo,
        cohort.contract,
        stage2_val_probs=vp_credo,
        y_val=y_val_e,
    )

    # Paired Cluster Bootstrap Test: Full CREDO vs Baseline Calibrated GBDT and vs SMOTE GBDT
    boot_vs_base = paired_cluster_bootstrap(
        y_tst_e,
        arm_test_probs["calibrated_gbdt"],
        tp_credo,
        average_precision_score,
        groups=grp_tst_e,
        n_boot=400,
        seed=seed,
    )
    boot_smote_vs_base = paired_cluster_bootstrap(
        y_tst_e,
        arm_test_probs["calibrated_gbdt"],
        arm_test_probs["smote_gbdt"],
        average_precision_score,
        groups=grp_tst_e,
        n_boot=400,
        seed=seed,
    )

    # Mechanism-Subtype Recall Breakdown (for AI4I & Gate 0.5 Prediction 5 & 8!)
    subtype_recall: dict[str, dict[str, float]] = {}
    if cohort.mechanism_subtype is not None:
        sub_tst = cohort.mechanism_subtype[tst_i][eval_tst_mask]
        tau_base_np = select_neyman_pearson_threshold(
            y_val_e,
            decoder_arms["calibrated_gbdt"].predict_proba(Xb_val_e),
            cohort.contract.max_fpr_budget,
        )
        tau_credo_np = select_neyman_pearson_threshold(
            y_val_e, vp_credo, cohort.contract.max_fpr_budget
        )
        base_alert = arm_test_probs["calibrated_gbdt"] >= tau_base_np
        credo_alert = tp_credo >= tau_credo_np
        for tag in sorted(set(sub_tst.tolist())):
            if tag == "none":
                continue
            m_tag = sub_tst == tag
            cnt = int(np.sum(m_tag))
            if cnt >= 2:
                subtype_recall[str(tag)] = {
                    "n_test_events": float(cnt),
                    "baseline_np_recall": float(np.mean(base_alert[m_tag])),
                    "credo_np_recall": float(np.mean(credo_alert[m_tag])),
                }

    branch_coincidence = [
        {
            "channel_id": br.channel_id,
            "kappa0": br.coincidence_report.negative_tail_coincidence_kappa0,
            "projected_joint_fpr": br.coincidence_report.projected_joint_fpr,
            "shifted_bg_fpr": br.coincidence_report.shifted_background_fpr,
            "orthogonality_weight": br.orthogonality_weight,
        }
        for br in lattice.branches_
    ]

    return {
        "cohort_name": cohort.name,
        "split_hash": split.split_hash,
        "n_train": int(tr_i.size),
        "n_test_evaluated": int(y_tst_e.size),
        "n_pos_test": int(y_tst_e.sum()),
        "gate_c": gate_c.model_dump(),
        "receipt_audit": rcpt_audit.model_dump(),
        "four_ceilings": ceil_diag.model_dump(),
        "arms": arm_metrics,
        "discovery_strategies": discovery_comparisons,
        "lattice_coincidence": branch_coincidence,
        "conformal_foveation": fov_report.model_dump(),
        "subtype_np_recall": subtype_recall,
        "paired_bootstrap_credo_vs_gbdt": {
            "delta_auprc": boot_vs_base.delta_point,
            "ci_low": boot_vs_base.ci_low,
            "ci_high": boot_vs_base.ci_high,
            "p_value": boot_vs_base.p_value_two_sided,
            "significant": boot_vs_base.statistically_significant,
        },
        "paired_bootstrap_smote_vs_gbdt": {
            "delta_auprc": boot_smote_vs_base.delta_point,
            "ci_low": boot_smote_vs_base.ci_low,
            "ci_high": boot_smote_vs_base.ci_high,
            "p_value": boot_smote_vs_base.p_value_two_sided,
            "significant": boot_smote_vs_base.statistically_significant,
        },
    }


def run_tier1_saturation_suite(
    out_dir: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run Tier-1 Canonical OpenML/imblearn Imbalanced Suite testing Frontier Saturation & MDE."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for spec in TIER1_PUBLIC_SPECS:
        X, y, sha = load_tier1_dataset(spec, cache_dir=cache_dir)
        df_all = X.copy()
        df_all["__y__"] = y
        split = freeze_split(df_all, "__y__", seed=seed)

        tr_i = np.array(split.train_idx, dtype=int)
        cal_i = np.array(split.cal_idx, dtype=int)
        val_i = np.array(split.val_idx, dtype=int)
        tst_i = np.array(split.test_idx, dtype=int)

        contract = DecisionContract(
            task_name=spec.display_name,
            target_precision=spec.target_precision,
            max_fpr_budget=spec.max_fpr_budget,
        )
        gate_c = gate_c_audit(
            contract, prevalence=float(y[tr_i].mean()), n_pos_test=int(y[tst_i].sum())
        )
        rcpt = audit_receipts(X.iloc[tr_i], y[tr_i], seed=seed)
        ceil = diagnose_four_ceilings(
            contract,
            X.iloc[tr_i],
            y[tr_i],
            X.iloc[cal_i],
            y[cal_i],
            X.iloc[val_i],
            y[val_i],
            rcpt,
            seed=seed,
        )
        arms = train_decoder_saturation_suite(
            X.iloc[tr_i], y[tr_i], X.iloc[cal_i], y[cal_i], seed=seed
        )

        arm_evals: dict[str, Any] = {}
        test_preds: dict[str, np.ndarray] = {}
        for a_name, arm in arms.items():
            vp = arm.predict_proba(X.iloc[val_i])
            tp = arm.predict_proba(X.iloc[tst_i])
            test_preds[a_name] = tp
            arm_evals[a_name] = evaluate_certified_frontier(
                a_name, y[val_i], vp, y[tst_i], tp, contract
            ).model_dump()

        # Paired bootstrap for SMOTE, ADASYN, Cost-Sensitive, Copula vs Calibrated GBDT
        deltas_vs_gbdt: dict[str, Any] = {}
        for comp_name in (
            "smote_gbdt",
            "adasyn_gbdt",
            "cost_sensitive_gbdt",
            "latent_copula_gbdt",
            "soft_avg_ensemble",
        ):
            b_res = paired_cluster_bootstrap(
                y[tst_i],
                test_preds["calibrated_gbdt"],
                test_preds[comp_name],
                average_precision_score,
                n_boot=300,
                seed=seed,
            )
            deltas_vs_gbdt[comp_name] = {
                "delta_auprc": b_res.delta_point,
                "ci_low": b_res.ci_low,
                "ci_high": b_res.ci_high,
                "p_value": b_res.p_value_two_sided,
                "significant_positive": bool(b_res.ci_low > 0.0),
                "ece_degradation_uncal": (
                    arm_evals.get(f"{comp_name}_uncal", arm_evals[comp_name])["ece"]
                    - arm_evals["calibrated_gbdt"]["ece"]
                ),
            }

        rec = {
            "dataset_key": spec.key,
            "display_name": spec.display_name,
            "dataset_sha256": sha,
            "n_rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "prevalence": float(y.mean()),
            "n_pos_test": int(y[tst_i].sum()),
            "gate_c": gate_c.model_dump(),
            "four_ceilings": ceil.model_dump(),
            "arms": arm_evals,
            "decoder_deltas_vs_calibrated_gbdt": deltas_vs_gbdt,
        }
        results.append(rec)

    (out_dir / "tier1_results.json").write_text(json.dumps(results, indent=2))
    _plot_tier1_saturation(results, out_dir / "tier1_saturation_and_thermometer.png")
    return results


def run_gate05_suite(out_dir: Path, seed: int = 42) -> dict[str, Any]:
    """Run Tier-0 Gate 0.5 multi-mechanism + selective-receipt ground-truth benchmark."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cohort = generate_gate05_cohort(n_samples=12000, seed=seed, selective_receipts=True)
    res = run_multi_channel_cohort_benchmark(cohort, seed=seed)
    (out_dir / "gate05_results.json").write_text(json.dumps(res, indent=2))
    return res


def run_tier2_credo_bench(
    out_dir: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run Tier-2 CREDO-Bench across all 4 public multi-channel cohorts + generate figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cohorts = [
        load_tier2_ai4i_maintenance(cache_dir=cache_dir, seed=seed),
        load_tier2_taiwan_credit_panel(cache_dir=cache_dir, seed=seed),
        load_tier2_diabetes130(cache_dir=cache_dir, max_patients=25000, seed=seed),
        load_tier2_credit_card_fraud(cache_dir=cache_dir, max_negatives=60000, seed=seed),
    ]
    results: list[dict[str, Any]] = []
    for cohort in cohorts:
        res = run_multi_channel_cohort_benchmark(cohort, seed=seed)
        results.append(res)

    (out_dir / "tier2_credo_bench_results.json").write_text(json.dumps(results, indent=2))
    _plot_tier2_credo_bench(results, out_dir / "tier2_credo_bench_summary.png")
    return results


def _plot_tier1_saturation(results: list[dict[str, Any]], fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)

    names = [r["display_name"] for r in results]
    y_pos = np.arange(len(names))

    # Panel A: SMOTE vs Calibrated GBDT AUPRC Delta with 95% Bootstrap CIs
    deltas = [r["decoder_deltas_vs_calibrated_gbdt"]["smote_gbdt"]["delta_auprc"] for r in results]
    lo = [
        d - r["decoder_deltas_vs_calibrated_gbdt"]["smote_gbdt"]["ci_low"]
        for d, r in zip(deltas, results, strict=True)
    ]
    hi = [
        r["decoder_deltas_vs_calibrated_gbdt"]["smote_gbdt"]["ci_high"] - d
        for d, r in zip(deltas, results, strict=True)
    ]

    axes[0].axvline(0.0, color="#333333", linestyle="--", linewidth=1.2)
    axes[0].errorbar(
        deltas,
        y_pos,
        xerr=[lo, hi],
        fmt="o",
        color="#1f77b4",
        ecolor="#6baed6",
        elinewidth=2.0,
        capsize=4,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("Test AUPRC Delta (SMOTE+GBDT minus Calibrated GBDT, 95% CI)")
    axes[0].set_title("A. Frontier Saturation: Resampling on Fixed Channel X")
    axes[0].grid(axis="x", alpha=0.3)

    # Panel B: Thermometer Noise (95% Recall CI Half-Width vs Test Positives n+)
    n_pos = [r["n_pos_test"] for r in results]
    hw = [r["gate_c"]["recall_ci_half_width_at_80"] for r in results]
    grid_n = np.linspace(5, max(max(n_pos) * 1.1, 280), 200)
    grid_hw = 1.96 * np.sqrt(0.8 * 0.2 / grid_n)

    axes[1].plot(
        grid_n, grid_hw, color="#d62728", linewidth=1.8, label="Theoretical 95% Recall Half-Width"
    )
    axes[1].axhline(
        0.05, color="#2ca02c", linestyle=":", linewidth=1.4, label="+-0.05 Resolution Gate (n+=246)"
    )
    axes[1].scatter(n_pos, hw, color="#111111", zorder=4)
    for x_v, y_v, nm in zip(n_pos, hw, names, strict=True):
        axes[1].annotate(
            nm.split()[0], (x_v, y_v), xytext=(4, 4), textcoords="offset points", fontsize=8
        )
    axes[1].set_xlabel("Independent Test Positives (n+)")
    axes[1].set_ylabel("Recall 95% CI Half-Width")
    axes[1].set_title("B. Observation 3: Thermometer Noise vs. Sample Positives")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _plot_tier2_credo_bench(results: list[dict[str, Any]], fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=160)

    cohorts = [r["cohort_name"].replace("UCI_", "").replace("_", " ") for r in results]
    x = np.arange(len(cohorts))
    width = 0.20

    base_tpr = [r["arms"]["calibrated_gbdt"]["np_tpr_at_fpr_budget"] for r in results]
    smote_tpr = [r["arms"]["smote_gbdt"]["np_tpr_at_fpr_budget"] for r in results]
    mi_tpr = [r["arms"]["monolithic_global_mi"]["np_tpr_at_fpr_budget"] for r in results]
    credo_tpr = [r["arms"]["full_credo_lattice"]["np_tpr_at_fpr_budget"] for r in results]

    axes[0].bar(x - 1.5 * width, base_tpr, width, label="Baseline GBDT (X_base)", color="#7f7f7f")
    axes[0].bar(x - 0.5 * width, smote_tpr, width, label="SMOTE + GBDT (X_base)", color="#aec7e8")
    axes[0].bar(x + 0.5 * width, mi_tpr, width, label="Global MI Channel + GBDT", color="#ffbb78")
    axes[0].bar(
        x + 1.5 * width, credo_tpr, width, label="Full CREDO (REG + Lattice)", color="#2ca02c"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cohorts, rotation=15, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Certified Sensitivity (TPR) at Strict FPR Budget")
    axes[0].set_title("A. CREDO-Bench: Neyman-Pearson Frontier Shift")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel B: Conformal Passive Abstention vs Two-Stage Conformal Foveation Utility
    u_s1 = [r["conformal_foveation"]["net_utility_stage1_only_per_1k"] for r in results]
    u_fov = [
        r["conformal_foveation"]["net_utility_foveated_net_of_query_cost_per_1k"] for r in results
    ]
    u_full = [
        r["conformal_foveation"]["net_utility_full_stage2_everywhere_per_1k"] for r in results
    ]

    w2 = 0.25
    axes[1].bar(x - w2, u_s1, w2, label="1-Stage Only (X_base)", color="#8c564b")
    axes[1].bar(x, u_fov, w2, label="CREDO Conformal Foveation (Net)", color="#1f77b4")
    axes[1].bar(x + w2, u_full, w2, label="Measure Stage-2 on 100% (Net)", color="#9467bd")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cohorts, rotation=15, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Net Expected Utility per 1,000 Cases")
    axes[1].set_title("B. Resolving Conformal Abstention via Active Foveation")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
