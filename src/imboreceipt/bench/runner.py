"""CREDO-Bench experiment runners, 8-arm knockout ablations, multi-seed campaigns, and plots."""

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
from scipy.stats import wilcoxon
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
from imboreceipt.core.bootstrap import (
    holm_bonferroni_adjust,
    nadeau_bengio_ttest,
    paired_cluster_bootstrap,
)
from imboreceipt.core.types import freeze_split, infer_semantic_types
from imboreceipt.discover.selector import (
    score_channels_baseline_strategy,
    score_channels_credo_reg,
    select_channels_within_budget,
)
from imboreceipt.discover.strata import build_consensus_residual_strata, sample_case_cohort_pilot
from imboreceipt.frontier.baselines import (
    fit_equal_capacity_monolithic_ensemble,
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

ABLATION_LADDER_ORDER: tuple[tuple[str, str], ...] = (
    ("calibrated_gbdt", "A0: Baseline Calibrated GBDT (X_base)"),
    ("smote_gbdt", "A1: SMOTE + GBDT (X_base)"),
    ("receipt_repaired_xbase", "A2: Receipt Repair Only (X_base)"),
    ("monolithic_random", "A3: Random Channels + Monolithic GBDT"),
    ("monolithic_global_mi", "A4: Full-Cohort Global MI + Monolithic GBDT"),
    ("monolithic_credo_reg", "A5: Case-Cohort CREDO REG + Monolithic GBDT"),
    ("monolithic_credo_ensemble", "A5b: CREDO REG + Equal-Capacity 6-Model Ensemble"),
    ("pure_orthogonal_lattice", "A6: CREDO Pure Orthogonal Lattice"),
    ("full_credo_lattice", "A7: Full CREDO (kappa_0-Gated Hybrid Lattice)"),
)


def run_multi_channel_cohort_benchmark(
    cohort: MultiChannelCohort,
    seed: int = 42,
    n_boot: int = 400,
) -> dict[str, Any]:
    """Execute the complete CREDO-Bench evaluation + 8-arm knockout ablation on a cohort."""
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
    Zp_tr, Zp_cal, Zp_val, Zp_tst = cohort.materialize_z_splits(tr_i, cal_i, val_i, tst_i)

    grp_tr = cohort.group_ids[tr_i] if cohort.group_ids is not None else None
    grp_cal = cohort.group_ids[cal_i] if cohort.group_ids is not None else None
    time_tr = cohort.time_order[tr_i] if cohort.time_order is not None else None
    time_cal = cohort.time_order[cal_i] if cohort.time_order is not None else None

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
        Xb_tr,
        yt_tr,
        Xb_cal,
        yt_cal,
        semantic_types=sem_types,
        include_extended_smote=False,
        seed=seed,
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

    # Arm A2: Receipt-Repaired Baseline on X_base alone
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
    strata = build_consensus_residual_strata(
        Xb_tr,
        yt_tr,
        semantic_types=sem_types,
        groups=grp_tr,
        time_order=time_tr,
        seed=seed,
    )
    pilot = sample_case_cohort_pilot(
        yt_tr, strata.oof_consensus_prob, controls_per_positive=4, seed=seed
    )
    spec_map = {c.channel_id: c for c in cohort.candidates}

    credo_scores = score_channels_credo_reg(
        Xb_tr,
        Zp_tr,
        yt_tr,
        cohort.candidates,
        strata,
        pilot,
        cohort.contract,
        groups_train=grp_tr,
        time_order_train=time_tr,
        seed=seed,
    )
    chosen_credo = select_channels_within_budget(
        credo_scores, spec_map, cohort.acquisition_budget, require_fdr=True
    )

    # Single-factor knockout fairness: ALL multi-channel arms (A3..A7) share the EXACT SAME
    # ReceiptRepairer training expansion & weights so Stage R never confounds Stage D or Stage O!
    Xb_tr_rep = repaired.X_expanded
    y_tr_rep = repaired.y_expanded
    sw_rep = repaired.sample_weight

    discovery_comparisons: dict[str, dict[str, Any]] = {}
    for strat in ("credo_reg", "global_mi", "global_shap_gain", "global_residual", "random"):
        if strat == "credo_reg":
            chosen = chosen_credo
        else:
            ranked = score_channels_baseline_strategy(
                strat,
                Xb_tr,
                Zp_tr,
                yt_tr,
                cohort.candidates,
                strata,
                groups_train=grp_tr,
                time_order_train=time_tr,
                seed=seed,
            )
            chosen = select_channels_within_budget(
                ranked, spec_map, cohort.acquisition_budget, require_fdr=False
            )

        chosen_ids = [c.channel_id for c in chosen]
        relevant_hits = sum(1 for cid in chosen_ids if cid in cohort.ground_truth_relevant_channels)
        precision_at_budget = float(relevant_hits / max(1, len(chosen_ids)))

        z_cols = [col for c in chosen for col in c.columns]
        XZ_tr_rep = pd.concat(
            [
                Xb_tr_rep.reset_index(drop=True),
                Zp_tr.iloc[repaired.source_indices][z_cols].reset_index(drop=True),
            ],
            axis=1,
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
            XZ_tr_rep,
            y_tr_rep,
            XZ_cal,
            yt_cal,
            sample_weight=sw_rep,
            arm_name=f"monolithic_{strat}",
            seed=seed,
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
        if strat in {"credo_reg", "global_mi", "global_residual", "random"}:
            arm_test_probs[f"monolithic_{strat}"] = tp_m
            arm_metrics[f"monolithic_{strat}"] = f_m.model_dump()

        if strat == "credo_reg":
            # A5b: Equal-Capacity 6-Learner Monolithic Ensemble on [X_base, Z_CREDO]
            ens_arm = fit_equal_capacity_monolithic_ensemble(
                XZ_tr_rep,
                y_tr_rep,
                XZ_cal,
                yt_cal,
                sample_weight=sw_rep,
                arm_name="monolithic_credo_ensemble",
                seed=seed,
            )
            vp_ens = ens_arm.predict_proba(XZ_val)
            tp_ens = ens_arm.predict_proba(XZ_tst)
            f_ens = evaluate_certified_frontier(
                "monolithic_credo_ensemble", y_val_e, vp_ens, y_tst_e, tp_ens, cohort.contract
            )
            arm_test_probs["monolithic_credo_ensemble"] = tp_ens
            arm_metrics["monolithic_credo_ensemble"] = f_ens.model_dump()

    # Stage O: Evaluate BOTH A6 (Pure Orthogonal Lattice) AND A7 (Full kappa_0-Gated Hybrid Lattice)
    Xb_tr_lat = Xb_tr_rep
    y_tr_lat = y_tr_rep
    sw_lat = sw_rep
    Z_tr_lattice = {
        c.channel_id: Zp_tr.iloc[repaired.source_indices][list(c.columns)].reset_index(drop=True)
        for c in chosen_credo
    }
    Z_cal_lattice = {c.channel_id: Zp_cal[list(c.columns)] for c in chosen_credo}
    Z_val_lattice = {c.channel_id: Zp_val_e[list(c.columns)] for c in chosen_credo}
    Z_tst_lattice = {c.channel_id: Zp_tst_e[list(c.columns)] for c in chosen_credo}

    # A6: Pure Orthogonal Lattice
    orth_lattice = ConfirmRescueLattice(
        max_fpr_budget=cohort.contract.max_fpr_budget,
        mode="pure_orthogonal",
        seed=seed,
    )
    orth_lattice.fit(
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
        groups_cal=grp_cal,
        time_order_cal=time_cal,
    )
    vp_orth = orth_lattice.predict_proba(Xb_val_e, Z_val_lattice)
    tp_orth = orth_lattice.predict_proba(Xb_tst_e, Z_tst_lattice)
    orth_frontier = evaluate_certified_frontier(
        "pure_orthogonal_lattice", y_val_e, vp_orth, y_tst_e, tp_orth, cohort.contract
    ).model_dump()
    arm_test_probs["pure_orthogonal_lattice"] = tp_orth
    arm_metrics["pure_orthogonal_lattice"] = orth_frontier

    # A7: Full CREDO (kappa_0-Gated Hybrid Lattice)
    lattice = ConfirmRescueLattice(
        max_fpr_budget=cohort.contract.max_fpr_budget,
        mode="hybrid_kappa0",
        seed=seed,
    )
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
        groups_cal=grp_cal,
        time_order_cal=time_cal,
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
    credo_frontier["mean_kappa0"] = lattice.mean_kappa0_
    credo_frontier["orthogonal_lattice_weight"] = lattice.orthogonal_lattice_weight_
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

    boot_vs_base = paired_cluster_bootstrap(
        y_tst_e,
        arm_test_probs["calibrated_gbdt"],
        tp_credo,
        average_precision_score,
        groups=grp_tst_e,
        n_boot=n_boot,
        seed=seed,
    )
    boot_smote_vs_base = paired_cluster_bootstrap(
        y_tst_e,
        arm_test_probs["calibrated_gbdt"],
        arm_test_probs["smote_gbdt"],
        average_precision_score,
        groups=grp_tst_e,
        n_boot=n_boot,
        seed=seed,
    )

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
        bool_alert = bool_credo_tst == 1
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
                    "credo_boolean_lattice_recall": float(np.mean(bool_alert[m_tag])),
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
        "seed": seed,
        "disjoint_test_blocks": bool(cohort.disjoint_test_blocks),
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
        "mean_kappa0": lattice.mean_kappa0_,
        "orthogonal_lattice_weight": lattice.orthogonal_lattice_weight_,
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
            X.iloc[tr_i],
            y[tr_i],
            X.iloc[cal_i],
            y[cal_i],
            include_extended_smote=True,
            seed=seed,
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
                n_boot=250,
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
            "seed": seed,
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
    uncapped: bool = False,
) -> list[dict[str, Any]]:
    """Run Tier-2 CREDO-Bench across all 4 public multi-channel cohorts + generate figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cohorts = [
        load_tier2_ai4i_maintenance(cache_dir=cache_dir, seed=seed),
        load_tier2_taiwan_credit_panel(cache_dir=cache_dir, seed=seed),
        load_tier2_diabetes130(
            cache_dir=cache_dir,
            max_patients=None if uncapped else 25000,
            seed=seed,
        ),
        load_tier2_credit_card_fraud(
            cache_dir=cache_dir,
            max_negatives=None if uncapped else 60000,
            seed=seed,
        ),
    ]
    results: list[dict[str, Any]] = []
    for cohort in cohorts:
        res = run_multi_channel_cohort_benchmark(cohort, seed=seed)
        results.append(res)

    (out_dir / "tier2_credo_bench_results.json").write_text(json.dumps(results, indent=2))
    _plot_tier2_credo_bench(results, out_dir / "tier2_credo_bench_summary.png")
    return results


def _agg_mean_sd(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(sd / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def run_multiseed_paper_campaign(
    out_dir: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    uncapped: bool = True,
) -> dict[str, Any]:
    """Execute the full multi-seed Paper A campaign across Tier 0, Tier 1, and Uncapped Tier 2.

    - Tier 0 (Gate 0.5): ``len(seeds)`` independent simulated multi-mechanism + receipt cohorts.
    - Tier 1 (8 OpenML/imblearn benchmarks): ``len(seeds)`` stratified 4-way splits per dataset.
    - Tier 2 (Uncapped Public Multi-Channel Cohorts):
      * AI4I 2020 (10,000 rows) x ``len(seeds)`` stratified splits
      * Taiwan Credit 6m Panel (30,000 rows) x ``len(seeds)`` stratified splits
      * UCI Diabetes 130-US (101,766 rows, 71,518 patients) x ``len(seeds)`` patient-grouped splits
      * ULB Credit Card Fraud (284,807 rows) x ``len(seeds)`` staggered chronological windows
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = len(seeds)

    # 1. Tier-1 Multi-Seed Campaign
    tier1_by_key: dict[str, list[dict[str, Any]]] = {s.key: [] for s in TIER1_PUBLIC_SPECS}
    for s in seeds:
        t1_seed_dir = out_dir / "seeds" / f"tier1_seed{s}"
        seed_runs = run_tier1_saturation_suite(t1_seed_dir, cache_dir=cache_dir, seed=s)
        for rec in seed_runs:
            tier1_by_key[rec["dataset_key"]].append(rec)

    tier1_summary: list[dict[str, Any]] = []
    for spec in TIER1_PUBLIC_SPECS:
        runs = tier1_by_key[spec.key]
        gbdt_ap = [r["arms"]["calibrated_gbdt"]["auprc"] for r in runs]
        smote_ap = [r["arms"]["smote_gbdt"]["auprc"] for r in runs]
        cop_ap = [r["arms"]["latent_copula_gbdt"]["auprc"] for r in runs]
        cs_ap = [r["arms"]["cost_sensitive_gbdt"]["auprc"] for r in runs]
        ens_ap = [r["arms"]["soft_avg_ensemble"]["auprc"] for r in runs]
        smote_deltas = [sm - gb for sm, gb in zip(smote_ap, gbdt_ap, strict=True)]
        ece_deg = [
            r["decoder_deltas_vs_calibrated_gbdt"]["smote_gbdt"]["ece_degradation_uncal"]
            for r in runs
        ]

        try:
            _, p_wilc = wilcoxon(smote_deltas, alternative="greater")
            p_w = float(p_wilc)
        except ValueError:
            p_w = 1.0

        nb_smote = nadeau_bengio_ttest(
            smote_deltas,
            n_test=int(round(runs[0]["n_rows"] * 0.15)),
            n_train=int(round(runs[0]["n_rows"] * 0.55)),
            disjoint_test_blocks=False,
        )
        tier1_summary.append(
            {
                "dataset_key": spec.key,
                "display_name": spec.display_name,
                "n_rows": runs[0]["n_rows"],
                "n_features": runs[0]["n_features"],
                "prevalence": runs[0]["prevalence"],
                "n_pos_test": runs[0]["n_pos_test"],
                "recall_ci_half_width": runs[0]["gate_c"]["recall_ci_half_width_at_80"],
                "modal_regime": max(
                    set(r["four_ceilings"]["bottleneck_regime"] for r in runs),
                    key=lambda reg: sum(
                        1 for r in runs if r["four_ceilings"]["bottleneck_regime"] == reg
                    ),
                ),
                "calibrated_gbdt_auprc": _agg_mean_sd(gbdt_ap),
                "smote_gbdt_auprc": _agg_mean_sd(smote_ap),
                "latent_copula_auprc": _agg_mean_sd(cop_ap),
                "cost_sensitive_auprc": _agg_mean_sd(cs_ap),
                "soft_ensemble_auprc": _agg_mean_sd(ens_ap),
                "smote_minus_gbdt_auprc": _agg_mean_sd(smote_deltas),
                "uncal_smote_ece_degradation": _agg_mean_sd(ece_deg),
                "wilcoxon_one_sided_p": p_w,
                "nadeau_bengio_one_sided_p": nb_smote.p_value_one_sided,
                "nadeau_bengio_ci95": [nb_smote.ci95_low, nb_smote.ci95_high],
            }
        )

    # 2. Tier-0 (Gate 0.5) + Tier-2 (Uncapped Public Multi-Channel) Multi-Seed Campaign
    multi_cohort_runs: dict[str, list[dict[str, Any]]] = {
        "Gate05_Synthetic_MultiMechanism": [],
        "UCI_AI4I_2020_Maintenance": [],
        "UCI_Taiwan_Credit_6m_Panel": [],
        "UCI_Diabetes_130US_Hospitals": [],
        "ULB_CreditCard_Fraud_Chronological": [],
    }

    for w_idx, s in enumerate(seeds):
        g05 = generate_gate05_cohort(n_samples=12000, seed=s, selective_receipts=True)
        multi_cohort_runs["Gate05_Synthetic_MultiMechanism"].append(
            run_multi_channel_cohort_benchmark(g05, seed=s, n_boot=250)
        )

        c_ai4i = load_tier2_ai4i_maintenance(cache_dir=cache_dir, seed=s)
        multi_cohort_runs["UCI_AI4I_2020_Maintenance"].append(
            run_multi_channel_cohort_benchmark(c_ai4i, seed=s, n_boot=250)
        )

        c_tw = load_tier2_taiwan_credit_panel(cache_dir=cache_dir, seed=s)
        multi_cohort_runs["UCI_Taiwan_Credit_6m_Panel"].append(
            run_multi_channel_cohort_benchmark(c_tw, seed=s, n_boot=250)
        )

        c_diab = load_tier2_diabetes130(
            cache_dir=cache_dir,
            max_patients=None if uncapped else 25000,
            seed=s,
        )
        multi_cohort_runs["UCI_Diabetes_130US_Hospitals"].append(
            run_multi_channel_cohort_benchmark(c_diab, seed=s, n_boot=250)
        )

        c_ulb = load_tier2_credit_card_fraud(
            cache_dir=cache_dir,
            max_negatives=None if uncapped else 60000,
            window_idx=w_idx if uncapped else None,
            n_windows=n_seeds,
            seed=s,
        )
        multi_cohort_runs["ULB_CreditCard_Fraud_Chronological"].append(
            run_multi_channel_cohort_benchmark(c_ulb, seed=s, n_boot=250)
        )

    cohort_summaries: list[dict[str, Any]] = []
    for c_name, runs in multi_cohort_runs.items():
        ablation_table: dict[str, Any] = {}
        for arm_key, arm_label in ABLATION_LADDER_ORDER:
            ap_vals = [r["arms"][arm_key]["auprc"] for r in runs]
            np_vals = [r["arms"][arm_key]["np_tpr_at_fpr_budget"] for r in runs]
            rq_vals = [r["arms"][arm_key]["recall_at_target_precision"] for r in runs]
            ut_vals = [r["arms"][arm_key]["net_utility_with_graded_per_1k"] for r in runs]
            ece_vals = [r["arms"][arm_key]["ece"] for r in runs]
            ablation_table[arm_key] = {
                "label": arm_label,
                "auprc": _agg_mean_sd(ap_vals),
                "np_tpr": _agg_mean_sd(np_vals),
                "recall_at_q": _agg_mean_sd(rq_vals),
                "net_utility_graded": _agg_mean_sd(ut_vals),
                "ece": _agg_mean_sd(ece_vals),
            }

        credo_minus_base = [
            r["arms"]["full_credo_lattice"]["auprc"] - r["arms"]["calibrated_gbdt"]["auprc"]
            for r in runs
        ]
        hybrid_minus_orth = [
            r["arms"]["full_credo_lattice"]["auprc"] - r["arms"]["pure_orthogonal_lattice"]["auprc"]
            for r in runs
        ]
        credo_minus_mi = [
            r["arms"]["monolithic_credo_reg"]["auprc"] - r["arms"]["monolithic_global_mi"]["auprc"]
            for r in runs
        ]

        credo_minus_ens = [
            r["arms"]["full_credo_lattice"]["auprc"]
            - r["arms"]["monolithic_credo_ensemble"]["auprc"]
            for r in runs
        ]
        is_indep = bool(
            runs[0].get("disjoint_test_blocks", False)
            or c_name == "Gate05_Synthetic_MultiMechanism"
        )
        nb_credo = nadeau_bengio_ttest(
            credo_minus_base,
            n_test=int(np.mean([r["n_test_evaluated"] for r in runs])),
            n_train=int(np.mean([r["n_train"] for r in runs])),
            disjoint_test_blocks=is_indep,
        )
        reg_prec = _agg_mean_sd(
            [r["discovery_strategies"]["credo_reg"]["hidden_channel_precision"] for r in runs]
        )
        mi_prec = _agg_mean_sd(
            [r["discovery_strategies"]["global_mi"]["hidden_channel_precision"] for r in runs]
        )

        try:
            _, p_credo = wilcoxon(credo_minus_base, alternative="greater")
            p_credo_f = float(p_credo)
        except ValueError:
            p_credo_f = 1.0

        # Subtype recall aggregation if present
        all_sub_tags = sorted({tag for r in runs for tag in r["subtype_np_recall"].keys()})
        subtype_agg: dict[str, Any] = {}
        for tag in all_sub_tags:
            b_rec = [
                r["subtype_np_recall"][tag]["baseline_np_recall"]
                for r in runs
                if tag in r["subtype_np_recall"]
            ]
            c_rec = [
                r["subtype_np_recall"][tag]["credo_np_recall"]
                for r in runs
                if tag in r["subtype_np_recall"]
            ]
            bool_rec = [
                r["subtype_np_recall"][tag]["credo_boolean_lattice_recall"]
                for r in runs
                if tag in r["subtype_np_recall"]
            ]
            subtype_agg[tag] = {
                "baseline_np_recall": _agg_mean_sd(b_rec),
                "credo_hybrid_np_recall": _agg_mean_sd(c_rec),
                "credo_boolean_lattice_recall": _agg_mean_sd(bool_rec),
            }

        cohort_summaries.append(
            {
                "cohort_name": c_name,
                "n_seeds": n_seeds,
                "mean_n_train": int(np.mean([r["n_train"] for r in runs])),
                "mean_n_test": int(np.mean([r["n_test_evaluated"] for r in runs])),
                "mean_n_pos_test": float(np.mean([r["n_pos_test"] for r in runs])),
                "receipt_adi": _agg_mean_sd(
                    [r["receipt_audit"]["ascertainment_distortion_index"] for r in runs]
                ),
                "pilot_fraction_measured": _agg_mean_sd(
                    [
                        r["discovery_strategies"]["credo_reg"]["pilot_fraction_measured"]
                        for r in runs
                    ]
                ),
                "mean_kappa0": _agg_mean_sd([r["mean_kappa0"] for r in runs]),
                "orthogonal_lattice_weight": _agg_mean_sd(
                    [r["orthogonal_lattice_weight"] for r in runs]
                ),
                "stage1_passive_abstention": _agg_mean_sd(
                    [r["conformal_foveation"]["stage1_passive_abstention_rate"] for r in runs]
                ),
                "stage2_sensor_savings": _agg_mean_sd(
                    [r["conformal_foveation"]["stage2_sensor_savings_fraction"] for r in runs]
                ),
                "foveated_net_utility": _agg_mean_sd(
                    [
                        r["conformal_foveation"]["net_utility_foveated_net_of_query_cost_per_1k"]
                        for r in runs
                    ]
                ),
                "stage1_only_net_utility": _agg_mean_sd(
                    [r["conformal_foveation"]["net_utility_stage1_only_per_1k"] for r in runs]
                ),
                "full_stage2_net_utility": _agg_mean_sd(
                    [
                        r["conformal_foveation"]["net_utility_full_stage2_everywhere_per_1k"]
                        for r in runs
                    ]
                ),
                "ablation_ladder": ablation_table,
                "credo_relevant_channel_precision": reg_prec,
                "global_mi_relevant_channel_precision": mi_prec,
                "credo_minus_base_auprc": _agg_mean_sd(credo_minus_base),
                "credo_minus_equal_ensemble_auprc": _agg_mean_sd(credo_minus_ens),
                "hybrid_minus_pure_orth_auprc": _agg_mean_sd(hybrid_minus_orth),
                "credo_reg_minus_global_mi_auprc": _agg_mean_sd(credo_minus_mi),
                "wilcoxon_credo_vs_base_p": p_credo_f,
                "nadeau_bengio_credo_vs_base_p": nb_credo.p_value_one_sided,
                "nadeau_bengio_ci95": [nb_credo.ci95_low, nb_credo.ci95_high],
                "subtype_recall": subtype_agg,
            }
        )

    # Apply Holm-Bonferroni FWER multiplicity adjustment across all 5 multi-channel cohorts!
    raw_nb_ps = [c["nadeau_bengio_credo_vs_base_p"] for c in cohort_summaries]
    holm_ps = holm_bonferroni_adjust(raw_nb_ps)
    for c, hp in zip(cohort_summaries, holm_ps, strict=True):
        c["holm_bonferroni_fwer_p"] = hp

    campaign = {
        "seeds": list(seeds),
        "uncapped": uncapped,
        "tier1_summary": tier1_summary,
        "multi_channel_summary": cohort_summaries,
        "raw_multi_channel_runs": multi_cohort_runs,
    }

    (out_dir / "campaign_summary.json").write_text(
        json.dumps(
            {
                "seeds": list(seeds),
                "uncapped": uncapped,
                "tier1_summary": tier1_summary,
                "multi_channel_summary": cohort_summaries,
            },
            indent=2,
        )
    )

    _plot_multiseed_fig1_tier1(tier1_summary, out_dir / "fig1_tier1_saturation")
    _plot_multiseed_fig2_ablations(cohort_summaries, out_dir / "fig2_ablation_ladder")
    _plot_multiseed_fig3_kappa0_foveation(cohort_summaries, out_dir / "fig3_kappa0_and_foveation")
    _write_latex_and_markdown_tables(tier1_summary, cohort_summaries, len(seeds), out_dir)

    return campaign


def _plot_multiseed_fig1_tier1(tier1_summary: list[dict[str, Any]], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=180)
    names = [r["display_name"] for r in tier1_summary]
    y_pos = np.arange(len(names))

    means = [r["smote_minus_gbdt_auprc"]["mean"] for r in tier1_summary]
    sds = [r["smote_minus_gbdt_auprc"]["sd"] for r in tier1_summary]

    axes[0].axvline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    axes[0].errorbar(
        means,
        y_pos,
        xerr=sds,
        fmt="o",
        color="#1f77b4",
        ecolor="#6baed6",
        elinewidth=2.2,
        capsize=4,
        label="Mean ± 1 SD across seeds",
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("Test AUPRC Delta: SMOTE+GBDT minus Calibrated GBDT")
    axes[0].set_title("A. Frontier Saturation Across 5 Stratified Splits")
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)

    n_pos = [r["n_pos_test"] for r in tier1_summary]
    hw = [r["recall_ci_half_width"] for r in tier1_summary]
    grid_n = np.linspace(4, max(max(n_pos) * 1.15, 270), 200)
    grid_hw = 1.96 * np.sqrt(0.8 * 0.2 / grid_n)

    axes[1].plot(grid_n, grid_hw, color="#d62728", linewidth=1.8, label="Binomial 95% Half-Width")
    axes[1].axhline(
        0.05, color="#2ca02c", linestyle=":", linewidth=1.5, label="±0.05 Target (n+=246)"
    )
    axes[1].scatter(n_pos, hw, color="#111111", zorder=4)
    for xv, yv, nm in zip(n_pos, hw, names, strict=True):
        axes[1].annotate(
            nm.split()[0], (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=8
        )
    axes[1].set_xlabel("Independent Test Positives per Split (n+)")
    axes[1].set_ylabel("Recall 95% CI Half-Width")
    axes[1].set_title("B. Thermometer Resolution vs. Test Positives")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_multiseed_fig2_ablations(cohorts: list[dict[str, Any]], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), dpi=180)
    short_names = [
        c["cohort_name"]
        .replace("Gate05_Synthetic_MultiMechanism", "Gate 0.5 (Truth)")
        .replace("UCI_AI4I_2020_Maintenance", "AI4I Maintenance")
        .replace("UCI_Taiwan_Credit_6m_Panel", "Taiwan Credit 6m")
        .replace("UCI_Diabetes_130US_Hospitals", "Diabetes 130-US (102k)")
        .replace("ULB_CreditCard_Fraud_Chronological", "ULB Fraud (285k)")
        for c in cohorts
    ]
    x = np.arange(len(short_names))
    key_arms = [
        ("calibrated_gbdt", "A0: Base GBDT", "#7f7f7f"),
        ("smote_gbdt", "A1: SMOTE+GBDT", "#aec7e8"),
        ("monolithic_global_mi", "A4: Global MI (100% rows)", "#ffbb78"),
        ("pure_orthogonal_lattice", "A6: Pure Orth Lattice", "#98df8a"),
        ("full_credo_lattice", "A7: Full CREDO (kappa_0 Hybrid)", "#2ca02c"),
    ]
    w = 0.15

    for idx, (akey, alabel, col) in enumerate(key_arms):
        means = [c["ablation_ladder"][akey]["auprc"]["mean"] for c in cohorts]
        sds = [c["ablation_ladder"][akey]["auprc"]["sd"] for c in cohorts]
        axes[0].bar(
            x + (idx - 2.0) * w,
            means,
            w,
            yerr=sds,
            capsize=2.5,
            label=alabel,
            color=col,
            edgecolor="#333333",
            linewidth=0.4,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=16, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Test AUPRC (Mean ± SD Across 5 Splits)")
    axes[0].set_title("A. 8-Arm Knockout Ablation: Test AUPRC")
    axes[0].legend(fontsize=7.8)
    axes[0].grid(axis="y", alpha=0.3)

    for idx, (akey, alabel, col) in enumerate(key_arms):
        means = [c["ablation_ladder"][akey]["np_tpr"]["mean"] for c in cohorts]
        sds = [c["ablation_ladder"][akey]["np_tpr"]["sd"] for c in cohorts]
        axes[1].bar(
            x + (idx - 2.0) * w,
            means,
            w,
            yerr=sds,
            capsize=2.5,
            label=alabel,
            color=col,
            edgecolor="#333333",
            linewidth=0.4,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=16, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Certified Sensitivity (TPR) at Strict FPR Budget")
    axes[1].set_title("B. Neyman-Pearson Frontier Shift (Mean ± SD)")
    axes[1].legend(fontsize=7.8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_multiseed_fig3_kappa0_foveation(cohorts: list[dict[str, Any]], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.3), dpi=180)
    short_names = [
        c["cohort_name"]
        .replace("Gate05_Synthetic_MultiMechanism", "Gate 0.5")
        .replace("UCI_AI4I_2020_Maintenance", "AI4I 2020")
        .replace("UCI_Taiwan_Credit_6m_Panel", "Taiwan Credit")
        .replace("UCI_Diabetes_130US_Hospitals", "Diabetes 130")
        .replace("ULB_CreditCard_Fraud_Chronological", "ULB Fraud")
        for c in cohorts
    ]

    # Panel A: Mean negative-tail coincidence kappa_0 vs learned lattice weight & hybrid lift
    k0_means = [c["mean_kappa0"]["mean"] for c in cohorts]
    orth_minus_mono = [
        c["ablation_ladder"]["pure_orthogonal_lattice"]["auprc"]["mean"]
        - c["ablation_ladder"]["monolithic_credo_reg"]["auprc"]["mean"]
        for c in cohorts
    ]
    hyb_minus_mono = [
        c["ablation_ladder"]["full_credo_lattice"]["auprc"]["mean"]
        - c["ablation_ladder"]["monolithic_credo_reg"]["auprc"]["mean"]
        for c in cohorts
    ]

    axes[0].axhline(0.0, color="#333333", linestyle="--", linewidth=1.1)
    axes[0].axvline(
        1.80, color="#d62728", linestyle=":", linewidth=1.3, label="kappa_0 Pivot (1.80)"
    )
    axes[0].scatter(
        k0_means,
        orth_minus_mono,
        s=70,
        color="#ff7f0e",
        marker="s",
        label="Pure Orth Lattice - Monolithic",
    )
    axes[0].scatter(
        k0_means,
        hyb_minus_mono,
        s=80,
        color="#2ca02c",
        marker="o",
        label="kappa_0-Hybrid Lattice - Monolithic",
    )
    for xv, y_o, y_h, nm in zip(
        k0_means, orth_minus_mono, hyb_minus_mono, short_names, strict=True
    ):
        axes[0].plot([xv, xv], [y_o, y_h], color="#888888", linestyle="-", linewidth=1.0, alpha=0.7)
        axes[0].annotate(nm, (xv, y_h), xytext=(5, 4), textcoords="offset points", fontsize=8.5)
    axes[0].set_xlabel("Calibration Negative-Tail Coincidence Ratio (kappa_0)")
    axes[0].set_ylabel("Test AUPRC Delta vs. Monolithic GBDT")
    axes[0].set_title("A. Prediction 4: kappa_0 Diagnoses Separate vs. Coupled Regimes")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Panel B: Case-Cohort Pilot Savings vs Conformal Foveation Stage-2 Sensor Savings
    x = np.arange(len(short_names))
    pilot_saved = [100.0 * (1.0 - c["pilot_fraction_measured"]["mean"]) for c in cohorts]
    s2_saved = [100.0 * c["stage2_sensor_savings"]["mean"] for c in cohorts]
    w2 = 0.34
    axes[1].bar(
        x - w2 / 2, pilot_saved, w2, label="Discovery Pilot Rows Saved (%)", color="#1f77b4"
    )
    axes[1].bar(
        x + w2 / 2, s2_saved, w2, label="Inference Stage-2 Sensor Saved (%)", color="#9467bd"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=15, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Measurement Budget Saved (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("B. Channel Acquisition Efficiency (Discovery + Deployment)")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_latex_and_markdown_tables(
    tier1: list[dict[str, Any]],
    cohorts: list[dict[str, Any]],
    n_seeds: int,
    out_dir: Path,
) -> None:
    md_lines = [
        f"# Multi-Seed (`{n_seeds}x`) Uncapped `CREDO-Bench` Campaign Report",
        "",
        "## Table 1: Tier-1 Canonical OpenML / `imblearn` Frontier Saturation",
        "",
        (
            "| Dataset | $N$ | Test $n_+$ | Recall 95% Half-Width | Calibrated GBDT AUPRC | "
            "SMOTE+GBDT AUPRC | $\\Delta$AUPRC (Mean $\\pm$ SD) | Nadeau-Bengio $p$ |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in tier1:
        gb = r["calibrated_gbdt_auprc"]
        sm = r["smote_gbdt_auprc"]
        dl = r["smote_minus_gbdt_auprc"]
        md_lines.append(
            f"| {r['display_name']} | {r['n_rows']:,} | {r['n_pos_test']} | "
            f"$\\pm {r['recall_ci_half_width']:.3f}$ | "
            f"${gb['mean']:.4f} \\pm {gb['sd']:.4f}$ | "
            f"${sm['mean']:.4f} \\pm {sm['sd']:.4f}$ | "
            f"${dl['mean']:+.4f} \\pm {dl['sd']:.4f}$ | {r['nadeau_bengio_one_sided_p']:.3f} |"
        )

    md_lines.extend(
        [
            "",
            "## Table 2: Multi-Channel 9-Arm Knockout Ablation Matrix (Test AUPRC, Mean $\\pm$ SD)",
            "",
            (
                "| Cohort | A0: Base GBDT | A1: SMOTE | A2: Receipt Only | A3: Random Ch | "
                "A4: Global MI | A5: CREDO REG Mono | A5b: 6-Model Ens | A6: Pure Orth Lattice | "
                "A7: Full CREDO ($\\kappa_0$ Hybrid) | Holm-Bonferroni $p$ |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for c in cohorts:
        ab = c["ablation_ladder"]
        cells = [
            f"${ab[k]['auprc']['mean']:.4f} \\pm {ab[k]['auprc']['sd']:.4f}$"
            for k, _ in ABLATION_LADDER_ORDER
        ]
        md_lines.append(
            f"| **{c['cohort_name']}** | "
            + " | ".join(cells)
            + f" | {c['holm_bonferroni_fwer_p']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Table 3: Certified Neyman-Pearson Sensitivity, $\\kappa_0$ Routing & Savings",
            "",
            (
                "| Cohort | Base NP TPR | Monolithic MI NP TPR | Pure Orth NP TPR | "
                "Full CREDO NP TPR | Mean $\\kappa_0$ | Learned Orth Weight $w_{\\text{orth}}$ | "
                "Pilot Rows Measured | Stage-2 Sensor Saved |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for c in cohorts:
        ab = c["ablation_ladder"]
        md_lines.append(
            f"| **{c['cohort_name']}** | "
            f"${ab['calibrated_gbdt']['np_tpr']['mean']:.3f}$ | "
            f"${ab['monolithic_global_mi']['np_tpr']['mean']:.3f}$ | "
            f"${ab['pure_orthogonal_lattice']['np_tpr']['mean']:.3f}$ | "
            f"**${ab['full_credo_lattice']['np_tpr']['mean']:.3f}$** | "
            f"${c['mean_kappa0']['mean']:.2f}$ | "
            f"${c['orthogonal_lattice_weight']['mean']:.2f}$ | "
            f"{100.0 * c['pilot_fraction_measured']['mean']:.1f}% | "
            f"{100.0 * c['stage2_sensor_savings']['mean']:.1f}% |"
        )

    (out_dir / "PAPER_CAMPAIGN_REPORT.md").write_text("\n".join(md_lines) + "\n")

    # Emit ready-to-include LaTeX booktabs table for Paper A
    tex = [
        "% Auto-generated by imboreceipt bench-paper-full",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        (
            "\\caption{Multi-seed (5$\\times$) knockout ablation Test AUPRC across Tier-0 "
            "and uncapped Tier-2 public cohorts.}"
        ),
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        (
            "Cohort & A0 Base & A1 SMOTE & A4 Global MI & A5 REG Mono & "
            "A6 Orth Lattice & A7 Full CREDO \\\\"
        ),
        "\\midrule",
    ]
    for c in cohorts:
        ab = c["ablation_ladder"]
        nm = c["cohort_name"].replace("_", "\\_")
        row_keys = (
            "calibrated_gbdt",
            "smote_gbdt",
            "monolithic_global_mi",
            "monolithic_credo_reg",
            "pure_orthogonal_lattice",
            "full_credo_lattice",
        )
        vals = [f"{ab[k]['auprc']['mean']:.3f}$\\pm${ab[k]['auprc']['sd']:.3f}" for k in row_keys]
        tex.append(f"{nm} & " + " & ".join(vals) + " \\\\")
    tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (out_dir / "latex_tables.tex").write_text("\n".join(tex))


def _plot_tier1_saturation(results: list[dict[str, Any]], fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=160)
    names = [r["display_name"] for r in results]
    y_pos = np.arange(len(names))

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

    n_pos = [r["n_pos_test"] for r in results]
    hw = [r["gate_c"]["recall_ci_half_width_at_80"] for r in results]
    grid_n = np.linspace(5, max(max(n_pos) * 1.1, 280), 200)
    grid_hw = 1.96 * np.sqrt(0.8 * 0.2 / grid_n)

    axes[1].plot(
        grid_n, grid_hw, color="#d62728", linewidth=1.8, label="Theoretical 95% Half-Width"
    )
    axes[1].axhline(
        0.05, color="#2ca02c", linestyle=":", linewidth=1.4, label="±0.05 Gate (n+=246)"
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
    width = 0.18

    base_tpr = [r["arms"]["calibrated_gbdt"]["np_tpr_at_fpr_budget"] for r in results]
    smote_tpr = [r["arms"]["smote_gbdt"]["np_tpr_at_fpr_budget"] for r in results]
    mi_tpr = [r["arms"]["monolithic_global_mi"]["np_tpr_at_fpr_budget"] for r in results]
    orth_tpr = [r["arms"]["pure_orthogonal_lattice"]["np_tpr_at_fpr_budget"] for r in results]
    credo_tpr = [r["arms"]["full_credo_lattice"]["np_tpr_at_fpr_budget"] for r in results]

    axes[0].bar(x - 2 * width, base_tpr, width, label="Base GBDT", color="#7f7f7f")
    axes[0].bar(x - width, smote_tpr, width, label="SMOTE+GBDT", color="#aec7e8")
    axes[0].bar(x, mi_tpr, width, label="Global MI + GBDT", color="#ffbb78")
    axes[0].bar(x + width, orth_tpr, width, label="Pure Orth Lattice", color="#98df8a")
    axes[0].bar(x + 2 * width, credo_tpr, width, label="Full CREDO (Hybrid)", color="#2ca02c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cohorts, rotation=15, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Certified Sensitivity (TPR) at Strict FPR Budget")
    axes[0].set_title("A. CREDO-Bench: Neyman-Pearson Frontier Shift")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

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
