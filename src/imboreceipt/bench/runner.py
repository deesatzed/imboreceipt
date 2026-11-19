"""CREDO-Bench experiment runners, 8-arm knockout ablations, multi-seed campaigns, and plots."""

from __future__ import annotations

import gc
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

from imboreceipt.bench.adversarial_sims import run_adversarial_simulation_suite
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
from imboreceipt.bench.wave2_datasets import (
    load_wave2_home_credit_relational,
    load_wave2_ieee_cis_fraud,
    load_wave2_neurips_baf,
    load_wave2_scania_aps,
)
from imboreceipt.contract.decision import DecisionContract, gate_c_audit
from imboreceipt.core.bootstrap import (
    holm_bonferroni_adjust,
    nadeau_bengio_ttest,
    paired_cluster_bootstrap,
)
from imboreceipt.core.types import freeze_split, infer_semantic_types
from imboreceipt.discover.selector import (
    score_channels_baseline_strategy,
    score_channels_conditional_gain,
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
    CertifiedFrontierMetrics,
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
    ("monolithic_random", "A3: Random Channels + Single GBDT"),
    ("monolithic_global_mi", "A4: Full-Cohort Marginal MI (100% rows) + Single GBDT"),
    ("monolithic_pilot_mi", "A4p: Stratified-Pilot Marginal MI + Single GBDT"),
    (
        "monolithic_unmatched_case_control_mi",
        "A4b: Unmatched Case-Control Marginal MI + Single GBDT",
    ),
    (
        "monolithic_unmatched_pilot_cmi",
        "A4u: Unmatched-Pilot Conditional Gain G_m + Single GBDT",
    ),
    (
        "monolithic_matched_pilot_cmi",
        "A5: Stratified-Pilot Conditional Gain G_m + Single GBDT (Primary)",
    ),
    ("monolithic_credo_reg", "A5r: Stratified-Pilot Composite REG + Single GBDT"),
    (
        "monolithic_full_cohort_cmi",
        "A5f: Full-Cohort Conditional Gain G_m (100% rows) + Single GBDT",
    ),
    ("monolithic_credo_ensemble", "A6e: Selected Channels + Equal-Capacity 6-Learner Ensemble"),
    ("pure_orthogonal_lattice", "A6o: Selected Channels + Pure Orthogonal Lattice"),
    ("full_credo_lattice", "A7: Selected Channels + Calibrated kappa_0-Hybrid Lattice"),
    ("unrestricted_all_native_gbdt", "Ref: Unrestricted All-Native-Features Single GBDT"),
)


def _compute_observation_propensity_index(
    Xb: pd.DataFrame,
    receipt_cols: list[str] | None,
) -> np.ndarray | None:
    """Construct standardized observed-testing/intake indicator index on X_base when present."""
    if not receipt_cols:
        return None
    vecs: list[np.ndarray] = []
    for col in receipt_cols:
        if col not in Xb.columns:
            continue
        s = Xb[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            v = pd.to_numeric(s, errors="coerce")
            med = float(v.median()) if v.notna().any() else 0.0
            arr = v.fillna(med).to_numpy(dtype=float)
        else:
            # Non-missing indicator for administrative/clinical intake fields
            arr = (~(s.isna() | (s.astype(str) == "__MISSING__"))).to_numpy(dtype=float)
        sd = float(np.std(arr))
        if sd > 1e-8:
            vecs.append((arr - float(np.mean(arr))) / sd)
    if not vecs:
        return None
    return np.mean(np.column_stack(vecs), axis=1)


def run_multi_channel_cohort_benchmark(
    cohort: MultiChannelCohort,
    seed: int = 42,
    n_boot: int = 400,
    include_complex_fusion: bool = True,
) -> dict[str, Any]:
    """Execute the reconstructed single-factor CREDO-Bench evaluation on a cohort."""
    df_meta = cohort.X_base.copy()
    df_meta["__y_split__"] = cohort.y_tilde
    group_col: str | None = None
    time_col: str | None = None
    if cohort.group_ids is not None:
        df_meta["__grp__"] = cohort.group_ids
        group_col = "__grp__"
    if cohort.time_order is not None:
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
    # Outcome-independent eligibility restriction when competing_censor_mask is present
    eval_cal_mask = np.ones(cal_i.size, dtype=bool)
    eval_val_mask = np.ones(val_i.size, dtype=bool)
    eval_tst_mask = np.ones(tst_i.size, dtype=bool)
    if cohort.competing_censor_mask is not None:
        eval_cal_mask = ~cohort.competing_censor_mask[cal_i]
        eval_val_mask = ~cohort.competing_censor_mask[val_i]
        eval_tst_mask = ~cohort.competing_censor_mask[tst_i]

    Xb_cal = Xb_cal.iloc[eval_cal_mask]
    Zp_cal = Zp_cal.iloc[eval_cal_mask]
    yt_cal = cohort.y_tilde[cal_i][eval_cal_mask]
    grp_cal = grp_cal[eval_cal_mask] if grp_cal is not None else None
    time_cal = time_cal[eval_cal_mask] if time_cal is not None else None

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

    # Stage E: Exploratory Bottleneck Diagnostic Panel & Baseline Arms on fixed X_base
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
    arm_val_probs: dict[str, np.ndarray] = {}

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
        arm_val_probs[name] = vp
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
    arm_val_probs["receipt_repaired_xbase"] = vp_rcpt
    arm_test_probs["receipt_repaired_xbase"] = tp_rcpt
    arm_metrics["receipt_repaired_xbase"] = evaluate_certified_frontier(
        "receipt_repaired_xbase", y_val_e, vp_rcpt, y_tst_e, tp_rcpt, cohort.contract
    ).model_dump()

    # Stage D: Consensus Strata + Score-Stratified HT Pilot vs Equal-Size Unmatched HT Pilot
    strata = build_consensus_residual_strata(
        Xb_tr,
        yt_tr,
        semantic_types=sem_types,
        groups=grp_tr,
        time_order=time_tr,
        seed=seed,
    )
    obs_prop_tr = _compute_observation_propensity_index(Xb_tr, cohort.receipt_cols)
    pilot_matched = sample_case_cohort_pilot(
        yt_tr,
        strata.oof_consensus_prob,
        oof_valid_mask=strata.oof_valid_mask,
        controls_per_positive=None,
        matching_mode="score_stratified",
        observation_propensity=obs_prop_tr,
        seed=seed,
    )
    pilot_unmatched = sample_case_cohort_pilot(
        yt_tr,
        strata.oof_consensus_prob,
        oof_valid_mask=strata.oof_valid_mask,
        controls_per_positive=None,
        matching_mode="unmatched_random",
        seed=seed,
    )
    spec_map = {c.channel_id: c for c in cohort.candidates}

    # Score candidate channels under all single-factor discovery designs
    matched_cmi_scores = score_channels_conditional_gain(
        Xb_tr,
        Zp_tr,
        yt_tr,
        cohort.candidates,
        strata,
        pilot_matched,
        base_semantic_types=sem_types,
        groups_train=grp_tr,
        time_order_train=time_tr,
        strategy_name="matched_pilot_cmi",
        seed=seed,
    )
    unmatched_cmi_scores = score_channels_conditional_gain(
        Xb_tr,
        Zp_tr,
        yt_tr,
        cohort.candidates,
        strata,
        pilot_unmatched,
        base_semantic_types=sem_types,
        groups_train=grp_tr,
        time_order_train=time_tr,
        strategy_name="unmatched_pilot_cmi",
        seed=seed,
    )
    credo_reg_scores = score_channels_credo_reg(
        Xb_tr,
        Zp_tr,
        yt_tr,
        cohort.candidates,
        strata,
        pilot_matched,
        cohort.contract,
        base_semantic_types=sem_types,
        groups_train=grp_tr,
        time_order_train=time_tr,
        seed=seed,
    )

    chosen_matched_cmi = select_channels_within_budget(
        matched_cmi_scores, spec_map, cohort.acquisition_budget, require_fdr=True
    )
    chosen_unmatched_cmi = select_channels_within_budget(
        unmatched_cmi_scores, spec_map, cohort.acquisition_budget, require_fdr=True
    )
    chosen_credo_reg = select_channels_within_budget(
        credo_reg_scores, spec_map, cohort.acquisition_budget, require_fdr=True
    )

    Xb_tr_rep = repaired.X_expanded
    y_tr_rep = repaired.y_expanded
    sw_rep = repaired.sample_weight

    # Exact deterministic cache of Single-GBDT predictions keyed by canonical selected channel tuple
    subset_gbdt_cache: dict[
        tuple[str, ...], tuple[np.ndarray, np.ndarray, CertifiedFrontierMetrics]
    ] = {}

    def _fit_or_reuse_single_gbdt(
        chosen_specs: list[Any],
        arm_label: str,
    ) -> tuple[np.ndarray, np.ndarray, CertifiedFrontierMetrics]:
        canon_ids = tuple(sorted(c.channel_id for c in chosen_specs))
        if canon_ids in subset_gbdt_cache:
            vp_cached, tp_cached, fm_cached = subset_gbdt_cache[canon_ids]
            return vp_cached, tp_cached, fm_cached.model_copy(update={"arm_name": arm_label})

        if not canon_ids:
            fm_empty = evaluate_certified_frontier(
                arm_label, y_val_e, vp_rcpt, y_tst_e, tp_rcpt, cohort.contract
            )
            subset_gbdt_cache[canon_ids] = (vp_rcpt, tp_rcpt, fm_empty)
            return vp_rcpt, tp_rcpt, fm_empty

        z_cols = [col for cid in canon_ids for col in spec_map[cid].columns]
        XZ_tr_r = pd.concat(
            [
                Xb_tr_rep.reset_index(drop=True),
                Zp_tr.iloc[repaired.source_indices][z_cols].reset_index(drop=True),
            ],
            axis=1,
        )
        XZ_c = pd.concat(
            [Xb_cal.reset_index(drop=True), Zp_cal[z_cols].reset_index(drop=True)], axis=1
        )
        XZ_v = pd.concat(
            [Xb_val_e.reset_index(drop=True), Zp_val_e[z_cols].reset_index(drop=True)], axis=1
        )
        XZ_t = pd.concat(
            [Xb_tst_e.reset_index(drop=True), Zp_tst_e[z_cols].reset_index(drop=True)], axis=1
        )

        m_arm = fit_single_calibrated_gbdt(
            XZ_tr_r,
            y_tr_rep,
            XZ_c,
            yt_cal,
            semantic_types=sem_types,
            sample_weight=sw_rep,
            arm_name=arm_label,
            seed=seed,
        )
        vp_m = m_arm.predict_proba(XZ_v)
        tp_m = m_arm.predict_proba(XZ_t)
        f_m = evaluate_certified_frontier(arm_label, y_val_e, vp_m, y_tst_e, tp_m, cohort.contract)
        subset_gbdt_cache[canon_ids] = (vp_m, tp_m, f_m)
        return vp_m, tp_m, f_m

    discovery_comparisons: dict[str, dict[str, Any]] = {}
    strategy_specs: list[tuple[str, list[Any], float, list[Any]]] = [
        (
            "matched_pilot_cmi",
            chosen_matched_cmi,
            pilot_matched.fraction_of_cohort_measured,
            matched_cmi_scores,
        ),
        (
            "unmatched_pilot_cmi",
            chosen_unmatched_cmi,
            pilot_unmatched.fraction_of_cohort_measured,
            unmatched_cmi_scores,
        ),
        (
            "credo_reg",
            chosen_credo_reg,
            pilot_matched.fraction_of_cohort_measured,
            credo_reg_scores,
        ),
    ]
    for b_strat in (
        "pilot_mi",
        "global_mi",
        "unmatched_case_control_mi",
        "full_cohort_cmi",
        "global_shap_gain",
        "global_residual",
        "random",
    ):
        b_ranked = score_channels_baseline_strategy(
            b_strat,
            Xb_tr,
            Zp_tr,
            yt_tr,
            cohort.candidates,
            strata,
            pilot=pilot_matched,
            base_semantic_types=sem_types,
            groups_train=grp_tr,
            time_order_train=time_tr,
            seed=seed,
        )
        b_chosen = select_channels_within_budget(
            b_ranked,
            spec_map,
            cohort.acquisition_budget,
            require_fdr=(b_strat == "full_cohort_cmi"),
        )
        p_frac = (
            pilot_matched.fraction_of_cohort_measured
            if b_strat in {"pilot_mi", "unmatched_case_control_mi"}
            else 1.0
        )
        strategy_specs.append((b_strat, b_chosen, p_frac, b_ranked))

    for strat_name, chosen_list, p_frac, raw_ranked in strategy_specs:
        chosen_ids = [c.channel_id for c in chosen_list]
        rel_hits = sum(1 for cid in chosen_ids if cid in cohort.ground_truth_relevant_channels)
        proxy_hits = sum(
            1
            for cid in chosen_ids
            if spec_map[cid].mechanism_tag in {"redundant_proxy", "redundant", "confounded_fp"}
        )
        decoy_hits = sum(
            1 for cid in chosen_ids if spec_map[cid].mechanism_tag in {"decoy", "noise_decoy"}
        )
        prec_at_b = float(rel_hits / max(1, len(chosen_ids)))

        arm_key = f"monolithic_{strat_name}"
        vp_m, tp_m, f_m = _fit_or_reuse_single_gbdt(chosen_list, arm_key)

        discovery_comparisons[strat_name] = {
            "chosen_channels": chosen_ids,
            "hidden_channel_precision": prec_at_b,
            "false_proxy_selection_rate": float(proxy_hits / max(1, len(chosen_ids))),
            "false_decoy_selection_rate": float(decoy_hits / max(1, len(chosen_ids))),
            "pilot_fraction_measured": p_frac,
            "temporal_cv_used": bool(raw_ranked[0].temporal_cv_used) if raw_ranked else False,
            "channel_scores": [
                {
                    "channel_id": r.channel_id,
                    "raw_score": r.raw_score,
                    "conditional_logloss_gain_nats": r.conditional_logloss_gain_nats,
                    "p_value": r.p_value,
                    "fdr_survives": r.fdr_survives,
                }
                for r in raw_ranked
            ],
            "metrics": f_m.model_dump(),
        }
        arm_val_probs[arm_key] = vp_m
        arm_test_probs[arm_key] = tp_m
        arm_metrics[arm_key] = f_m.model_dump()

    # Unrestricted All-Native-Features Reference Model (100% of native channels on 100% of rows)
    native_cands = [
        c
        for c in cohort.candidates
        if c.mechanism_tag not in {"redundant_proxy", "redundant", "decoy", "noise_decoy"}
    ]
    vp_unref, tp_unref, f_unref = _fit_or_reuse_single_gbdt(
        native_cands, "unrestricted_all_native_gbdt"
    )
    arm_val_probs["unrestricted_all_native_gbdt"] = vp_unref
    arm_test_probs["unrestricted_all_native_gbdt"] = tp_unref
    arm_metrics["unrestricted_all_native_gbdt"] = f_unref.model_dump()

    # Downstream complexity comparisons on the primary selected channels (matched_pilot_cmi)
    primary_selected = chosen_matched_cmi
    z_cols_prim = [col for c in primary_selected for col in c.columns]

    if include_complex_fusion and z_cols_prim:
        XZ_tr_prim = pd.concat(
            [
                Xb_tr_rep.reset_index(drop=True),
                Zp_tr.iloc[repaired.source_indices][z_cols_prim].reset_index(drop=True),
            ],
            axis=1,
        )
        XZ_cal_prim = pd.concat(
            [Xb_cal.reset_index(drop=True), Zp_cal[z_cols_prim].reset_index(drop=True)], axis=1
        )
        XZ_val_prim = pd.concat(
            [Xb_val_e.reset_index(drop=True), Zp_val_e[z_cols_prim].reset_index(drop=True)], axis=1
        )
        XZ_tst_prim = pd.concat(
            [Xb_tst_e.reset_index(drop=True), Zp_tst_e[z_cols_prim].reset_index(drop=True)], axis=1
        )

        ens_arm = fit_equal_capacity_monolithic_ensemble(
            XZ_tr_prim,
            y_tr_rep,
            XZ_cal_prim,
            yt_cal,
            semantic_types=sem_types,
            sample_weight=sw_rep,
            arm_name="monolithic_credo_ensemble",
            seed=seed,
        )
        vp_ens = ens_arm.predict_proba(XZ_val_prim)
        tp_ens = ens_arm.predict_proba(XZ_tst_prim)
        f_ens = evaluate_certified_frontier(
            "monolithic_credo_ensemble", y_val_e, vp_ens, y_tst_e, tp_ens, cohort.contract
        )
        arm_val_probs["monolithic_credo_ensemble"] = vp_ens
        arm_test_probs["monolithic_credo_ensemble"] = tp_ens
        arm_metrics["monolithic_credo_ensemble"] = f_ens.model_dump()

        Z_tr_lattice = {
            c.channel_id: Zp_tr.iloc[repaired.source_indices][list(c.columns)].reset_index(
                drop=True
            )
            for c in primary_selected
        }
        Z_cal_lattice = {c.channel_id: Zp_cal[list(c.columns)] for c in primary_selected}
        Z_val_lattice = {c.channel_id: Zp_val_e[list(c.columns)] for c in primary_selected}
        Z_tst_lattice = {c.channel_id: Zp_tst_e[list(c.columns)] for c in primary_selected}

        orth_lattice = ConfirmRescueLattice(
            max_fpr_budget=cohort.contract.max_fpr_budget,
            mode="pure_orthogonal",
            seed=seed,
        ).fit(
            Xb_tr_rep,
            Z_tr_lattice,
            y_tr_rep,
            Xb_cal,
            Z_cal_lattice,
            yt_cal,
            Xb_val_e,
            Z_val_lattice,
            y_val_e,
            base_semantic_types=sem_types,
            sample_weight_train=sw_rep,
            groups_cal=grp_cal,
            time_order_cal=time_cal,
        )
        vp_orth = orth_lattice.predict_proba(Xb_val_e, Z_val_lattice)
        tp_orth = orth_lattice.predict_proba(Xb_tst_e, Z_tst_lattice)
        arm_val_probs["pure_orthogonal_lattice"] = vp_orth
        arm_test_probs["pure_orthogonal_lattice"] = tp_orth
        arm_metrics["pure_orthogonal_lattice"] = evaluate_certified_frontier(
            "pure_orthogonal_lattice", y_val_e, vp_orth, y_tst_e, tp_orth, cohort.contract
        ).model_dump()

        lattice = ConfirmRescueLattice(
            max_fpr_budget=cohort.contract.max_fpr_budget,
            mode="hybrid_kappa0",
            seed=seed,
        ).fit(
            Xb_tr_rep,
            Z_tr_lattice,
            y_tr_rep,
            Xb_cal,
            Z_cal_lattice,
            yt_cal,
            Xb_val_e,
            Z_val_lattice,
            y_val_e,
            base_semantic_types=sem_types,
            sample_weight_train=sw_rep,
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
        arm_val_probs["full_credo_lattice"] = vp_credo
        arm_test_probs["full_credo_lattice"] = tp_credo
        arm_metrics["full_credo_lattice"] = credo_frontier
        mean_k0 = lattice.mean_kappa0_
        orth_w = lattice.orthogonal_lattice_weight_
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
    else:
        # Populate secondary complexity keys from primary Single-GBDT arm when skipped
        vp_credo = arm_val_probs["monolithic_matched_pilot_cmi"]
        tp_credo = arm_test_probs["monolithic_matched_pilot_cmi"]
        bool_credo_tst = np.zeros(y_tst_e.size, dtype=int)
        for sec_k in ("monolithic_credo_ensemble", "pure_orthogonal_lattice", "full_credo_lattice"):
            arm_val_probs[sec_k] = vp_credo
            arm_test_probs[sec_k] = tp_credo
            arm_metrics[sec_k] = dict(arm_metrics["monolithic_matched_pilot_cmi"], arm_name=sec_k)
        mean_k0 = 1.0
        orth_w = 0.0
        branch_coincidence = []

    # Runtime Selective Acquisition (All 6 Equal-Budget Policies on Primary Single GBDT A5)
    s1_cal_p = decoder_arms["calibrated_gbdt"].predict_proba(Xb_cal)
    foveator = BinaryConformalFoveator(
        alpha_neg=0.20, alpha_pos=0.20, stage2_query_cost_per_case=0.15
    ).fit(s1_cal_p, yt_cal)

    vp_primary_s2 = arm_val_probs["monolithic_matched_pilot_cmi"]
    tp_primary_s2 = arm_test_probs["monolithic_matched_pilot_cmi"]
    fov_report = foveator.evaluate_foveation(
        y_tst_e,
        arm_test_probs["calibrated_gbdt"],
        tp_primary_s2,
        cohort.contract,
        stage1_val_probs=arm_val_probs["calibrated_gbdt"],
        stage2_val_probs=vp_primary_s2,
        y_val=y_val_e,
    )

    boot_vs_base = paired_cluster_bootstrap(
        y_tst_e,
        arm_test_probs["calibrated_gbdt"],
        tp_primary_s2,
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
            arm_val_probs["calibrated_gbdt"],
            cohort.contract.max_fpr_budget,
        )
        tau_prim_np = select_neyman_pearson_threshold(
            y_val_e, vp_primary_s2, cohort.contract.max_fpr_budget
        )
        base_alert = arm_test_probs["calibrated_gbdt"] > tau_base_np
        credo_alert = tp_primary_s2 > tau_prim_np
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

    # Explicit Total Measurement Cost Accounting (Review Sec. 5.3)
    m_cand = len(cohort.candidates)
    b_sel = len(primary_selected)
    n_pilot = int(pilot_matched.sampled_indices.size)
    n_dev_downstream = int(tr_i.size + Xb_cal.shape[0] + Xb_val_e.shape[0])
    cost_accounting = {
        "n_candidate_channels": m_cand,
        "n_selected_channels": b_sel,
        "discovery_pilot_channel_accesses": n_pilot * m_cand,
        "discovery_full_table_channel_accesses": int(tr_i.size) * m_cand,
        "discovery_access_savings_fraction": float(1.0 - (n_pilot / max(1, int(tr_i.size)))),
        "downstream_train_cal_val_selected_channel_accesses": n_dev_downstream * b_sel,
        "downstream_unrestricted_all_channel_accesses": n_dev_downstream * len(native_cands),
        "deployment_selective_channel_accesses_per_1k": float(
            1000.0 * fov_report.foveated_stage2_query_rate * b_sel
        ),
        "deployment_all_selected_channel_accesses_per_1k": float(1000.0 * b_sel),
        "deployment_unrestricted_all_native_accesses_per_1k": float(1000.0 * len(native_cands)),
    }

    return {
        "cohort_name": cohort.name,
        "dataset_provenance": cohort.dataset_provenance,
        "include_synthetic_proxies": bool(cohort.include_synthetic_proxies),
        "eligibility_audit": {
            "n_raw_rows_before_eligibility": cohort.n_raw_rows_before_eligibility,
            "n_excluded_ineligible": cohort.n_excluded_ineligible,
            "n_excluded_ineligible_positives": cohort.n_excluded_ineligible_positives,
            "eligibility_note": cohort.eligibility_note,
        },
        "seed": seed,
        "disjoint_test_blocks": bool(cohort.disjoint_test_blocks),
        "split_protocol": {
            "split_hash": split.split_hash,
            "grouped_by": split.grouped_by,
            "ordered_by": split.ordered_by,
            "temporal_burnin_fraction": strata.temporal_burnin_fraction,
            "temporal_cv_feasible": strata.temporal_cv_feasible,
        },
        "split_hash": split.split_hash,
        "n_train": int(tr_i.size),
        "n_cal_evaluated": int(yt_cal.size),
        "n_val_evaluated": int(y_val_e.size),
        "n_test_evaluated": int(y_tst_e.size),
        "n_pos_test": int(y_tst_e.sum()),
        "gate_c": gate_c.model_dump(),
        "receipt_audit": rcpt_audit.model_dump(),
        "four_ceilings": ceil_diag.model_dump(),
        "cost_accounting": cost_accounting,
        "arms": arm_metrics,
        "discovery_strategies": discovery_comparisons,
        "lattice_coincidence": branch_coincidence,
        "mean_kappa0": mean_k0,
        "orthogonal_lattice_weight": orth_w,
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
            "easy_ensemble",
            "rusboost",
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
        eec_ap = [r["arms"]["easy_ensemble"]["auprc"] for r in runs]
        rus_ap = [r["arms"]["rusboost"]["auprc"] for r in runs]
        cop_ap = [r["arms"]["latent_copula_gbdt"]["auprc"] for r in runs]
        cs_ap = [r["arms"]["cost_sensitive_gbdt"]["auprc"] for r in runs]
        ens_ap = [r["arms"]["soft_avg_ensemble"]["auprc"] for r in runs]
        smote_deltas = [sm - gb for sm, gb in zip(smote_ap, gbdt_ap, strict=True)]
        eec_deltas = [ee - gb for ee, gb in zip(eec_ap, gbdt_ap, strict=True)]
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
        nb_eec = nadeau_bengio_ttest(
            eec_deltas,
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
                "easy_ensemble_auprc": _agg_mean_sd(eec_ap),
                "rusboost_auprc": _agg_mean_sd(rus_ap),
                "latent_copula_auprc": _agg_mean_sd(cop_ap),
                "cost_sensitive_auprc": _agg_mean_sd(cs_ap),
                "soft_ensemble_auprc": _agg_mean_sd(ens_ap),
                "smote_minus_gbdt_auprc": _agg_mean_sd(smote_deltas),
                "easy_ensemble_minus_gbdt_auprc": _agg_mean_sd(eec_deltas),
                "uncal_smote_ece_degradation": _agg_mean_sd(ece_deg),
                "wilcoxon_one_sided_p": p_w,
                "nadeau_bengio_one_sided_p": nb_smote.p_value_one_sided,
                "easy_ensemble_nb_one_sided_p": nb_eec.p_value_one_sided,
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

    cohort_summaries = _summarize_multi_cohort_runs(multi_cohort_runs, n_seeds)

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


def _summarize_multi_cohort_runs(
    multi_cohort_runs: dict[str, list[dict[str, Any]]],
    n_seeds: int,
) -> list[dict[str, Any]]:
    """Aggregate per-seed cohort benchmark receipts with conservative NB & Holm (1979) FWER."""
    cohort_summaries: list[dict[str, Any]] = []
    for c_name, runs in multi_cohort_runs.items():
        if not runs:
            continue
        ablation_table: dict[str, Any] = {}
        for arm_key, arm_label in ABLATION_LADDER_ORDER:
            ap_vals = [r["arms"][arm_key]["auprc"] for r in runs]
            np_vals = [r["arms"][arm_key]["np_tpr_at_fpr_budget"] for r in runs]
            fpr_vals = [r["arms"][arm_key]["np_realized_fpr"] for r in runs]
            cert_vals = [float(r["arms"][arm_key].get("np_certified", True)) for r in runs]
            rq_vals = [r["arms"][arm_key]["recall_at_target_precision"] for r in runs]
            ut_vals = [r["arms"][arm_key]["net_utility_with_graded_per_1k"] for r in runs]
            ece_vals = [r["arms"][arm_key]["ece"] for r in runs]
            ablation_table[arm_key] = {
                "label": arm_label,
                "auprc": _agg_mean_sd(ap_vals),
                "np_tpr": _agg_mean_sd(np_vals),
                "np_realized_fpr": _agg_mean_sd(fpr_vals),
                "np_certified_rate": float(np.mean(cert_vals)),
                "recall_at_q": _agg_mean_sd(rq_vals),
                "net_utility_graded": _agg_mean_sd(ut_vals),
                "ece": _agg_mean_sd(ece_vals),
            }

        # Primary single-factor differences (conservative NB overlap correction on all real cohorts;
        # only Gate05 draws independent simulated populations per seed)
        is_truly_indep = bool(c_name == "Gate05_Synthetic_MultiMechanism")
        n_tst_m = int(np.mean([r["n_test_evaluated"] for r in runs]))
        n_trn_m = int(np.mean([r["n_train"] for r in runs]))

        prim_minus_base_ap = [
            r["arms"]["monolithic_matched_pilot_cmi"]["auprc"]
            - r["arms"]["calibrated_gbdt"]["auprc"]
            for r in runs
        ]
        prim_minus_base_tpr = [
            r["arms"]["monolithic_matched_pilot_cmi"]["np_tpr_at_fpr_budget"]
            - r["arms"]["calibrated_gbdt"]["np_tpr_at_fpr_budget"]
            for r in runs
        ]
        credo_minus_base = [
            r["arms"]["full_credo_lattice"]["auprc"] - r["arms"]["calibrated_gbdt"]["auprc"]
            for r in runs
        ]
        matched_minus_unmatched_cmi = [
            r["arms"]["monolithic_matched_pilot_cmi"]["auprc"]
            - r["arms"]["monolithic_unmatched_pilot_cmi"]["auprc"]
            for r in runs
        ]
        cmi_minus_pilot_mi = [
            r["arms"]["monolithic_matched_pilot_cmi"]["auprc"]
            - r["arms"]["monolithic_pilot_mi"]["auprc"]
            for r in runs
        ]
        cmi_minus_global_mi = [
            r["arms"]["monolithic_matched_pilot_cmi"]["auprc"]
            - r["arms"]["monolithic_global_mi"]["auprc"]
            for r in runs
        ]
        reg_minus_cmi = [
            r["arms"]["monolithic_credo_reg"]["auprc"]
            - r["arms"]["monolithic_matched_pilot_cmi"]["auprc"]
            for r in runs
        ]
        credo_minus_mi = [
            r["arms"]["monolithic_credo_reg"]["auprc"] - r["arms"]["monolithic_global_mi"]["auprc"]
            for r in runs
        ]
        credo_minus_unmatched = [
            r["arms"]["monolithic_credo_reg"]["auprc"]
            - r["arms"]["monolithic_unmatched_case_control_mi"]["auprc"]
            for r in runs
        ]
        hybrid_minus_single_gbdt = [
            r["arms"]["full_credo_lattice"]["auprc"]
            - r["arms"]["monolithic_matched_pilot_cmi"]["auprc"]
            for r in runs
        ]
        hybrid_minus_orth = [
            r["arms"]["full_credo_lattice"]["auprc"] - r["arms"]["pure_orthogonal_lattice"]["auprc"]
            for r in runs
        ]
        credo_minus_ens = [
            r["arms"]["full_credo_lattice"]["auprc"]
            - r["arms"]["monolithic_credo_ensemble"]["auprc"]
            for r in runs
        ]
        fov_minus_lit_conf = [
            r["conformal_foveation"]["credo_deficit_foveated_auprc"]
            - r["conformal_foveation"]["literal_conformal_01_auprc"]
            for r in runs
        ]
        fov_minus_top_risk = [
            r["conformal_foveation"]["credo_deficit_foveated_auprc"]
            - r["conformal_foveation"]["top_risk_k_auprc"]
            for r in runs
        ]

        def _nb(
            vals: list[float],
            _n_test: int = n_tst_m,
            _n_train: int = n_trn_m,
            _indep: bool = is_truly_indep,
        ):
            return nadeau_bengio_ttest(
                vals, n_test=_n_test, n_train=_n_train, disjoint_test_blocks=_indep
            )

        nb_prim_ap = _nb(prim_minus_base_ap)
        nb_prim_tpr = _nb(prim_minus_base_tpr)
        nb_credo = _nb(credo_minus_base)
        nb_matched_vs_unm_cmi = _nb(matched_minus_unmatched_cmi)
        nb_cmi_vs_pmi = _nb(cmi_minus_pilot_mi)
        nb_cmi_vs_gmi = _nb(cmi_minus_global_mi)
        nb_reg_vs_cmi = _nb(reg_minus_cmi)
        nb_reg_vs_mi = _nb(credo_minus_mi)
        nb_reg_vs_unm = _nb(credo_minus_unmatched)
        nb_hyb_vs_single = _nb(hybrid_minus_single_gbdt)
        nb_fov_vs_lit = _nb(fov_minus_lit_conf)
        nb_fov_vs_top = _nb(fov_minus_top_risk)

        # Aggregate all 6 runtime routing policies
        routing_summary: dict[str, Any] = {}
        for pol_k in (
            "credo_foveation",
            "literal_conformal_01",
            "top_risk_k",
            "uncertainty_k",
            "voi_k",
            "random_k",
        ):
            routing_summary[pol_k] = {
                "auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["policies"][pol_k]["auprc"] for r in runs]
                ),
                "raw_ap_retention": _agg_mean_sd(
                    [r["conformal_foveation"]["policies"][pol_k]["raw_ap_retention"] for r in runs]
                ),
                "incremental_ap_gain_retention": _agg_mean_sd(
                    [
                        r["conformal_foveation"]["policies"][pol_k]["incremental_ap_gain_retention"]
                        for r in runs
                    ]
                ),
                "net_utility_per_1k": _agg_mean_sd(
                    [
                        r["conformal_foveation"]["policies"][pol_k]["net_utility_per_1k"]
                        for r in runs
                    ]
                ),
                "recall": _agg_mean_sd(
                    [r["conformal_foveation"]["policies"][pol_k]["recall"] for r in runs]
                ),
                "precision": _agg_mean_sd(
                    [r["conformal_foveation"]["policies"][pol_k]["precision"] for r in runs]
                ),
                "unqueried_positive_fraction": _agg_mean_sd(
                    [
                        r["conformal_foveation"]["policies"][pol_k]["unqueried_positive_fraction"]
                        for r in runs
                    ]
                ),
                "unqueried_missed_positive_fraction": _agg_mean_sd(
                    [
                        r["conformal_foveation"]["policies"][pol_k][
                            "unqueried_missed_positive_fraction"
                        ]
                        for r in runs
                    ]
                ),
                "additional_tp_per_100_queries": _agg_mean_sd(
                    [
                        r["conformal_foveation"]["policies"][pol_k]["additional_tp_per_100_queries"]
                        for r in runs
                    ]
                ),
            }

        discovery_precision_by_strat = {
            st_k: {
                "relevant_precision": _agg_mean_sd(
                    [r["discovery_strategies"][st_k]["hidden_channel_precision"] for r in runs]
                ),
                "false_proxy_rate": _agg_mean_sd(
                    [r["discovery_strategies"][st_k]["false_proxy_selection_rate"] for r in runs]
                ),
                "false_decoy_rate": _agg_mean_sd(
                    [r["discovery_strategies"][st_k]["false_decoy_selection_rate"] for r in runs]
                ),
            }
            for st_k in runs[0]["discovery_strategies"].keys()
        }

        try:
            _, p_credo = wilcoxon(prim_minus_base_ap, alternative="greater")
            p_credo_f = float(p_credo)
        except ValueError:
            p_credo_f = 1.0

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
                "dataset_provenance": runs[0].get("dataset_provenance", "real_observational"),
                "include_synthetic_proxies": runs[0].get("include_synthetic_proxies", False),
                "eligibility_audit": runs[0].get("eligibility_audit", {}),
                "split_hashes": [r["split_hash"] for r in runs],
                "chosen_channels_matched_cmi": [
                    r["discovery_strategies"]["matched_pilot_cmi"]["chosen_channels"] for r in runs
                ],
                "n_seeds": n_seeds,
                "mean_n_train": n_trn_m,
                "mean_n_test": n_tst_m,
                "mean_n_pos_test": float(np.mean([r["n_pos_test"] for r in runs])),
                "receipt_adi": _agg_mean_sd(
                    [r["receipt_audit"]["ascertainment_distortion_index"] for r in runs]
                ),
                "pilot_fraction_measured": _agg_mean_sd(
                    [
                        r["discovery_strategies"]["matched_pilot_cmi"]["pilot_fraction_measured"]
                        for r in runs
                    ]
                ),
                "cost_accounting": runs[0].get("cost_accounting", {}),
                "mean_kappa0": _agg_mean_sd([r["mean_kappa0"] for r in runs]),
                "orthogonal_lattice_weight": _agg_mean_sd(
                    [r["orthogonal_lattice_weight"] for r in runs]
                ),
                "stage1_class0_coverage": _agg_mean_sd(
                    [r["conformal_foveation"]["stage1_class0_coverage"] for r in runs]
                ),
                "stage1_class1_coverage": _agg_mean_sd(
                    [r["conformal_foveation"]["stage1_class1_coverage"] for r in runs]
                ),
                "stage1_singleton_pos_rate": _agg_mean_sd(
                    [r["conformal_foveation"]["stage1_singleton_pos_rate"] for r in runs]
                ),
                "stage1_singleton_pos_ppv": _agg_mean_sd(
                    [r["conformal_foveation"]["stage1_singleton_pos_ppv"] for r in runs]
                ),
                "stage1_passive_abstention": _agg_mean_sd(
                    [r["conformal_foveation"]["stage1_passive_abstention_rate"] for r in runs]
                ),
                "stage2_sensor_savings": _agg_mean_sd(
                    [r["conformal_foveation"]["stage2_sensor_savings_fraction"] for r in runs]
                ),
                "raw_ap_retention": _agg_mean_sd(
                    [r["conformal_foveation"]["raw_ap_retention"] for r in runs]
                ),
                "incremental_ap_gain_retention": _agg_mean_sd(
                    [r["conformal_foveation"]["incremental_ap_gain_retention"] for r in runs]
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
                "stage1_only_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["stage1_only_auprc"] for r in runs]
                ),
                "literal_conformal_01_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["literal_conformal_01_auprc"] for r in runs]
                ),
                "classical_midband_conformal_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["classical_midband_conformal_auprc"] for r in runs]
                ),
                "top_risk_k_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["top_risk_k_auprc"] for r in runs]
                ),
                "uncertainty_k_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["uncertainty_k_auprc"] for r in runs]
                ),
                "voi_k_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["voi_k_auprc"] for r in runs]
                ),
                "random_routing_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["random_routing_auprc"] for r in runs]
                ),
                "credo_deficit_foveated_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["credo_deficit_foveated_auprc"] for r in runs]
                ),
                "full_stage2_auprc": _agg_mean_sd(
                    [r["conformal_foveation"]["full_stage2_auprc"] for r in runs]
                ),
                "routing_policies": routing_summary,
                "ablation_ladder": ablation_table,
                "discovery_precision_by_strategy": discovery_precision_by_strat,
                "credo_relevant_channel_precision": discovery_precision_by_strat["credo_reg"][
                    "relevant_precision"
                ],
                "global_mi_relevant_channel_precision": discovery_precision_by_strat["global_mi"][
                    "relevant_precision"
                ],
                "unmatched_cc_relevant_channel_precision": discovery_precision_by_strat[
                    "unmatched_case_control_mi"
                ]["relevant_precision"],
                "primary_cmi_minus_base_auprc": _agg_mean_sd(prim_minus_base_ap),
                "primary_cmi_minus_base_np_tpr": _agg_mean_sd(prim_minus_base_tpr),
                "matched_minus_unmatched_cmi_auprc": _agg_mean_sd(matched_minus_unmatched_cmi),
                "cmi_minus_pilot_mi_auprc": _agg_mean_sd(cmi_minus_pilot_mi),
                "cmi_minus_global_mi_auprc": _agg_mean_sd(cmi_minus_global_mi),
                "reg_minus_cmi_auprc": _agg_mean_sd(reg_minus_cmi),
                "hybrid_minus_single_gbdt_auprc": _agg_mean_sd(hybrid_minus_single_gbdt),
                "credo_minus_base_auprc": _agg_mean_sd(credo_minus_base),
                "credo_minus_equal_ensemble_auprc": _agg_mean_sd(credo_minus_ens),
                "hybrid_minus_pure_orth_auprc": _agg_mean_sd(hybrid_minus_orth),
                "credo_reg_minus_global_mi_auprc": _agg_mean_sd(credo_minus_mi),
                "credo_reg_minus_unmatched_cc_auprc": _agg_mean_sd(credo_minus_unmatched),
                "foveated_credo_minus_midband_auprc": _agg_mean_sd(fov_minus_lit_conf),
                "foveated_credo_minus_top_risk_auprc": _agg_mean_sd(fov_minus_top_risk),
                "wilcoxon_credo_vs_base_p": p_credo_f,
                "nb_prim_vs_base_auprc_p": nb_prim_ap.p_value_one_sided,
                "nb_prim_vs_base_auprc_ci95": [nb_prim_ap.ci95_low, nb_prim_ap.ci95_high],
                "nb_prim_vs_base_np_tpr_p": nb_prim_tpr.p_value_one_sided,
                "nb_prim_vs_base_np_tpr_ci95": [nb_prim_tpr.ci95_low, nb_prim_tpr.ci95_high],
                "nb_matched_vs_unmatched_cmi_p": nb_matched_vs_unm_cmi.p_value_one_sided,
                "nb_cmi_vs_pilot_mi_p": nb_cmi_vs_pmi.p_value_one_sided,
                "nb_cmi_vs_global_mi_p": nb_cmi_vs_gmi.p_value_one_sided,
                "nb_reg_vs_cmi_two_sided_p": nb_reg_vs_cmi.p_value_two_sided,
                "nb_hybrid_vs_single_two_sided_p": nb_hyb_vs_single.p_value_two_sided,
                "nadeau_bengio_credo_vs_base_p": nb_credo.p_value_one_sided,
                "nb_reg_vs_mi_p": nb_reg_vs_mi.p_value_one_sided,
                "nb_reg_vs_unmatched_p": nb_reg_vs_unm.p_value_one_sided,
                "nb_foveation_vs_midband_p": nb_fov_vs_lit.p_value_one_sided,
                "nb_foveation_vs_top_risk_two_sided_p": nb_fov_vs_top.p_value_two_sided,
                "nadeau_bengio_ci95": [nb_credo.ci95_low, nb_credo.ci95_high],
                "subtype_recall": subtype_agg,
            }
        )

    # Apply Holm (1979) step-down FWER adjustment across EVERY primary hypothesis family
    families = [
        ("nb_prim_vs_base_auprc_p", "holm_prim_vs_base_auprc_p"),
        ("nb_prim_vs_base_np_tpr_p", "holm_prim_vs_base_np_tpr_p"),
        ("nb_matched_vs_unmatched_cmi_p", "holm_matched_vs_unmatched_cmi_p"),
        ("nb_cmi_vs_pilot_mi_p", "holm_cmi_vs_pilot_mi_p"),
        ("nb_cmi_vs_global_mi_p", "holm_cmi_vs_global_mi_p"),
        ("nadeau_bengio_credo_vs_base_p", "holm_bonferroni_fwer_p"),
        ("nb_foveation_vs_midband_p", "holm_foveation_vs_midband_fwer_p"),
        ("nb_foveation_vs_top_risk_two_sided_p", "holm_foveation_vs_top_risk_two_sided_p"),
        ("nb_hybrid_vs_single_two_sided_p", "holm_hybrid_vs_single_two_sided_p"),
    ]
    for raw_k, adj_k in families:
        adj_vals = holm_bonferroni_adjust([c[raw_k] for c in cohort_summaries])
        for c, av in zip(cohort_summaries, adj_vals, strict=True):
            c[adj_k] = av
    return cohort_summaries


def run_wave2_paper_campaign(
    out_dir: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    wave1_summary_path: Path = Path("artifacts/bench/paper_hardened/campaign_summary.json"),
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    uncapped: bool = True,
) -> dict[str, Any]:
    """Execute Wave-2 Expansion: 10-dataset CLIMB Tier-1 + 4 High-Scale Multi-Table Flagships.

    Merges Wave-1 hardened summaries (when present) to emit a unified 9-Cohort + 10-Dataset
    campaign summary with joint Holm-Bonferroni FWER multiplicity control across all 9 cohorts!
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = len(seeds)

    print(
        f"[Wave-2] Step 1/5: Running 10-dataset Tier-1 + CLIMB Ensembles ({n_seeds} seeds)...",
        flush=True,
    )
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tier1_by_key: dict[str, list[dict[str, Any]]] = {s.key: [] for s in TIER1_PUBLIC_SPECS}
    for s in seeds:
        t1_seed_dir = out_dir / "seeds" / f"tier1_seed{s}"
        t1_json = t1_seed_dir / "tier1_results.json"
        if t1_json.exists():
            seed_runs = json.loads(t1_json.read_text())
        else:
            seed_runs = run_tier1_saturation_suite(t1_seed_dir, cache_dir=cache_dir, seed=s)
        for rec in seed_runs:
            tier1_by_key[rec["dataset_key"]].append(rec)
        gc.collect()

    tier1_summary: list[dict[str, Any]] = []
    for spec in TIER1_PUBLIC_SPECS:
        runs = tier1_by_key[spec.key]
        gbdt_ap = [r["arms"]["calibrated_gbdt"]["auprc"] for r in runs]
        smote_ap = [r["arms"]["smote_gbdt"]["auprc"] for r in runs]
        eec_ap = [r["arms"]["easy_ensemble"]["auprc"] for r in runs]
        rus_ap = [r["arms"]["rusboost"]["auprc"] for r in runs]
        cop_ap = [r["arms"]["latent_copula_gbdt"]["auprc"] for r in runs]
        cs_ap = [r["arms"]["cost_sensitive_gbdt"]["auprc"] for r in runs]
        ens_ap = [r["arms"]["soft_avg_ensemble"]["auprc"] for r in runs]
        smote_deltas = [sm - gb for sm, gb in zip(smote_ap, gbdt_ap, strict=True)]
        eec_deltas = [ee - gb for ee, gb in zip(eec_ap, gbdt_ap, strict=True)]
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
        nb_eec = nadeau_bengio_ttest(
            eec_deltas,
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
                "easy_ensemble_auprc": _agg_mean_sd(eec_ap),
                "rusboost_auprc": _agg_mean_sd(rus_ap),
                "latent_copula_auprc": _agg_mean_sd(cop_ap),
                "cost_sensitive_auprc": _agg_mean_sd(cs_ap),
                "soft_ensemble_auprc": _agg_mean_sd(ens_ap),
                "smote_minus_gbdt_auprc": _agg_mean_sd(smote_deltas),
                "easy_ensemble_minus_gbdt_auprc": _agg_mean_sd(eec_deltas),
                "uncal_smote_ece_degradation": _agg_mean_sd(ece_deg),
                "wilcoxon_one_sided_p": p_w,
                "nadeau_bengio_one_sided_p": nb_smote.p_value_one_sided,
                "easy_ensemble_nb_one_sided_p": nb_eec.p_value_one_sided,
                "nadeau_bengio_ci95": [nb_smote.ci95_low, nb_smote.ci95_high],
            }
        )

    wave2_runs: dict[str, list[dict[str, Any]]] = {
        "Gate05_Synthetic_MultiMechanism": [],
        "UCI_AI4I_2020_Maintenance": [],
        "UCI_Taiwan_Credit_6m_Panel": [],
        "UCI_Diabetes_130US_Hospitals": [],
        "ULB_CreditCard_Fraud_Chronological": [],
        "Scania_APS_76k_Fleet": [],
        "Home_Credit_307k_7Table_Relational": [],
        "IEEE_CIS_590k_MultiTable": [],
        "NeurIPS_BAF_1M_Chronological": [],
    }

    cap_rows = None if uncapped else 150000

    def _run_or_load_ckpt(c_key: str, s_val: int, loader_fn: Any) -> dict[str, Any]:
        cp = ckpt_dir / f"{c_key}_seed{s_val}.json"
        if cp.exists():
            return json.loads(cp.read_text())
        coh_obj = loader_fn()
        res_obj = run_multi_channel_cohort_benchmark(coh_obj, seed=s_val, n_boot=200)
        cp.write_text(json.dumps(res_obj, indent=2))
        print(
            f"  [Done] {c_key} seed={s_val}: "
            f"Base={res_obj['arms']['calibrated_gbdt']['auprc']:.4f} | "
            f"MI={res_obj['arms']['monolithic_global_mi']['auprc']:.4f} | "
            f"UnmCC={res_obj['arms']['monolithic_unmatched_case_control_mi']['auprc']:.4f} | "
            f"REG={res_obj['arms']['monolithic_credo_reg']['auprc']:.4f} | "
            f"CREDO={res_obj['arms']['full_credo_lattice']['auprc']:.4f} | "
            f"Fov={res_obj['conformal_foveation']['credo_deficit_foveated_auprc']:.4f} vs "
            f"Mid={res_obj['conformal_foveation']['classical_midband_conformal_auprc']:.4f}",
            flush=True,
        )
        del coh_obj
        gc.collect()
        return res_obj

    print("[Wave-2] Running Wave-1 5 Cohorts under Journal-Ready Protocol...", flush=True)
    for w_idx, s in enumerate(seeds):
        wave2_runs["Gate05_Synthetic_MultiMechanism"].append(
            _run_or_load_ckpt(
                "Gate05_Synthetic_MultiMechanism",
                s,
                lambda s=s: generate_gate05_cohort(
                    n_samples=12000, seed=s, selective_receipts=True
                ),
            )
        )
        wave2_runs["UCI_AI4I_2020_Maintenance"].append(
            _run_or_load_ckpt(
                "UCI_AI4I_2020_Maintenance",
                s,
                lambda s=s: load_tier2_ai4i_maintenance(cache_dir=cache_dir, seed=s),
            )
        )
        wave2_runs["UCI_Taiwan_Credit_6m_Panel"].append(
            _run_or_load_ckpt(
                "UCI_Taiwan_Credit_6m_Panel",
                s,
                lambda s=s: load_tier2_taiwan_credit_panel(cache_dir=cache_dir, seed=s),
            )
        )
        wave2_runs["UCI_Diabetes_130US_Hospitals"].append(
            _run_or_load_ckpt(
                "UCI_Diabetes_130US_Hospitals",
                s,
                lambda s=s: load_tier2_diabetes130(
                    cache_dir=cache_dir, max_patients=None if uncapped else 25000, seed=s
                ),
            )
        )
        wave2_runs["ULB_CreditCard_Fraud_Chronological"].append(
            _run_or_load_ckpt(
                "ULB_CreditCard_Fraud_Chronological",
                s,
                lambda w_idx=w_idx, s=s: load_tier2_credit_card_fraud(
                    cache_dir=cache_dir,
                    max_negatives=None if uncapped else 60000,
                    window_idx=w_idx if uncapped else None,
                    n_windows=n_seeds,
                    seed=s,
                ),
            )
        )

    print("[Wave-2] Step 2/5: Running Scania APS 76k Fleet across seeds...", flush=True)
    for s in seeds:
        wave2_runs["Scania_APS_76k_Fleet"].append(
            _run_or_load_ckpt(
                "Scania_APS_76k_Fleet",
                s,
                lambda s=s: load_wave2_scania_aps(cache_dir=cache_dir, seed=s),
            )
        )

    print(
        "[Wave-2] Step 3/5: Running Home Credit 307k 7-Table Relational across seeds...", flush=True
    )
    for s in seeds:
        wave2_runs["Home_Credit_307k_7Table_Relational"].append(
            _run_or_load_ckpt(
                "Home_Credit_307k_7Table_Relational",
                s,
                lambda s=s: load_wave2_home_credit_relational(
                    cache_dir=cache_dir, max_rows=cap_rows, seed=s
                ),
            )
        )

    print(
        "[Wave-2] Step 4/5: Running IEEE-CIS 590k 2-Table Fraud across disjoint eras...", flush=True
    )
    for w_idx, s in enumerate(seeds):
        wave2_runs["IEEE_CIS_590k_MultiTable"].append(
            _run_or_load_ckpt(
                "IEEE_CIS_590k_MultiTable",
                s,
                lambda w_idx=w_idx, s=s: load_wave2_ieee_cis_fraud(
                    cache_dir=cache_dir,
                    window_idx=w_idx if uncapped else None,
                    n_windows=n_seeds,
                    max_rows=cap_rows,
                    seed=s,
                ),
            )
        )

    print("[Wave-2] Step 5/5: Running NeurIPS BAF 1M Fraud across disjoint eras...", flush=True)
    for w_idx, s in enumerate(seeds):
        wave2_runs["NeurIPS_BAF_1M_Chronological"].append(
            _run_or_load_ckpt(
                "NeurIPS_BAF_1M_Chronological",
                s,
                lambda w_idx=w_idx, s=s: load_wave2_neurips_baf(
                    cache_dir=cache_dir,
                    window_idx=w_idx if uncapped else None,
                    n_windows=n_seeds,
                    max_rows=cap_rows,
                    seed=s,
                ),
            )
        )

    combined_summaries = _summarize_multi_cohort_runs(wave2_runs, n_seeds)

    payload = {
        "seeds": list(seeds),
        "uncapped": uncapped,
        "tier1_summary": tier1_summary,
        "wave2_only_summary": combined_summaries[5:],
        "multi_channel_summary": combined_summaries,
    }
    (out_dir / "campaign_summary.json").write_text(json.dumps(payload, indent=2))

    _plot_multiseed_fig1_tier1(tier1_summary, out_dir / "fig1_tier1_saturation")
    _plot_multiseed_fig2_ablations(combined_summaries, out_dir / "fig2_ablation_ladder")
    _plot_multiseed_fig3_kappa0_foveation(combined_summaries, out_dir / "fig3_kappa0_and_foveation")
    _write_latex_and_markdown_tables(tier1_summary, combined_summaries, len(seeds), out_dir)
    return payload


def run_reconstructed_paper_campaign(
    out_dir: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    uncapped: bool = True,
) -> dict[str, Any]:
    """Execute the full reconstructed CREDO evaluation suite with versioned receipts.

    Runs four clean, separated evaluation tracks with checkpoint resume:
    1. Tier-1 Fixed-Table Resampling vs Stabilized Baseline (10 datasets x 5 splits) with
       both illustrative planning half-widths and observed Wilson 95% intervals.
    2. Primary Native-Channel Multi-Cohort Suite (9 cohorts x 5 splits/windows,
       ``include_synthetic_proxies=False`` on all 8 public cohorts + Unrestricted All-Native
       GBDT reference + all 4 single-factor experiment families + all 6 routing policies).
    3. Separate Injected-Proxy Stress Test Suite (8 public cohorts x 5 splits/windows,
       ``include_synthetic_proxies=True``, evaluating false proxy/decoy selection rates and
       downstream Single-GBDT AUPRC).
    4. Seven-Scenario Adversarial & Failure-Mode Simulation Suite (x 5 seeds).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = len(seeds)
    ckpt_native = out_dir / "checkpoints_native"
    ckpt_proxy = out_dir / "checkpoints_proxy_stress"
    ckpt_native.mkdir(parents=True, exist_ok=True)
    ckpt_proxy.mkdir(parents=True, exist_ok=True)

    print(f"[Reconstructed 1/4] Tier-1 Fixed-Table Suite ({n_seeds} seeds)...", flush=True)
    tier1_by_key: dict[str, list[dict[str, Any]]] = {s.key: [] for s in TIER1_PUBLIC_SPECS}
    for s in seeds:
        t1_seed_dir = out_dir / "seeds" / f"tier1_seed{s}"
        t1_json = t1_seed_dir / "tier1_results.json"
        if t1_json.exists():
            seed_runs = json.loads(t1_json.read_text())
        else:
            seed_runs = run_tier1_saturation_suite(t1_seed_dir, cache_dir=cache_dir, seed=s)
        for rec in seed_runs:
            tier1_by_key[rec["dataset_key"]].append(rec)
        gc.collect()

    tier1_summary: list[dict[str, Any]] = []
    for spec in TIER1_PUBLIC_SPECS:
        runs = tier1_by_key[spec.key]
        gbdt_ap = [r["arms"]["calibrated_gbdt"]["auprc"] for r in runs]
        smote_ap = [r["arms"]["smote_gbdt"]["auprc"] for r in runs]
        eec_ap = [r["arms"]["easy_ensemble"]["auprc"] for r in runs]
        rus_ap = [r["arms"]["rusboost"]["auprc"] for r in runs]
        cop_ap = [r["arms"]["latent_copula_gbdt"]["auprc"] for r in runs]
        cs_ap = [r["arms"]["cost_sensitive_gbdt"]["auprc"] for r in runs]
        ens_ap = [r["arms"]["soft_avg_ensemble"]["auprc"] for r in runs]
        smote_deltas = [sm - gb for sm, gb in zip(smote_ap, gbdt_ap, strict=True)]
        eec_deltas = [ee - gb for ee, gb in zip(eec_ap, gbdt_ap, strict=True)]
        ece_deg = [
            r["decoder_deltas_vs_calibrated_gbdt"]["smote_gbdt"]["ece_degradation_uncal"]
            for r in runs
        ]
        obs_wilson_hws = [
            0.5
            * (
                r["arms"]["calibrated_gbdt"]["np_tpr_wilson_high"]
                - r["arms"]["calibrated_gbdt"]["np_tpr_wilson_low"]
            )
            for r in runs
        ]
        np_cert_rate = float(
            np.mean([float(r["arms"]["calibrated_gbdt"].get("np_certified", False)) for r in runs])
        )
        nb_smote = nadeau_bengio_ttest(
            smote_deltas,
            n_test=int(round(runs[0]["n_rows"] * 0.15)),
            n_train=int(round(runs[0]["n_rows"] * 0.55)),
            disjoint_test_blocks=False,
        )
        nb_eec = nadeau_bengio_ttest(
            eec_deltas,
            n_test=int(round(runs[0]["n_rows"] * 0.15)),
            n_train=int(round(runs[0]["n_rows"] * 0.55)),
            disjoint_test_blocks=False,
        )
        tier1_summary.append(
            {
                "dataset_key": spec.key,
                "display_name": spec.display_name,
                "dataset_sha256": runs[0]["dataset_sha256"],
                "n_rows": runs[0]["n_rows"],
                "n_features": runs[0]["n_features"],
                "prevalence": runs[0]["prevalence"],
                "n_pos_test": runs[0]["n_pos_test"],
                "planning_halfwidth_at_80": runs[0]["gate_c"]["recall_ci_half_width_at_80"],
                "recall_ci_half_width": runs[0]["gate_c"]["recall_ci_half_width_at_80"],
                "observed_wilson_halfwidth": _agg_mean_sd(obs_wilson_hws),
                "np_certified_rate": np_cert_rate,
                "modal_regime": max(
                    set(r["four_ceilings"]["bottleneck_regime"] for r in runs),
                    key=lambda reg: sum(
                        1 for r in runs if r["four_ceilings"]["bottleneck_regime"] == reg
                    ),
                ),
                "calibrated_gbdt_auprc": _agg_mean_sd(gbdt_ap),
                "smote_gbdt_auprc": _agg_mean_sd(smote_ap),
                "easy_ensemble_auprc": _agg_mean_sd(eec_ap),
                "rusboost_auprc": _agg_mean_sd(rus_ap),
                "latent_copula_auprc": _agg_mean_sd(cop_ap),
                "cost_sensitive_auprc": _agg_mean_sd(cs_ap),
                "soft_ensemble_auprc": _agg_mean_sd(ens_ap),
                "smote_minus_gbdt_auprc": _agg_mean_sd(smote_deltas),
                "easy_ensemble_minus_gbdt_auprc": _agg_mean_sd(eec_deltas),
                "uncal_smote_ece_degradation": _agg_mean_sd(ece_deg),
                "nadeau_bengio_one_sided_p": nb_smote.p_value_one_sided,
                "nadeau_bengio_two_sided_p": nb_smote.p_value_two_sided,
                "easy_ensemble_nb_one_sided_p": nb_eec.p_value_one_sided,
                "nadeau_bengio_ci95": [nb_smote.ci95_low, nb_smote.ci95_high],
            }
        )
    t1_holm = holm_bonferroni_adjust([r["nadeau_bengio_one_sided_p"] for r in tier1_summary])
    for r, hp in zip(tier1_summary, t1_holm, strict=True):
        r["holm_smote_vs_gbdt_p"] = hp

    cap_rows = None if uncapped else 150000

    def _cohort_builders(with_proxies: bool) -> list[tuple[str, Any]]:
        builders: list[tuple[str, Any]] = []
        if not with_proxies:
            builders.append(
                (
                    "Gate05_Synthetic_MultiMechanism",
                    lambda w_i, s_v: generate_gate05_cohort(
                        n_samples=12000, seed=s_v, selective_receipts=True
                    ),
                )
            )
        builders.extend(
            [
                (
                    "UCI_AI4I_2020_Maintenance",
                    lambda w_i, s_v, wp=with_proxies: load_tier2_ai4i_maintenance(
                        cache_dir=cache_dir, seed=s_v, include_synthetic_proxies=wp
                    ),
                ),
                (
                    "UCI_Taiwan_Credit_6m_Panel",
                    lambda w_i, s_v, wp=with_proxies: load_tier2_taiwan_credit_panel(
                        cache_dir=cache_dir, seed=s_v, include_synthetic_proxies=wp
                    ),
                ),
                (
                    "UCI_Diabetes_130US_Hospitals",
                    lambda w_i, s_v, wp=with_proxies: load_tier2_diabetes130(
                        cache_dir=cache_dir,
                        max_patients=None if uncapped else 25000,
                        seed=s_v,
                        include_synthetic_proxies=wp,
                    ),
                ),
                (
                    "ULB_CreditCard_Fraud_Chronological",
                    lambda w_i, s_v, wp=with_proxies: load_tier2_credit_card_fraud(
                        cache_dir=cache_dir,
                        max_negatives=None if uncapped else 60000,
                        window_idx=w_i if uncapped else None,
                        n_windows=n_seeds,
                        seed=s_v,
                        include_synthetic_proxies=wp,
                    ),
                ),
                (
                    "Scania_APS_76k_Fleet",
                    lambda w_i, s_v, wp=with_proxies: load_wave2_scania_aps(
                        cache_dir=cache_dir, seed=s_v, include_synthetic_proxies=wp
                    ),
                ),
                (
                    "Home_Credit_307k_7Table_Relational",
                    lambda w_i, s_v, wp=with_proxies: load_wave2_home_credit_relational(
                        cache_dir=cache_dir,
                        max_rows=cap_rows,
                        seed=s_v,
                        include_synthetic_proxies=wp,
                    ),
                ),
                (
                    "IEEE_CIS_590k_MultiTable",
                    lambda w_i, s_v, wp=with_proxies: load_wave2_ieee_cis_fraud(
                        cache_dir=cache_dir,
                        window_idx=w_i if uncapped else None,
                        n_windows=n_seeds,
                        max_rows=cap_rows,
                        seed=s_v,
                        include_synthetic_proxies=wp,
                    ),
                ),
                (
                    "NeurIPS_BAF_1M_Chronological",
                    lambda w_i, s_v, wp=with_proxies: load_wave2_neurips_baf(
                        cache_dir=cache_dir,
                        window_idx=w_i if uncapped else None,
                        n_windows=n_seeds,
                        max_rows=cap_rows,
                        seed=s_v,
                        include_synthetic_proxies=wp,
                    ),
                ),
            ]
        )
        return builders

    print(
        f"[Reconstructed 2/4] Primary Native-Channel Suite (9 cohorts x {n_seeds} splits)...",
        flush=True,
    )
    native_runs: dict[str, list[dict[str, Any]]] = {}
    for c_key, builder in _cohort_builders(with_proxies=False):
        native_runs[c_key] = []
        for w_idx, s in enumerate(seeds):
            cp = ckpt_native / f"{c_key}_seed{s}.json"
            if cp.exists():
                res_obj = json.loads(cp.read_text())
            else:
                coh_obj = builder(w_idx, s)
                res_obj = run_multi_channel_cohort_benchmark(
                    coh_obj, seed=s, n_boot=200, include_complex_fusion=True
                )
                cp.write_text(json.dumps(res_obj, indent=2))
                del coh_obj
                gc.collect()
            print(
                f"  [Native] {c_key} s={s}: "
                f"Base={res_obj['arms']['calibrated_gbdt']['auprc']:.4f} | "
                f"GlobMI={res_obj['arms']['monolithic_global_mi']['auprc']:.4f} | "
                f"UnmCMI={res_obj['arms']['monolithic_unmatched_pilot_cmi']['auprc']:.4f} | "
                f"StratCMI(A5)={res_obj['arms']['monolithic_matched_pilot_cmi']['auprc']:.4f} | "
                f"AllNat={res_obj['arms']['unrestricted_all_native_gbdt']['auprc']:.4f} | "
                f"Fov={res_obj['conformal_foveation']['credo_deficit_foveated_auprc']:.4f} "
                f"TopK={res_obj['conformal_foveation']['top_risk_k_auprc']:.4f} "
                f"Lit01={res_obj['conformal_foveation']['literal_conformal_01_auprc']:.4f}",
                flush=True,
            )
            native_runs[c_key].append(res_obj)

    native_summaries = _summarize_multi_cohort_runs(native_runs, n_seeds)

    print(
        f"[Reconstructed 3/4] Injected-Proxy Stress-Test Suite (8 public cohorts x {n_seeds})...",
        flush=True,
    )
    proxy_runs: dict[str, list[dict[str, Any]]] = {}
    for c_key, builder in _cohort_builders(with_proxies=True):
        proxy_runs[c_key] = []
        for w_idx, s in enumerate(seeds):
            cp = ckpt_proxy / f"{c_key}_seed{s}.json"
            if cp.exists():
                res_obj = json.loads(cp.read_text())
            else:
                coh_obj = builder(w_idx, s)
                res_obj = run_multi_channel_cohort_benchmark(
                    coh_obj, seed=s, n_boot=100, include_complex_fusion=False
                )
                cp.write_text(json.dumps(res_obj, indent=2))
                del coh_obj
                gc.collect()
            g_fp = res_obj["discovery_strategies"]["global_mi"]["false_proxy_selection_rate"]
            s_fp = res_obj["discovery_strategies"]["matched_pilot_cmi"][
                "false_proxy_selection_rate"
            ]
            print(
                f"  [ProxyStress] {c_key} s={s}: "
                f"GlobMI={res_obj['arms']['monolithic_global_mi']['auprc']:.4f} "
                f"(proxy={g_fp:.2f}) | "
                f"StratCMI={res_obj['arms']['monolithic_matched_pilot_cmi']['auprc']:.4f} "
                f"(proxy={s_fp:.2f})",
                flush=True,
            )
            proxy_runs[c_key].append(res_obj)

    proxy_summaries = _summarize_multi_cohort_runs(proxy_runs, n_seeds)

    print(
        f"[Reconstructed 4/4] 7-Scenario Adversarial Simulation Suite ({n_seeds} seeds)...",
        flush=True,
    )
    adv_json = out_dir / "adversarial_simulations.json"
    if adv_json.exists():
        adv_suite = json.loads(adv_json.read_text())
    else:
        adv_suite = run_adversarial_simulation_suite(seeds=seeds)
        adv_json.write_text(json.dumps(adv_suite, indent=2))

    # Build explicit split & selection manifest for external auditability (Review Sec. 6.E)
    split_manifest: dict[str, Any] = {}
    for c_key, runs in native_runs.items():
        split_manifest[c_key] = [
            {
                "seed": r["seed"],
                "split_hash": r["split_hash"],
                "split_protocol": r["split_protocol"],
                "n_train": r["n_train"],
                "n_cal": r["n_cal_evaluated"],
                "n_val": r["n_val_evaluated"],
                "n_test": r["n_test_evaluated"],
                "n_pos_test": r["n_pos_test"],
                "selected_channels_matched_cmi": r["discovery_strategies"]["matched_pilot_cmi"][
                    "chosen_channels"
                ],
                "selected_channels_unmatched_cmi": r["discovery_strategies"]["unmatched_pilot_cmi"][
                    "chosen_channels"
                ],
                "selected_channels_global_mi": r["discovery_strategies"]["global_mi"][
                    "chosen_channels"
                ],
                "cost_accounting": r["cost_accounting"],
            }
            for r in runs
        ]
    (out_dir / "split_and_selection_manifest.json").write_text(json.dumps(split_manifest, indent=2))

    payload = {
        "seeds": list(seeds),
        "uncapped": uncapped,
        "tier1_summary": tier1_summary,
        "native_channel_summary": native_summaries,
        "multi_channel_summary": native_summaries,
        "injected_proxy_stress_summary": proxy_summaries,
        "adversarial_simulations_summary": adv_suite["summary"],
    }
    (out_dir / "campaign_summary.json").write_text(json.dumps(payload, indent=2))

    _plot_multiseed_fig1_tier1(tier1_summary, out_dir / "fig1_tier1_saturation")
    _plot_multiseed_fig2_ablations(native_summaries, out_dir / "fig2_ablation_ladder")
    _plot_multiseed_fig3_kappa0_foveation(native_summaries, out_dir / "fig3_kappa0_and_foveation")
    _write_latex_and_markdown_tables(tier1_summary, native_summaries, len(seeds), out_dir)
    return payload


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
        label="Mean ± 1 SD across 5 splits",
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel("Test AUPRC Delta: SMOTE+GBDT minus Stabilized Calibrated GBDT")
    axes[0].set_title("A. Fixed-Table Resampling vs. Stabilized Baseline")
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[0].grid(axis="x", alpha=0.3)

    n_pos = [r["n_pos_test"] for r in tier1_summary]
    plan_hw = [r["recall_ci_half_width"] for r in tier1_summary]
    obs_hw = [
        r.get("observed_wilson_halfwidth", {}).get("mean", r["recall_ci_half_width"])
        for r in tier1_summary
    ]
    grid_n = np.linspace(4, max(max(n_pos) * 1.15, 270), 200)
    grid_hw = 1.96 * np.sqrt(0.8 * 0.2 / grid_n)

    axes[1].plot(
        grid_n,
        grid_hw,
        color="#d62728",
        linewidth=1.6,
        linestyle="--",
        label="Planning Curve (r=0.80)",
    )
    axes[1].axhline(
        0.05, color="#2ca02c", linestyle=":", linewidth=1.5, label="±0.05 Planning Target (n+=246)"
    )
    axes[1].scatter(
        n_pos, plan_hw, color="#888888", marker="^", s=45, zorder=3, label="Planning Half-Width"
    )
    axes[1].scatter(
        n_pos,
        obs_hw,
        color="#111111",
        marker="o",
        s=55,
        zorder=4,
        label="Observed Wilson Half-Width",
    )
    label_offsets = {
        "Abalone": (6, -10),
        "Ozone": (6, 5),
        "PC1": (6, -9),
        "Yeast": (6, 4),
        "Thyroid": (-36, -11),
        "Mammography": (6, 4),
    }
    for xv, yv, nm in zip(n_pos, obs_hw, names, strict=True):
        short_lbl = nm.split()[0]
        xy_off = label_offsets.get(short_lbl, (5, 4))
        axes[1].annotate(short_lbl, (xv, yv), xytext=xy_off, textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Test Positives per Split (n+)")
    axes[1].set_ylabel("Sensitivity 95% Interval Half-Width")
    axes[1].set_title("B. Operating-Point Sensitivity Uncertainty vs. Test n+")
    axes[1].legend(fontsize=8.0)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_multiseed_fig2_ablations(cohorts: list[dict[str, Any]], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.7), dpi=180)
    short_names = [
        c["cohort_name"]
        .replace("Gate05_Synthetic_MultiMechanism", "Gate 0.5 (Sim)")
        .replace("UCI_AI4I_2020_Maintenance", "AI4I Maint (10k)")
        .replace("UCI_Taiwan_Credit_6m_Panel", "Taiwan Credit (27k)")
        .replace("UCI_Diabetes_130US_Hospitals", "Diabetes Adm (99k)")
        .replace("ULB_CreditCard_Fraud_Chronological", "ULB Fraud (285k)")
        .replace("Scania_APS_76k_Fleet", "Scania APS (76k)")
        .replace("Home_Credit_307k_7Table_Relational", "Home Credit 7T (308k)")
        .replace("IEEE_CIS_590k_MultiTable", "IEEE-CIS 2T (591k)")
        .replace("NeurIPS_BAF_1M_Chronological", "NeurIPS BAF (1.0M)")
        for c in cohorts
    ]
    x = np.arange(len(short_names))
    key_arms = [
        ("calibrated_gbdt", "A0: Base GBDT (X_base)", "#7f7f7f"),
        ("monolithic_global_mi", "A4: Global MI (100% rows)", "#ffbb78"),
        ("monolithic_unmatched_pilot_cmi", "A4u: Unmatched Pilot G_m", "#9edae5"),
        ("monolithic_matched_pilot_cmi", "A5: Stratified Pilot G_m (Primary)", "#1f77b4"),
        ("full_credo_lattice", "A7: Hybrid Lattice (Selected)", "#2ca02c"),
        ("unrestricted_all_native_gbdt", "Ref: All-Native Unrestricted", "#393b79"),
    ]
    w = 0.135

    for idx, (akey, alabel, col) in enumerate(key_arms):
        means = [c["ablation_ladder"][akey]["auprc"]["mean"] for c in cohorts]
        sds = [c["ablation_ladder"][akey]["auprc"]["sd"] for c in cohorts]
        axes[0].bar(
            x + (idx - 2.5) * w,
            means,
            w,
            yerr=sds,
            capsize=2.0,
            label=alabel,
            color=col,
            edgecolor="#333333",
            linewidth=0.4,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=16, ha="right", fontsize=8.3)
    axes[0].set_ylabel("Test AUPRC (Mean ± SD Across 5 Splits)")
    axes[0].set_title("A. Native-Channel Discovery & Complexity Ablation: Test AUPRC")
    axes[0].legend(fontsize=7.3, ncol=2)
    axes[0].grid(axis="y", alpha=0.3)

    for idx, (akey, alabel, col) in enumerate(key_arms):
        means = [c["ablation_ladder"][akey]["np_tpr"]["mean"] for c in cohorts]
        sds = [c["ablation_ladder"][akey]["np_tpr"]["sd"] for c in cohorts]
        axes[1].bar(
            x + (idx - 2.5) * w,
            means,
            w,
            yerr=sds,
            capsize=2.0,
            label=alabel,
            color=col,
            edgecolor="#333333",
            linewidth=0.4,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=16, ha="right", fontsize=8.3)
    axes[1].set_ylabel("Operating-Point Sensitivity at Target FPR alpha")
    axes[1].set_title("B. Operating-Point Sensitivity Shift (Tong et al. Thresholds)")
    axes[1].legend(fontsize=7.3, ncol=2)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_multiseed_fig3_kappa0_foveation(cohorts: list[dict[str, Any]], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4), dpi=180)
    short_names = [
        c["cohort_name"]
        .replace("Gate05_Synthetic_MultiMechanism", "Gate 0.5")
        .replace("UCI_AI4I_2020_Maintenance", "AI4I")
        .replace("UCI_Taiwan_Credit_6m_Panel", "Taiwan")
        .replace("UCI_Diabetes_130US_Hospitals", "Diabetes")
        .replace("ULB_CreditCard_Fraud_Chronological", "ULB")
        .replace("Scania_APS_76k_Fleet", "Scania")
        .replace("Home_Credit_307k_7Table_Relational", "HomeCredit")
        .replace("IEEE_CIS_590k_MultiTable", "IEEE-CIS")
        .replace("NeurIPS_BAF_1M_Chronological", "BAF-1M")
        for c in cohorts
    ]
    x = np.arange(len(short_names))

    # Panel A: Equal-Budget Runtime Acquisition Policies (Test AUPRC across 5 key policies)
    pol_specs = [
        ("random_k", "Random-K", "#bdbdbd"),
        ("literal_conformal_01", "Literal Conformal {0,1}", "#ff7f0e"),
        ("uncertainty_k", "Uncertainty-K", "#9467bd"),
        ("voi_k", "VOI-K (Loss Reduction)", "#8c564b"),
        ("top_risk_k", "Top-Risk-K", "#17becf"),
        ("credo_foveation", "CREDO Upper-Tail", "#2ca02c"),
    ]
    wp = 0.135
    for idx, (pkey, plabel, pcol) in enumerate(pol_specs):
        means = [c["routing_policies"][pkey]["auprc"]["mean"] for c in cohorts]
        axes[0].bar(
            x + (idx - 2.5) * wp,
            means,
            wp,
            label=plabel,
            color=pcol,
            edgecolor="#333333",
            linewidth=0.35,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=16, ha="right", fontsize=8.4)
    axes[0].set_ylabel("Routed Test AUPRC at Equal Stage-2 Budget K")
    axes[0].set_title("A. Runtime Selective Acquisition: 6 Equal-Budget Policies")
    axes[0].legend(fontsize=7.6, ncol=2)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel B: Simulated Channel Access Savings (Discovery Pilot vs Deployment Stage-2)
    pilot_saved = [100.0 * (1.0 - c["pilot_fraction_measured"]["mean"]) for c in cohorts]
    s2_saved = [100.0 * c["stage2_sensor_savings"]["mean"] for c in cohorts]
    w2 = 0.34
    axes[1].bar(
        x - w2 / 2, pilot_saved, w2, label="Discovery Pilot Rows Saved (%)", color="#1f77b4"
    )
    axes[1].bar(
        x + w2 / 2, s2_saved, w2, label="Deployment Stage-2 Accesses Saved (%)", color="#2ca02c"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=16, ha="right", fontsize=8.4)
    axes[1].set_ylabel("Simulated Access Budget Saved (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("B. Simulated Access Efficiency (Discovery + Deployment)")
    axes[1].legend(fontsize=8.3)
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
        f"# Reconstructed Multi-Seed (`{n_seeds}x`) `CREDO-Bench` Campaign Report",
        "",
        (
            "## Table 1: Tier-1 Fixed-Table Suite "
            "(Stabilized Calibrated GBDT vs. Resampling & Ensembles)"
        ),
        "",
        (
            "| Dataset | $N$ | Test $n_+$ | Plan HW | Obs Wilson HW | Calibrated GBDT AUPRC | "
            "SMOTE+GBDT | EasyEnsemble | RUSBoost | NB $p$ (SMOTE / EEC) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in tier1:
        gb = r["calibrated_gbdt_auprc"]
        sm = r["smote_gbdt_auprc"]
        ee = r.get("easy_ensemble_auprc", sm)
        ru = r.get("rusboost_auprc", sm)
        p_ee = r.get("easy_ensemble_nb_one_sided_p", 1.0)
        obs_hw = r.get("observed_wilson_halfwidth", {}).get("mean", r["recall_ci_half_width"])
        md_lines.append(
            f"| {r['display_name']} | {r['n_rows']:,} | {r['n_pos_test']} | "
            f"$\\pm {r['recall_ci_half_width']:.3f}$ | $\\pm {obs_hw:.3f}$ | "
            f"${gb['mean']:.4f} \\pm {gb['sd']:.4f}$ | "
            f"${sm['mean']:.4f} \\pm {sm['sd']:.4f}$ | "
            f"${ee['mean']:.4f} \\pm {ee['sd']:.4f}$ | "
            f"${ru['mean']:.4f} \\pm {ru['sd']:.4f}$ | "
            f"{r['nadeau_bengio_one_sided_p']:.3f} / {p_ee:.3f} |"
        )

    ab_headers = " | ".join(lbl for _, lbl in ABLATION_LADDER_ORDER)
    ab_sep = "|---|" + "|".join("---:" for _ in ABLATION_LADDER_ORDER) + "|---:|"
    md_lines.extend(
        [
            "",
            (
                "## Table 2: Native-Channel Discovery & Complexity Ablation Matrix "
                "(Test AUPRC, Mean $\\pm$ SD)"
            ),
            "",
            f"| Cohort | {ab_headers} | Holm $p$ (A5 vs A0) |",
            ab_sep,
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
            + f" | {c['holm_prim_vs_base_auprc_p']:.5f} |"
        )

    md_lines.extend(
        [
            "",
            "## Table 3: Runtime Selective Acquisition — All 6 Equal-Budget Policies",
            "",
            (
                "| Cohort | Pilot Rows | Stage-2 Saved | Stage-1 Only | Random-K | "
                "Literal `{0,1}` | Uncertainty-K | VOI-K | Top-Risk-K | CREDO Upper-Tail | "
                "100% Selected S2 | Holm $p$ (vs `{0,1}`) | $p_2$ (vs Top-Risk-K) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for c in cohorts:
        md_lines.append(
            f"| **{c['cohort_name']}** | "
            f"{100.0 * c['pilot_fraction_measured']['mean']:.1f}% | "
            f"{100.0 * c['stage2_sensor_savings']['mean']:.1f}% | "
            f"${c['stage1_only_auprc']['mean']:.4f}$ | "
            f"${c['random_routing_auprc']['mean']:.4f}$ | "
            f"${c['literal_conformal_01_auprc']['mean']:.4f}$ | "
            f"${c['uncertainty_k_auprc']['mean']:.4f}$ | "
            f"${c['voi_k_auprc']['mean']:.4f}$ | "
            f"${c['top_risk_k_auprc']['mean']:.4f}$ | "
            f"**${c['credo_deficit_foveated_auprc']['mean']:.4f}$** | "
            f"${c['full_stage2_auprc']['mean']:.4f}$ | "
            f"{c['holm_foveation_vs_midband_fwer_p']:.5f} | "
            f"{c['nb_foveation_vs_top_risk_two_sided_p']:.4f} |"
        )

    (out_dir / "PAPER_CAMPAIGN_REPORT.md").write_text("\n".join(md_lines) + "\n")

    # Emit ready-to-include LaTeX booktabs table
    tex = [
        "% Auto-generated by imboreceipt bench-reconstructed",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        (
            "\\caption{Multi-seed (5$\\times$) native-channel discovery and complexity "
            "ablation Test AUPRC across all 9 cohorts.}"
        ),
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        (
            "Cohort & A0 Base & A4 Global MI & A4u Unm Pilot $G_m$ & A5 Strat Pilot $G_m$ & "
            "A7 Hybrid Lattice & All-Native Ref \\\\"
        ),
        "\\midrule",
    ]
    for c in cohorts:
        ab = c["ablation_ladder"]
        nm = c["cohort_name"].replace("_", "\\_")
        row_keys = (
            "calibrated_gbdt",
            "monolithic_global_mi",
            "monolithic_unmatched_pilot_cmi",
            "monolithic_matched_pilot_cmi",
            "full_credo_lattice",
            "unrestricted_all_native_gbdt",
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
