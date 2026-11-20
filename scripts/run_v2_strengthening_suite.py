"""Run the v2 Strengthening Suite on branch v2-track0 (reusing existing Stage-A checkpoints).

Computes across R=20 splits (120 primary splits) + 100 Sim-A replicates:
1. Track 0 (E0, E1, E2): Proposition 2 Mean-Centered Logit-Normal VOI + Prespecified
   Randomized Tie-Breaking + Uncentered-vs-Centered Router Head-to-Head + Flip-Level
   Net Benefit Decomposition + Updated CfBH at True 20% Query Rate.
2. Experiment Upgrade 1 (Sim A Split-Pilot): Disjoint 50/50 Split-Pilot Inference vs.
   4-Fold CV Two-Phase SE (verifying ~95% coverage of G_N(eta_fit) & <=5% null FWER).
3. Experiment Upgrade 2 (Pilot-Fraction Frontier): Neyman vs. SRS sequential discovery
   across rho in {5%, 10%, 15%, 25%, 50%, 100%} on all 6 real cohorts (R=20 splits).
4. Experiment Upgrade 3 (Decision-Curve Sensitivity): Bayes threshold tau* sweep
   (0.5x, 1x, 2x C_FN) and query-cost c_query sweep (0.25x..4x) across all 6 cohorts.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imboreceipt.bench.datasets import DEFAULT_CACHE_DIR
from imboreceipt.bench.fine_grained_cohorts import (
    FineGrainedCohort,
    load_fine_diabetes_discharge,
    load_fine_home_credit,
    load_fine_ieee_cis,
    load_fine_neurips_baf,
    load_fine_scania_aps,
    load_fine_taiwan_credit,
)
from imboreceipt.bench.two_phase_campaign import (
    _agg,
    _eval_downstream_predictions,
    _fit_downstream_calibrated_gbdt,
    _fit_oof_stage1_baseline,
)
from imboreceipt.core.bootstrap import (
    nadeau_bengio_ttest,
    tost_equivalence_and_noninferiority,
)
from imboreceipt.core.types import freeze_split
from imboreceipt.discover.selector import CandidateChannelSpec
from imboreceipt.discover.two_phase import (
    discover_sequential_conditional_gain,
    evaluate_split_pilot_step1_inference,
    sample_two_phase_pilot,
)
from imboreceipt.lattice.streaming_routing import StreamingSelectiveRouter

PRIMARY_COHORTS: tuple[str, ...] = (
    "UCI_Diabetes_Discharge",
    "Home_Credit_7Table",
    "IEEE_CIS_Fraud",
    "Scania_APS_Fleet",
    "NeurIPS_BAF",
    "UCI_Taiwan_Credit",
)


def _load_cohort(
    cohort_key: str, split_idx: int, n_splits: int, cache_dir: Path, seed: int
) -> FineGrainedCohort:
    if cohort_key == "UCI_Diabetes_Discharge":
        return load_fine_diabetes_discharge(
            cache_dir=cache_dir, include_stress_channels=False, seed=seed
        )
    if cohort_key == "Home_Credit_7Table":
        return load_fine_home_credit(
            cache_dir=cache_dir, include_stress_channels=False, max_rows=140000, seed=seed
        )
    if cohort_key == "IEEE_CIS_Fraud":
        return load_fine_ieee_cis(
            cache_dir=cache_dir,
            window_idx=split_idx,
            n_windows=n_splits,
            max_window_rows=130000,
            include_stress_channels=False,
            seed=seed,
        )
    if cohort_key == "Scania_APS_Fleet":
        return load_fine_scania_aps(cache_dir=cache_dir, include_stress_channels=False, seed=seed)
    if cohort_key == "NeurIPS_BAF":
        return load_fine_neurips_baf(
            cache_dir=cache_dir,
            window_idx=split_idx,
            n_windows=n_splits,
            max_window_rows=140000,
            include_stress_channels=False,
            seed=seed,
        )
    if cohort_key == "UCI_Taiwan_Credit":
        return load_fine_taiwan_credit(
            cache_dir=cache_dir,
            restrict_baseline_current=False,
            include_stress_channels=False,
            seed=seed,
        )
    raise ValueError(f"Unknown cohort_key: {cohort_key}")


def run_sim_a_split_pilot_comparison(
    out_path: Path, n_reps: int = 100, seed: int = 100
) -> dict[str, Any]:
    """Compare 4-fold CV vs. Disjoint 50/50 Split-Pilot Inference on 15% Neyman pilots."""
    cv_g: list[float] = []
    cv_se_tot: list[float] = []
    cv_se_p1: list[float] = []
    cv_null_fwer: list[float] = []
    cv_power: list[float] = []
    untrunc_no_lcb_fwer: list[float] = []
    pospart_no_lcb_fwer: list[float] = []

    sp_g: list[float] = []
    sp_se_tot: list[float] = []
    sp_se_p2: list[float] = []
    sp_target_cond: list[float] = []
    sp_cov_cond: list[float] = []
    sp_null_fwer: list[float] = []
    sp_power: list[float] = []

    census_g: list[float] = []

    for rep in range(n_reps):
        r_seed = seed + 1000 + rep
        rng = np.random.default_rng(r_seed)
        n = 6000
        x = rng.normal(0.0, 1.0, (n, 3))
        z1 = rng.normal(0.0, 1.0, n)
        eta = -3.4 + 0.9 * x[:, 0] - 0.5 * x[:, 1] + 1.05 * z1
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))

        X_b = pd.DataFrame(x, columns=["x0", "x1", "x2"])
        z_cols: dict[str, np.ndarray] = {"z_true": z1.astype(np.float32)}
        cands = [CandidateChannelSpec("ch_true", ("z_true",), 1.0, "true")]
        for j in range(1, 9):
            zn = rng.normal(0.0, 1.0, n).astype(np.float32)
            z_cols[f"z_null_{j}"] = zn
            cands.append(CandidateChannelSpec(f"ch_null_{j}", (f"z_null_{j}",), 1.0, "null"))
        Z_df = pd.DataFrame(z_cols)

        oof_b, val_m = _fit_oof_stage1_baseline(X_b, y, {}, None, None, seed=r_seed)
        pilot_full = sample_two_phase_pilot(y, oof_b, val_m, pilot_fraction=1.0, design="full")
        res_cen = discover_sequential_conditional_gain(
            X_b, Z_df, y, [cands[0]], oof_b, pilot_full, max_budget=1.0, seed=r_seed
        )
        census_g.append(float(res_cen.initial_scores["ch_true"]))

        pilot = sample_two_phase_pilot(
            y, oof_b, val_m, pilot_fraction=0.15, design="neyman", seed=r_seed
        )
        res_cv = discover_sequential_conditional_gain(
            X_b,
            Z_df,
            y,
            cands,
            oof_b,
            pilot,
            max_budget=3.0,
            multiplicity="holm",
            alpha_select=0.05,
            seed=r_seed,
        )
        cv_g.append(float(res_cv.initial_scores["ch_true"]))
        cv_se_tot.append(float(res_cv.initial_ses["ch_true"]))
        cv_se_p1.append(
            float(res_cv.step_trace[0].se_phase1) if res_cv.step_trace else cv_se_tot[-1]
        )
        cv_null_fwer.append(float(any(cid.startswith("ch_null_") for cid in res_cv.selected_ids)))
        cv_power.append(float("ch_true" in res_cv.selected_ids))
        untrunc_no_lcb_fwer.append(
            float(any(res_cv.initial_scores.get(f"ch_null_{j}", 0.0) > 0.0 for j in range(1, 9)))
        )
        pospart_no_lcb_fwer.append(
            float(
                any(
                    res_cv.initial_pospart_scores.get(f"ch_null_{j}", 0.0) > 0.0
                    for j in range(1, 9)
                )
            )
        )

        # Disjoint 50/50 split-pilot evaluation on the IDENTICAL 15% Neyman pilot
        res_sp = evaluate_split_pilot_step1_inference(
            X_b, Z_df, y, cands, oof_b, pilot, alpha_select=0.05, seed=r_seed
        )
        g_s = float(res_sp["scores"]["ch_true"])
        se_s = float(res_sp["ses_total"]["ch_true"])
        se2_s = float(res_sp["ses_phase2"]["ch_true"])
        targ_s = float(res_sp["census_targets_conditional"]["ch_true"])
        sp_g.append(g_s)
        sp_se_tot.append(se_s)
        sp_se_p2.append(se2_s)
        sp_target_cond.append(targ_s)
        sp_cov_cond.append(float(abs(g_s - targ_s) <= 1.96 * se_s))
        sp_null_fwer.append(
            float(any(cid.startswith("ch_null_") for cid in res_sp["selected_ids"]))
        )
        sp_power.append(float("ch_true" in res_sp["selected_ids"]))

    cv_mean = float(np.mean(cv_g))
    sp_mean = float(np.mean(sp_g))
    cen_mean = float(np.mean(census_g))
    cv_cov_uncond = [
        float(abs(g - cv_mean) <= 1.96 * s) for g, s in zip(cv_g, cv_se_tot, strict=True)
    ]
    cv_cov_p1 = [
        float(abs(g - cv_mean) <= 1.96 * s1) for g, s1 in zip(cv_g, cv_se_p1, strict=True)
    ]
    cv_cov_cen = [
        float(abs(g - cen_mean) <= 1.96 * s) for g, s in zip(cv_g, cv_se_tot, strict=True)
    ]
    sp_cov_uncond = [
        float(abs(g - sp_mean) <= 1.96 * s) for g, s in zip(sp_g, sp_se_tot, strict=True)
    ]

    payload = {
        "n_replicates": n_reps,
        "pospart_no_lcb_null_fwer": _agg(pospart_no_lcb_fwer),
        "untruncated_no_lcb_null_fwer": _agg(untrunc_no_lcb_fwer),
        "cv_4fold_full_pilot": {
            "mean_gain": cv_mean,
            "empirical_sd": float(np.std(cv_g, ddof=1)),
            "mean_se_total": float(np.mean(cv_se_tot)),
            "ci95_cov_unconditional": _agg(cv_cov_uncond),
            "ci95_cov_phase1_only": _agg(cv_cov_p1),
            "ci95_cov_vs_census_mean": _agg(cv_cov_cen),
            "null_fwer": _agg(cv_null_fwer),
            "true_channel_power": _agg(cv_power),
        },
        "split_pilot_50_50": {
            "mean_gain": sp_mean,
            "empirical_sd": float(np.std(sp_g, ddof=1)),
            "mean_se_total": float(np.mean(sp_se_tot)),
            "mean_se_phase2": float(np.mean(sp_se_p2)),
            "ci95_cov_data_adaptive_census": _agg(sp_cov_cond),
            "ci95_cov_unconditional": _agg(sp_cov_uncond),
            "null_fwer": _agg(sp_null_fwer),
            "true_channel_power": _agg(sp_power),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved Sim-A v2 split-pilot results:", out_path)
    return payload


def run_v2_cohort_split(
    cohort_key: str,
    split_idx: int,
    base_ckpt_dir: Path,
    v2_ckpt_dir: Path,
    n_splits: int = 20,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    base_seed: int = 100,
) -> dict[str, Any]:
    """Run Track-0 Stage-B repair + Pilot-Fraction Frontier + Decision-Curve Sweep for 1 split."""
    out_file = v2_ckpt_dir / f"{cohort_key}_split{split_idx:02d}.json"
    if out_file.exists():
        return json.loads(out_file.read_text(encoding="utf-8"))

    base_file = base_ckpt_dir / f"{cohort_key}_split{split_idx:02d}.json"
    base_ckpt = json.loads(base_file.read_text(encoding="utf-8"))

    seed = base_seed + split_idx
    coh = _load_cohort(cohort_key, split_idx, n_splits, cache_dir, seed)

    split_df = coh.X_base.copy()
    split_df["__y__"] = coh.y
    grp_col = None
    time_col = None
    if coh.group_ids is not None:
        split_df["__grp__"] = coh.group_ids
        grp_col = "__grp__"
    if coh.time_order is not None:
        split_df["__time__"] = coh.time_order
        time_col = "__time__"

    sp = freeze_split(split_df, "__y__", group_col=grp_col, time_col=time_col, seed=seed)
    tr_i = np.asarray(sp.train_idx, dtype=int)
    ca_i = np.asarray(sp.cal_idx, dtype=int)
    te_i = np.asarray(sp.test_idx, dtype=int)

    Xb_tr, Xb_ca, Xb_te = (
        coh.X_base.iloc[tr_i].reset_index(drop=True),
        coh.X_base.iloc[ca_i].reset_index(drop=True),
        coh.X_base.iloc[te_i].reset_index(drop=True),
    )
    Z_tr, Z_ca, Z_te = (
        coh.Z_pool.iloc[tr_i].reset_index(drop=True),
        coh.Z_pool.iloc[ca_i].reset_index(drop=True),
        coh.Z_pool.iloc[te_i].reset_index(drop=True),
    )
    y_tr, y_ca, y_te = coh.y[tr_i], coh.y[ca_i], coh.y[te_i]
    grp_tr = coh.group_ids[tr_i] if coh.group_ids is not None else None
    tim_tr = coh.time_order[tr_i] if coh.time_order is not None else None

    p1_ca, p1_te, _ = _fit_downstream_calibrated_gbdt(
        Xb_tr, y_tr, Xb_ca, y_ca, Xb_te, coh.base_semantic_types, seed=seed
    )
    base_metrics = _eval_downstream_predictions(y_te, p1_te, coh, n_selected_channels=0)

    oof_p0, oof_valid = _fit_oof_stage1_baseline(
        Xb_tr, y_tr, coh.base_semantic_types, grp_tr, tim_tr, seed=seed
    )

    cand_map = {c.channel_id: c for c in coh.candidates}
    pred_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}

    def _fit_ids(sel_ids: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        key = tuple(sorted(sel_ids))
        if not key:
            return p1_ca, p1_te, base_metrics
        if key in pred_cache:
            return pred_cache[key]
        specs = [cand_map[cid] for cid in sel_ids]
        cols = [col for s in specs for col in s.columns]
        pca_m, pte_m, _ = _fit_downstream_calibrated_gbdt(
            pd.concat([Xb_tr, Z_tr[cols]], axis=1),
            y_tr,
            pd.concat([Xb_ca, Z_ca[cols]], axis=1),
            y_ca,
            pd.concat([Xb_te, Z_te[cols]], axis=1),
            coh.base_semantic_types,
            seed=seed,
        )
        met_m = _eval_downstream_predictions(y_te, pte_m, coh, n_selected_channels=len(specs))
        pred_cache[key] = (pca_m, pte_m, met_m)
        return pca_m, pte_m, met_m

    # Pre-populate from saved checkpointed selections (10%, 15%, 100% Neyman & 15% SRS)
    saved_native = base_ckpt["native_arms"]
    sel_15_ney = list(saved_native["neyman_15pct_seq"]["selected_ids"])
    p2_ca_prim, p2_te_prim, _ = _fit_ids(sel_15_ney)

    # 1. Track 0 (E0 + E1 + E2) + Upgrade 3 (Contract & Query-Fee Sensitivity)
    router_v2 = StreamingSelectiveRouter(
        coh.contract, per_instance_query_cost=coh.per_instance_query_cost, mean_centered=True
    ).fit(y_ca, p1_ca, p2_ca_prim)
    streaming_v2 = router_v2.evaluate_budget_frontier(y_te, p1_te, p2_te_prim, seed=seed)

    # 2. Upgrade 2: Complete Pilot-Fraction Efficiency Curve rho in {5%, 10%, 15%, 25%, 50%, 100%}
    pilot_curve: dict[str, dict[str, Any]] = {}

    # Reuse already-computed arms where identical
    for tag, arm_key in [
        ("neyman_10", "neyman_10pct_seq"),
        ("neyman_15", "neyman_15pct_seq"),
        ("srs_15", "srs_15pct_seq"),
        ("full_100", "full_100pct_seq"),
    ]:
        arm_d = saved_native[arm_key]
        pilot_curve[tag] = {
            "auprc": float(arm_d["auprc"]),
            "net_benefit_per_1k": float(arm_d["net_benefit_per_1k"]),
            "n_selected": int(len(arm_d["selected_ids"])),
            "selected_ids": list(arm_d["selected_ids"]),
            "pilot_fraction": float(arm_d.get("pilot_fraction", 1.0)),
        }

    new_pilot_specs = [
        ("neyman_05", "neyman", 0.05),
        ("neyman_25", "neyman", 0.25),
        ("neyman_50", "neyman", 0.50),
        ("srs_05", "srs", 0.05),
        ("srs_10", "srs", 0.10),
        ("srs_25", "srs", 0.25),
        ("srs_50", "srs", 0.50),
    ]
    for p_tag, p_des, p_frac in new_pilot_specs:
        pil = sample_two_phase_pilot(
            y_tr, oof_p0, oof_valid, pilot_fraction=p_frac, design=p_des, seed=seed
        )
        d_res = discover_sequential_conditional_gain(
            Xb_tr,
            Z_tr,
            y_tr,
            coh.candidates,
            oof_p0,
            pil,
            coh.acquisition_budget,
            base_semantic_types=coh.base_semantic_types,
            groups_train=grp_tr,
            time_order_train=tim_tr,
            method_name=p_tag,
            seed=seed,
        )
        _, _, met_p = _fit_ids(d_res.selected_ids)
        pilot_curve[p_tag] = {
            "auprc": float(met_p["auprc"]),
            "net_benefit_per_1k": float(met_p["net_benefit_per_1k"]),
            "n_selected": len(d_res.selected_ids),
            "selected_ids": d_res.selected_ids,
            "pilot_fraction": float(d_res.pilot_fraction),
        }

    result = {
        "cohort_key": cohort_key,
        "split_idx": split_idx,
        "n_train": int(tr_i.size),
        "n_cal": int(ca_i.size),
        "n_test": int(te_i.size),
        "pilot_curve": pilot_curve,
        "streaming_v2": streaming_v2,
    }
    v2_ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    gc.collect()
    return result


def summarize_all_v2(
    base_summary_path: Path,
    v2_ckpt_dir: Path,
    sim_a_v2_path: Path,
    out_summary_path: Path,
    n_splits: int = 20,
) -> dict[str, Any]:
    """Aggregate all 120 v2 splits + Sim A split-pilot into a unified v2 summary JSON."""
    base_sum = json.loads(base_summary_path.read_text(encoding="utf-8"))
    sim_a_v2 = json.loads(sim_a_v2_path.read_text(encoding="utf-8"))

    cohorts_v2: dict[str, Any] = {}
    for c_idx, c_key in enumerate(PRIMARY_COHORTS):
        splits = [
            json.loads((v2_ckpt_dir / f"{c_key}_split{s:02d}.json").read_text(encoding="utf-8"))
            for s in range(n_splits)
        ]
        n_tr_mean = int(round(float(np.mean([sp["n_train"] for sp in splits]))))
        n_te_mean = int(round(float(np.mean([sp["n_test"] for sp in splits]))))

        # 1. Pilot-fraction curve summary + TOST vs full_100 + NB tests vs SRS
        p_tags = [
            "neyman_05",
            "neyman_10",
            "neyman_15",
            "neyman_25",
            "neyman_50",
            "srs_05",
            "srs_10",
            "srs_15",
            "srs_25",
            "srs_50",
            "full_100",
        ]
        full_ap = [sp["pilot_curve"]["full_100"]["auprc"] for sp in splits]
        p_summary: dict[str, Any] = {}
        for pt in p_tags:
            aps = [sp["pilot_curve"][pt]["auprc"] for sp in splits]
            nbs = [sp["pilot_curve"][pt]["net_benefit_per_1k"] for sp in splits]
            nsels = [float(sp["pilot_curve"][pt]["n_selected"]) for sp in splits]
            entry: dict[str, Any] = {
                "auprc": _agg(aps),
                "net_benefit_per_1k": _agg(nbs),
                "n_selected": _agg(nsels),
            }
            if pt != "full_100":
                diffs_100 = [a - f for a, f in zip(aps, full_ap, strict=True)]
                tost_015 = tost_equivalence_and_noninferiority(
                    diffs_100, margin=0.015, n_test=n_te_mean, n_train=n_tr_mean
                )
                tost_005 = tost_equivalence_and_noninferiority(
                    diffs_100, margin=0.005, n_test=n_te_mean, n_train=n_tr_mean
                )
                entry["delta_vs_full100_auprc"] = {
                    "mean": tost_015.mean_delta,
                    "ci90_low": tost_015.ci90_low,
                    "ci90_high": tost_015.ci90_high,
                    "p_ni_015": tost_015.p_noninferiority,
                    "p_ni_005": tost_005.p_noninferiority,
                }
            p_summary[pt] = entry

        neyman_vs_srs: dict[str, Any] = {}
        for frac_s in ("05", "10", "15", "25", "50"):
            n_aps = [sp["pilot_curve"][f"neyman_{frac_s}"]["auprc"] for sp in splits]
            s_aps = [sp["pilot_curve"][f"srs_{frac_s}"]["auprc"] for sp in splits]
            nb_res = nadeau_bengio_ttest(
                [a - b for a, b in zip(n_aps, s_aps, strict=True)],
                n_test=n_te_mean,
                n_train=n_tr_mean,
            )
            neyman_vs_srs[f"rho_{frac_s}pct"] = {
                "mean_diff": nb_res.mean_delta,
                "se_nb": nb_res.corrected_se,
                "p_one_sided": nb_res.p_value_one_sided,
                "p_two_sided": nb_res.p_value_two_sided,
            }

        # 2. Stage-B Streaming Curves + Flip Decomposition + E0/E1 Audit
        pol_names = list(splits[0]["streaming_v2"]["curves"].keys())
        q_grid = splits[0]["streaming_v2"]["query_rates"]
        curves_agg: dict[str, list[dict[str, Any]]] = {}
        for pol in pol_names:
            pts_list: list[dict[str, Any]] = []
            for q_i, q_val in enumerate(q_grid):
                pts = [sp["streaming_v2"]["curves"][pol][q_i] for sp in splits]
                pts_list.append(
                    {
                        "nominal_q": float(q_val),
                        "realized_q": _agg([float(p["realized_query_rate"]) for p in pts]),
                        "auprc": _agg([float(p["auprc"]) for p in pts]),
                        "net_benefit_per_1k": _agg([float(p["net_benefit_per_1k"]) for p in pts]),
                        "cfbh_recall": _agg([float(p["cfbh_recall"]) for p in pts]),
                        "cfbh_fdp": _agg([float(p["cfbh_realized_fdp"]) for p in pts]),
                        "beneficial_flip_nb_gain_per_1k": _agg(
                            [float(p["beneficial_flip_nb_gain_per_1k"]) for p in pts]
                        ),
                        "harmful_flip_nb_loss_per_1k": _agg(
                            [float(p["harmful_flip_nb_loss_per_1k"]) for p in pts]
                        ),
                        "query_fee_nb_drag_per_1k": _agg(
                            [float(p["query_fee_nb_drag_per_1k"]) for p in pts]
                        ),
                        "unchanged_queried_rate": _agg(
                            [float(p["unchanged_queried_rate"]) for p in pts]
                        ),
                        "above_tau_unchanged_queried_rate": _agg(
                            [float(p["above_tau_unchanged_queried_rate"]) for p in pts]
                        ),
                    }
                )
            curves_agg[pol] = pts_list

        def _agg_point_dict(
            key_name: str, _splits: list[dict[str, Any]] = splits
        ) -> dict[str, Any]:
            pts = [sp["streaming_v2"][key_name] for sp in _splits]
            return {
                "realized_query_rate": _agg([float(p["realized_query_rate"]) for p in pts]),
                "auprc": _agg([float(p["auprc"]) for p in pts]),
                "net_benefit_per_1k": _agg([float(p["net_benefit_per_1k"]) for p in pts]),
                "cfbh_recall": _agg([float(p["cfbh_recall"]) for p in pts]),
                "cfbh_fdp": _agg([float(p["cfbh_realized_fdp"]) for p in pts]),
                "beneficial_flip_nb_gain_per_1k": _agg(
                    [float(p["beneficial_flip_nb_gain_per_1k"]) for p in pts]
                ),
                "harmful_flip_nb_loss_per_1k": _agg(
                    [float(p["harmful_flip_nb_loss_per_1k"]) for p in pts]
                ),
                "query_fee_nb_drag_per_1k": _agg(
                    [float(p["query_fee_nb_drag_per_1k"]) for p in pts]
                ),
                "unchanged_queried_rate": _agg(
                    [float(p["unchanged_queried_rate"]) for p in pts]
                ),
            }

        e0_keys = list(splits[0]["streaming_v2"]["e0_e1_diagnostics"].keys())
        e0_agg = {
            ek: _agg([float(sp["streaming_v2"]["e0_e1_diagnostics"][ek]) for sp in splits])
            for ek in e0_keys
        }

        cf_keys = (
            "cfbh_stage1",
            "cfbh_streaming_voi_20pct",
            "cfbh_voi_autonomous",
            "cfbh_full_stage2",
        )
        cfbh_agg: dict[str, Any] = {
            "target_fdr": float(
                splits[0]["streaming_v2"]["conformal_fdr_comparison"]["target_fdr"]
            ),
            "mondrian_singleton1_fdp": _agg(
                [
                    float(
                        sp["streaming_v2"]["conformal_fdr_comparison"][
                            "mondrian_singleton1_alpha20_fdp"
                        ]
                    )
                    for sp in splits
                ]
            ),
            "mondrian_singleton1_recall": _agg(
                [
                    float(
                        sp["streaming_v2"]["conformal_fdr_comparison"][
                            "mondrian_singleton1_alpha20_recall"
                        ]
                    )
                    for sp in splits
                ]
            ),
        }
        for cfk in cf_keys:
            cfbh_agg[cfk] = {
                "realized_fdp": _agg(
                    [
                        float(
                            sp["streaming_v2"]["conformal_fdr_comparison"][cfk]["realized_fdp"]
                        )
                        for sp in splits
                    ]
                ),
                "realized_recall": _agg(
                    [
                        float(
                            sp["streaming_v2"]["conformal_fdr_comparison"][cfk]["realized_recall"]
                        )
                        for sp in splits
                    ]
                ),
                "selection_rate": _agg(
                    [
                        float(
                            sp["streaming_v2"]["conformal_fdr_comparison"][cfk]["selection_rate"]
                        )
                        for sp in splits
                    ]
                ),
            }

        # 3. Contract & Query-Cost Sensitivity Sweeps
        tau_len = len(splits[0]["streaming_v2"]["contract_sensitivity"]["tau_star_sweep"])
        tau_agg = []
        for idx_t in range(tau_len):
            rows_t = [
                sp["streaming_v2"]["contract_sensitivity"]["tau_star_sweep"][idx_t]
                for sp in splits
            ]
            tau_agg.append(
                {
                    "cost_fn_multiplier": float(rows_t[0]["cost_fn_multiplier"]),
                    "tau_star": float(rows_t[0]["tau_star"]),
                    "nb_stage1_0pct": _agg([float(r["nb_stage1_0pct"]) for r in rows_t]),
                    "nb_top_risk_20pct": _agg([float(r["nb_top_risk_20pct"]) for r in rows_t]),
                    "nb_voi_20pct": _agg([float(r["nb_voi_20pct"]) for r in rows_t]),
                    "nb_voi_autonomous": _agg([float(r["nb_voi_autonomous"]) for r in rows_t]),
                    "auto_realized_q": _agg([float(r["auto_realized_q"]) for r in rows_t]),
                    "nb_full_100pct": _agg([float(r["nb_full_100pct"]) for r in rows_t]),
                }
            )

        fee_len = len(splits[0]["streaming_v2"]["contract_sensitivity"]["query_cost_sweep"])
        fee_agg = []
        for idx_f in range(fee_len):
            rows_f = [
                sp["streaming_v2"]["contract_sensitivity"]["query_cost_sweep"][idx_f]
                for sp in splits
            ]
            fee_agg.append(
                {
                    "query_cost_multiplier": float(rows_f[0]["query_cost_multiplier"]),
                    "c_query": float(rows_f[0]["c_query"]),
                    "nb_stage1_0pct": _agg([float(r["nb_stage1_0pct"]) for r in rows_f]),
                    "nb_voi_20pct": _agg([float(r["nb_voi_20pct"]) for r in rows_f]),
                    "nb_voi_autonomous": _agg([float(r["nb_voi_autonomous"]) for r in rows_f]),
                    "auto_realized_q": _agg([float(r["auto_realized_q"]) for r in rows_f]),
                    "nb_full_100pct": _agg([float(r["nb_full_100pct"]) for r in rows_f]),
                }
            )

        cohorts_v2[c_key] = {
            "base_cohort_meta": base_sum["cohorts"][c_idx].get("cohort_name", c_key),
            "pilot_fraction_curve": p_summary,
            "neyman_vs_srs_tests": neyman_vs_srs,
            "streaming_curves_v2": curves_agg,
            "voi_autonomous_centered": _agg_point_dict("voi_autonomous"),
            "voi_autonomous_uncentered": _agg_point_dict("voi_uncentered_autonomous"),
            "e0_e1_diagnostics": e0_agg,
            "cfbh_v2": cfbh_agg,
            "tau_star_sweep": tau_agg,
            "query_cost_sweep": fee_agg,
        }

    final_out = {
        "sim_A_split_pilot_v2": sim_a_v2,
        "cohorts_v2": cohorts_v2,
    }
    out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    out_summary_path.write_text(json.dumps(final_out, indent=2), encoding="utf-8")
    print("Wrote unified v2 summary:", out_summary_path)
    return final_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sim_a", "cohort", "summarize"], required=True)
    parser.add_argument("--cohorts", nargs="*", default=list(PRIMARY_COHORTS))
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("results/two_phase"))
    args = parser.parse_args()

    v2_ckpt_dir = args.out_dir / "v2_checkpoints"
    sim_a_path = args.out_dir / "v2_sim_a_split_pilot.json"

    if args.mode == "sim_a":
        run_sim_a_split_pilot_comparison(sim_a_path, n_reps=100, seed=100)
    elif args.mode == "cohort":
        base_ckpt_dir = args.out_dir / "checkpoints"
        for ck in args.cohorts:
            for s in range(args.n_splits):
                run_v2_cohort_split(ck, s, base_ckpt_dir, v2_ckpt_dir, n_splits=args.n_splits)
                print(f"[{ck}] completed v2 split {s + 1}/{args.n_splits}", flush=True)
    elif args.mode == "summarize":
        summarize_all_v2(
            args.out_dir / "two_phase_campaign_summary.json",
            v2_ckpt_dir,
            sim_a_path,
            args.out_dir / "v2_campaign_summary.json",
            n_splits=args.n_splits,
        )


if __name__ == "__main__":
    main()
