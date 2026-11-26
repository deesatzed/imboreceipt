"""Generate 5 high-resolution (300 DPI) publication figures for Paper 1 (imboreceipt v2.2.0).

Figures:
  1. fig1_discovery_noninferiority_and_stress.(pdf|png):
     (A) Pilot-fraction TOST non-inferiority across rho = 15% vs 25% Neyman pilots.
     (B) Full 120-split leak-free train-fit stress-pool spurious channel waste (null/twin/echo %)
         across all 6 cohorts (R=20 splits each).
  2. fig2_streaming_budget_curves.(pdf|png):
     2x3 grid of Stage-B retrospective selective-querying Net Benefit frontiers across all 6
     cohorts + Auto VOI > c_query + Boundary Uncertainty + Covert-AFA + DB-CMI markers.
  3. fig3_conformal_fdr_selection.(pdf|png):
     (A) CfBH FDP vs Mondrian {1} across all 6 cohorts.
     (B) Certified discovery recall lift across query regimes.
     (C) 100-replication temporal shift audit (In-Era Exchangeable CfBH vs Stale Era-0 CfBH vs
         Unlabeled Jin-Candes WCS vs Labeled Burn-In n=2,500, with 95% error bars).
  4. fig4_pilot_curve_and_flip_decomposition.(pdf|png):
     (A) 5%..100% Neyman vs SRS pilot efficiency frontier.
     (B) Matched-budget q=20% Net Benefit lift: VOI vs Boundary Uncertainty vs Top-Risk.
     (C) c_query in {0.00, 0.01, 0.04, 0.08, 0.16} sweep: Auto VOI minus Full 100% Querying.
  5. fig5_clinical_triad_and_calibration.(pdf|png):
     Simulated acquisition scenario on Race-Blind UCI Diabetes 130-US (99,343 encounters):
     (A) Primary Stage-B Contrasts (Bagged VOI vs Full 100%, vs Boundary Uncertainty matched q,
         vs Boundary Uncertainty 20%, vs Uncentered VOI, and vs Stage-1 0%; 10k cluster 95% CIs).
     (B) Net Benefit across c_query in {0.00, 0.01, 0.02, 0.04, 0.08, 0.16} (Auto VOI vs Full
         100% vs Boundary Uncertainty matched q vs Stage-1 0%).
     (C) Race-Blind Subgroup Equity (Delta NB vs Full 100% & vs Stage-1; 10k cluster 95% CIs).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "results" / "two_phase"
    man_dir = root / "manuscript"
    docs_assets = root / "docs" / "assets"
    docs_assets.mkdir(parents=True, exist_ok=True)

    v2 = json.loads((out_dir / "v2_campaign_summary.json").read_text(encoding="utf-8"))
    v22 = json.loads((out_dir / "paper1_v2_2_remediation_results.json").read_text(encoding="utf-8"))

    cohorts_v2 = v2["cohorts_v2"]
    stress_120 = v22["block2_full_120split_leakfree_stress"]["leak_free_120split_stress"]
    can_h = v22["block3_stageb_comparators_and_feesweep"]["canonical_holdouts"]
    cfbh_100 = v22["block4_cfbh_shift_and_wcs"]["cfbh_wcs_burnin_100rep"]
    diab_rb = v22["block5_raceblind_diabetes_scenario"]

    order = [
        ("UCI_Diabetes_Discharge", "UCI Diabetes (Sim. Scenario)", "Diabetes"),
        ("Home_Credit_7Table", "Home Credit (7-Table)", "HomeCredit"),
        ("IEEE_CIS_Fraud", "IEEE-CIS Auth", "IEEE-CIS"),
        ("Scania_APS_Fleet", "Scania APS Fleet", "Scania"),
        ("NeurIPS_BAF", "NeurIPS BAF", "BAF"),
        ("UCI_Taiwan_Credit", "UCI Taiwan Credit", "Taiwan"),
    ]
    short_x = [sx for _, _, sx in order]

    # -----------------------------------------------------------------------
    # FIGURE 1: Stage-A Pilot Regime Non-Inferiority (15% vs 25%) & Full 120-Split Leak-Free Stress
    # -----------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(14.5, 5.0), dpi=300)
    y_pos = np.arange(len(order))

    v2_ckpt = out_dir / "v2_checkpoints"
    d15_m, d15_lo, d15_hi = [], [], []
    d25_m, d25_lo, d25_hi = [], [], []
    for ck, _, _ in order:
        arr15, arr25 = [], []
        for s_i in range(20):
            js = json.loads((v2_ckpt / f"{ck}_split{s_i:02d}.json").read_text(encoding="utf-8"))
            pc = js["pilot_curve"]
            arr15.append(pc["neyman_15"]["auprc"] - pc["full_100"]["auprc"])
            arr25.append(pc["neyman_25"]["auprc"] - pc["full_100"]["auprc"])
        for arr, ml, ll, hl in (
            (arr15, d15_m, d15_lo, d15_hi),
            (arr25, d25_m, d25_lo, d25_hi),
        ):
            a = np.asarray(arr)
            m_val = float(np.mean(a))
            se_nb = float(np.std(a, ddof=1)) * np.sqrt(1.0 / 20.0 + 0.15 / 0.55)
            ml.append(m_val)
            ll.append(m_val - 1.729 * se_nb)
            hl.append(m_val + 1.729 * se_nb)

    axes1[0].axvline(0.0, color="#222222", linewidth=1.0)
    axes1[0].axvline(
        -0.005, color="#888888", linestyle=":", linewidth=1.1, label="Strict NI (-0.005; 0/6 @ 15%)"
    )
    axes1[0].axvline(
        -0.015, color="#d62728", linestyle="--", linewidth=1.2, label="Pragmatic NI (-0.015)"
    )
    axes1[0].errorbar(
        d15_m,
        y_pos + 0.13,
        xerr=[np.array(d15_m) - np.array(d15_lo), np.array(d15_hi) - np.array(d15_m)],
        fmt="o",
        color="#1f77b4",
        capsize=3.5,
        linewidth=1.6,
        label="Neyman 15% Pilot (3/6 pass -0.015)",
    )
    axes1[0].errorbar(
        d25_m,
        y_pos - 0.13,
        xerr=[np.array(d25_m) - np.array(d25_lo), np.array(d25_hi) - np.array(d25_m)],
        fmt="s",
        color="#2ca02c",
        capsize=3.5,
        linewidth=1.6,
        label="Neyman 25% Pilot (5/6 pass -0.015)",
    )
    axes1[0].set_yticks(y_pos)
    axes1[0].set_yticklabels([lbl for _, lbl, _ in order], fontsize=9.1)
    axes1[0].invert_yaxis()
    axes1[0].set_xlim(-0.026, 0.013)
    axes1[0].set_ylim(7.15, -0.45)
    axes1[0].set_xlabel("Test AUPRC Difference vs. 100%-Cohort Seq-CMI (90% TOST CI)")
    axes1[0].set_title("A. Pilot-Fraction Regime & TOST Non-Inferiority", fontweight="bold")
    axes1[0].legend(fontsize=7.4, loc="lower left")
    axes1[0].grid(axis="x", alpha=0.3)

    # Panel B: Full 120-Split Leak-Free Train-Fit Stress Pool Spurious Selection
    x1b = np.arange(len(order))
    wb1 = 0.15

    def _spur120(ck: str, arm: str) -> float:
        d = stress_120[ck]["arms"][arm]
        return float(d["null_pct"] + d["twin_pct"] + d["echo_pct"])

    sp_prop = [_spur120(ck, "proposed_ht_holm") for ck, _, _ in order]
    sp_one = [_spur120(ck, "oneshot_cmi") for ck, _, _ in order]
    sp_bperm = [_spur120(ck, "cpi_pilot") for ck, _, _ in order]
    sp_ko = [_spur120(ck, "crt_knockoff_pilot") for ck, _, _ in order]
    sp_unwt = [_spur120(ck, "unweighted_pilot_no_lcb") for ck, _, _ in order]

    axes1[1].bar(
        x1b - 2 * wb1, sp_prop, wb1, label="Neyman Seq-CMI + Holm LCB (0.0%)", color="#2ca02c"
    )
    axes1[1].bar(x1b - wb1, sp_one, wb1, label="One-Shot CMI (Dup. Twins)", color="#ff7f0e")
    axes1[1].bar(x1b, sp_bperm, wb1, label="Baseline-Strat. Perm. (Dup. Twins)", color="#9467bd")
    axes1[1].bar(x1b + wb1, sp_ko, wb1, label="CRT / Knockoffs (Dup. Twins)", color="#1f77b4")
    axes1[1].bar(x1b + 2 * wb1, sp_unwt, wb1, label="Unweighted Pilot (No HT/LCB)", color="#d62728")
    axes1[1].set_xticks(x1b)
    axes1[1].set_xticklabels(short_x, fontsize=9.0)
    top_y1b = max(max(sp_one), max(sp_bperm), max(sp_ko), max(sp_unwt)) * 1.32
    axes1[1].set_ylim(0.0, max(50.0, top_y1b))
    axes1[1].set_ylabel("Spurious Budget Waste (Null + Dup. Twin + Echo, %)")
    axes1[1].set_title("B. Leak-Free 120-Split Stress-Pool Spurious Selection", fontweight="bold")
    axes1[1].legend(fontsize=7.3, loc="upper left")
    axes1[1].grid(axis="y", alpha=0.3)

    fig1.tight_layout()
    for ext in (".pdf", ".png"):
        fig1.savefig(
            out_dir / f"fig1_discovery_noninferiority_and_stress{ext}", bbox_inches="tight"
        )
    plt.close(fig1)

    # -----------------------------------------------------------------------
    # FIGURE 2: 2x3 Grid of All 6 Cohorts Selective Query Frontiers + DB-CMI & Covert-AFA
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(14.6, 7.8), dpi=300)
    pol_plot = [
        ("streaming_voi", "Centered VOI (Prop. 2)", "#2ca02c", "-o"),
        ("streaming_voi_uncentered", "Uncentered VOI", "#17becf", "--v"),
        ("streaming_uncertainty", "Boundary Uncertainty (-|p1-tau*|)", "#9467bd", "-.^"),
        ("streaming_top_risk", "Top-Risk", "#1f77b4", "--s"),
        ("mondrian_ambiguity", "Mondrian {0,1}", "#ff7f0e", ":d"),
        ("streaming_random", "Uniform Random", "#888888", ":."),
    ]
    for idx, (ck, title_lbl, _) in enumerate(order):
        r_i, c_i = divmod(idx, 3)
        ax = axes2[r_i, c_i]
        cd = cohorts_v2[ck]
        for pk, plbl, pcol, psty in pol_plot:
            pts = cd["streaming_curves_v2"][pk]
            qs = [100.0 * p["realized_q"]["mean"] for p in pts]
            nbs = [p["net_benefit_per_1k"]["mean"] for p in pts]
            ax.plot(qs, nbs, psty, color=pcol, linewidth=1.55, markersize=4.0, label=plbl)
        qa = 100.0 * cd["voi_autonomous_centered"]["realized_query_rate"]["mean"]
        nba = cd["voi_autonomous_centered"]["net_benefit_per_1k"]["mean"]
        ax.scatter(
            [qa], [nba], color="#d62728", marker="*", s=135, zorder=7, label="Auto VOI > c_q"
        )

        ch_d = can_h[ck]["arms_default_cq004"]
        s1_can = ch_d["stage1_0pct"]["nb"]
        s1_r20 = cd["streaming_curves_v2"]["streaming_voi"][0]["net_benefit_per_1k"]["mean"]
        afa_sh = s1_r20 + (ch_d["covert_afa_20pct_flat"]["nb"] - s1_can)
        dbc_sh = s1_r20 + (ch_d["dbcmi_afa_20pct_flat"]["nb"] - s1_can)
        ax.scatter(
            [100.0 * ch_d["covert_afa_20pct_flat"]["q_any"]],
            [afa_sh],
            color="#8c564b",
            marker="X",
            s=70,
            zorder=6,
            label="Covert-AFA (20%)",
        )
        ax.scatter(
            [100.0 * ch_d["dbcmi_afa_20pct_flat"]["q_any"]],
            [dbc_sh],
            color="#e377c2",
            marker="P",
            s=70,
            zorder=6,
            label="DB-CMI (20%)",
        )

        ax.set_title(title_lbl, fontsize=10.1, fontweight="bold")
        if r_i == 1:
            ax.set_xlabel("Realized Selective Query Rate (%)")
        if c_i == 0:
            ax.set_ylabel("Net Benefit per 1,000 (Net of Query Fee)")
        if idx == 0:
            ax.legend(fontsize=6.4, loc="lower right")
        ax.grid(alpha=0.3)

    fig2.tight_layout()
    for ext in (".pdf", ".png"):
        fig2.savefig(out_dir / f"fig2_streaming_budget_curves{ext}", bbox_inches="tight")
    plt.close(fig2)

    # -----------------------------------------------------------------------
    # FIGURE 3: Conformal FDR Selection + 100-Replication WCS vs Burn-In Shift Audit
    # -----------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=300)
    x3 = np.arange(len(order))
    w3 = 0.17

    fdp_m1 = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["mondrian_singleton1_fdp"]["mean"] for ck, _, _ in order
    ]
    fdp_s1 = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["cfbh_stage1"]["realized_fdp"]["mean"]
        for ck, _, _ in order
    ]
    fdp_v20 = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["cfbh_streaming_voi_20pct"]["realized_fdp"]["mean"]
        for ck, _, _ in order
    ]
    fdp_aut = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["cfbh_voi_autonomous"]["realized_fdp"]["mean"]
        for ck, _, _ in order
    ]
    fdp_s2 = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["cfbh_full_stage2"]["realized_fdp"]["mean"]
        for ck, _, _ in order
    ]
    tgt_fdr = [100.0 * cohorts_v2[ck]["cfbh_v2"]["target_fdr"] for ck, _, _ in order]

    axes3[0].bar(x3 - 2 * w3, fdp_m1, w3, label="Mondrian {1} (alpha=0.20)", color="#d62728")
    axes3[0].bar(x3 - w3, fdp_s1, w3, label="CfBH Stage-1 (0%)", color="#7f7f7f")
    axes3[0].bar(x3, fdp_v20, w3, label="CfBH + VOI (20%)", color="#2ca02c")
    axes3[0].bar(x3 + w3, fdp_aut, w3, label="CfBH + Auto VOI", color="#ff7f0e")
    axes3[0].bar(x3 + 2 * w3, fdp_s2, w3, label="CfBH Stage-2 (100%)", color="#1f77b4")
    axes3[0].scatter(
        x3,
        tgt_fdr,
        color="#111111",
        marker="_",
        s=180,
        linewidths=2.2,
        zorder=7,
        label="Target FDR",
    )
    axes3[0].set_xticks(x3)
    axes3[0].set_xticklabels(short_x, fontsize=8.4, rotation=15)
    axes3[0].set_ylim(0, 120)
    axes3[0].set_ylabel("Realized False Discovery Proportion (%)")
    axes3[0].set_title("A. CfBH vs. Mondrian {1} Across Cohorts", fontweight="bold", fontsize=9.7)
    axes3[0].legend(fontsize=6.7, ncol=2, loc="upper left")
    axes3[0].grid(axis="y", alpha=0.3)

    rec_s1 = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["cfbh_stage1"]["realized_recall"]["mean"]
        for ck, _, _ in order
    ]
    rec_v20 = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["cfbh_streaming_voi_20pct"]["realized_recall"]["mean"]
        for ck, _, _ in order
    ]
    rec_aut = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["cfbh_voi_autonomous"]["realized_recall"]["mean"]
        for ck, _, _ in order
    ]
    rec_s2 = [
        100.0 * cohorts_v2[ck]["cfbh_v2"]["cfbh_full_stage2"]["realized_recall"]["mean"]
        for ck, _, _ in order
    ]
    w3b = 0.20
    axes3[1].bar(x3 - 1.5 * w3b, rec_s1, w3b, label="CfBH Stage-1 (0%)", color="#7f7f7f")
    axes3[1].bar(x3 - 0.5 * w3b, rec_v20, w3b, label="CfBH + VOI (20%)", color="#2ca02c")
    axes3[1].bar(x3 + 0.5 * w3b, rec_aut, w3b, label="CfBH + Auto VOI", color="#ff7f0e")
    axes3[1].bar(x3 + 1.5 * w3b, rec_s2, w3b, label="CfBH Stage-2 (100%)", color="#1f77b4")
    axes3[1].set_xticks(x3)
    axes3[1].set_xticklabels(short_x, fontsize=8.4, rotation=15)
    axes3[1].set_ylabel("Certified Recall (%) at Target FDR")
    axes3[1].set_title("B. Certified Discovery Recall Gain", fontweight="bold", fontsize=9.7)
    axes3[1].legend(fontsize=7.2)
    axes3[1].grid(axis="y", alpha=0.3)

    # Panel C: 100-Replication Temporal Drift: Exchangeable vs Stale vs WCS vs Burn-In
    sh_keys = ["IEEE_CIS_Fraud", "NeurIPS_BAF"]
    sh_lbls = ["IEEE-CIS\n(Era 0 -> Era 3)", "NeurIPS BAF\n(Era 0 -> Era 3)"]
    x_sh = np.arange(2)
    w_sh = 0.19
    arms_sh = [
        ("in_era_exchangeable_cfbh", "In-Era Exchangeable", "#1f77b4", -1.5),
        ("stale_era0_unweighted_cfbh", "Stale Era-0 CfBH", "#d62728", -0.5),
        ("unlabeled_covariate_shift_wcs", "Unlabeled WCS (Jin-Candes)", "#9467bd", 0.5),
        ("labeled_burnin_2500_cfbh", "Labeled Burn-In (n=2,500)", "#2ca02c", 1.5),
    ]
    for ak, albl, acol, off in arms_sh:
        means = [100.0 * cfbh_100[k]["methods"][ak]["mean_fdr"] for k in sh_keys]
        errs = [1.96 * 100.0 * cfbh_100[k]["methods"][ak]["se_fdr"] for k in sh_keys]
        axes3[2].bar(
            x_sh + off * w_sh,
            means,
            w_sh,
            yerr=errs,
            capsize=2.8,
            label=albl,
            color=acol,
        )
    targ_sh = [100.0 * cfbh_100[k]["target_fdr"] for k in sh_keys]
    axes3[2].scatter(
        x_sh,
        targ_sh,
        color="#111111",
        marker="_",
        s=260,
        linewidths=2.4,
        zorder=7,
        label="Target FDR",
    )
    axes3[2].set_xticks(x_sh)
    axes3[2].set_xticklabels(sh_lbls, fontsize=8.8)
    axes3[2].set_ylim(0, 108)
    axes3[2].set_ylabel("Mean Realized FDR (%, 100 Reps +- 1.96 SE)")
    axes3[2].set_title("C. Temporal Shift: WCS vs. Burn-In", fontweight="bold", fontsize=9.7)
    axes3[2].legend(fontsize=6.7, loc="upper left")
    axes3[2].grid(axis="y", alpha=0.3)

    fig3.tight_layout()
    for ext in (".pdf", ".png"):
        fig3.savefig(out_dir / f"fig3_conformal_fdr_selection{ext}", bbox_inches="tight")
    plt.close(fig3)

    # -----------------------------------------------------------------------
    # FIGURE 4: Pilot Curves, Matched q=20% Routing vs BU, & c_query Sweep vs Full 100%
    # -----------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, 3, figsize=(15.0, 4.6), dpi=300)
    fracs = [5, 10, 15, 25, 50, 100]
    ney_keys = ["neyman_05", "neyman_10", "neyman_15", "neyman_25", "neyman_50", "full_100"]
    srs_keys = ["srs_05", "srs_10", "srs_15", "srs_25", "srs_50", "full_100"]
    panel_a_specs = [
        ("Scania_APS_Fleet", "Scania (1.8%)", "#1f77b4"),
        ("IEEE_CIS_Fraud", "IEEE-CIS (3.5%)", "#2ca02c"),
        ("NeurIPS_BAF", "NeurIPS BAF (1.1%)", "#d62728"),
    ]
    for ck, clbl, col in panel_a_specs:
        pc = cohorts_v2[ck]["pilot_fraction_curve"]
        full_v = pc["full_100"]["auprc"]["mean"]
        ney_y = [100.0 * pc[k]["auprc"]["mean"] / full_v for k in ney_keys]
        srs_y = [100.0 * pc[k]["auprc"]["mean"] / full_v for k in srs_keys]
        axes4[0].plot(
            fracs, ney_y, "-o", color=col, linewidth=1.9, markersize=4.8, label=f"{clbl} Neyman"
        )
        axes4[0].plot(
            fracs,
            srs_y,
            "--x",
            color=col,
            linewidth=1.3,
            alpha=0.75,
            markersize=4.8,
            label=f"{clbl} SRS",
        )
    axes4[0].axhline(100.0, color="#444444", linestyle=":", linewidth=1.0)
    axes4[0].set_xlabel("Phase-2 Pilot Fraction rho (%)")
    axes4[0].set_ylabel("Test AUPRC Retained (% of 100%-Cohort)")
    axes4[0].set_title("A. Pilot-Fraction Efficiency: Neyman vs. SRS", fontweight="bold")
    axes4[0].legend(fontsize=6.9, loc="lower right")
    axes4[0].grid(alpha=0.3)

    # Panel B: Canonical q=20% NB lift over Stage-1: VOI vs Boundary Uncertainty vs Top-Risk
    xb = np.arange(len(order))
    wb = 0.25
    v20_lift, bu20_lift, top20_lift = [], [], []
    for ck, _, _ in order:
        ad = can_h[ck]["arms_default_cq004"]
        s1_v = ad["stage1_0pct"]["nb"]
        v20_lift.append(ad["gh_voi_20pct"]["nb"] - s1_v)
        bu20_lift.append(ad["bu_20pct"]["nb"] - s1_v)
        top20_lift.append(ad["top_risk_20pct"]["nb"] - s1_v)
    axes4[1].bar(xb - wb, v20_lift, wb, label="Centered VOI (q=20%)", color="#2ca02c")
    axes4[1].bar(xb, bu20_lift, wb, label="Boundary Uncertainty (q=20%)", color="#9467bd")
    axes4[1].bar(xb + wb, top20_lift, wb, label="Top-Risk (q=20%)", color="#1f77b4")
    axes4[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes4[1].set_xticks(xb)
    axes4[1].set_xticklabels(short_x, fontsize=8.4, rotation=12)
    axes4[1].set_ylabel("Net Benefit Lift Over Stage 1 (/1,000)")
    axes4[1].set_title("B. Matched-Budget Routing (q=20%)", fontweight="bold")
    axes4[1].legend(fontsize=7.3)
    axes4[1].grid(axis="y", alpha=0.3)

    # Panel C: c_query sweep: Delta NB (Auto VOI minus Full 100% Querying) across all 6 cohorts
    cq_vals = [0.00, 0.01, 0.04, 0.08, 0.16]
    cq_specs = [
        ("UCI_Diabetes_Discharge", "Diabetes", "#1f77b4", "-o"),
        ("Home_Credit_7Table", "Home Credit", "#9467bd", "-s"),
        ("IEEE_CIS_Fraud", "IEEE-CIS", "#2ca02c", "-^"),
        ("Scania_APS_Fleet", "Scania", "#d62728", "-d"),
        ("NeurIPS_BAF", "NeurIPS BAF", "#ff7f0e", "-v"),
        ("UCI_Taiwan_Credit", "Taiwan", "#8c564b", "-P"),
    ]
    axes4[2].axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    for ck, clbl, col, sty in cq_specs:
        sw = can_h[ck]["c_query_sweep"]
        d_full = [r["delta_auto_vs_full100"]["delta_nb"] for r in sw]
        axes4[2].plot(cq_vals, d_full, sty, color=col, linewidth=1.65, markersize=4.5, label=clbl)
    axes4[2].set_xlabel("Per-Instance Query Cost c_query (Units of C_FP)")
    axes4[2].set_ylabel("Delta NB: Auto VOI minus Full 100% (/1,000)")
    axes4[2].set_title("C. Fee-Saving Advantage vs. Full Querying", fontweight="bold")
    axes4[2].legend(fontsize=7.1, ncol=2, loc="upper left")
    axes4[2].grid(alpha=0.3)

    fig4.tight_layout()
    for ext in (".pdf", ".png"):
        fig4.savefig(out_dir / f"fig4_pilot_curve_and_flip_decomposition{ext}", bbox_inches="tight")
    plt.close(fig4)

    # -----------------------------------------------------------------------
    # FIGURE 5: Race-Blind UCI Diabetes Simulated Acquisition Scenario
    # -----------------------------------------------------------------------
    fig5, axes5 = plt.subplots(1, 3, figsize=(15.4, 4.8), dpi=300)

    triad_rb = diab_rb["outer_19999_r20_triad_race_blind"]["bagged_voi_primary"]
    q_skip_pct = 100.0 * (1.0 - triad_rb["q_bag_gh_auto"])
    contrasts = [
        (
            f"Bagged VOI vs. Full 100%\n(Skips {q_skip_pct:.1f}% of Queries)",
            triad_rb["gh_auto_vs_full100"],
            "#2ca02c",
        ),
        (
            "Bagged VOI vs. Boundary\nUncertainty (Matched q)",
            triad_rb["gh_auto_vs_bu_matched"],
            "#9467bd",
        ),
        (
            "Bagged VOI (20%) vs.\nBoundary Uncertainty (20%)",
            triad_rb["gh20_vs_bu20"],
            "#ff7f0e",
        ),
        (
            "Bagged VOI vs.\nUncentered VOI (Coherence)",
            triad_rb["gh_auto_vs_uncentered"],
            "#17becf",
        ),
        (
            "Bagged VOI vs.\nStage-1 (0%, Total Lift)",
            triad_rb["gh_auto_vs_stage1"],
            "#1f77b4",
        ),
    ]
    y5a = np.arange(len(contrasts))
    m5a = [c[1]["delta_nb"] for c in contrasts]
    lo5a = [c[1]["delta_nb"] - c[1]["ci95_low"] for c in contrasts]
    hi5a = [c[1]["ci95_high"] - c[1]["delta_nb"] for c in contrasts]
    cols5a = [c[2] for c in contrasts]

    axes5[0].axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    for i_c in range(len(contrasts)):
        axes5[0].errorbar(
            [m5a[i_c]],
            [y5a[i_c]],
            xerr=[[lo5a[i_c]], [hi5a[i_c]]],
            fmt="o",
            color=cols5a[i_c],
            capsize=4.0,
            linewidth=2.0,
            markersize=6.5,
        )
    axes5[0].set_yticks(y5a)
    axes5[0].set_yticklabels([c[0] for c in contrasts], fontsize=8.2)
    axes5[0].invert_yaxis()
    axes5[0].set_xlabel("Paired Delta Net Benefit /1,000 (10k Patient-Cluster 95% CI)")
    axes5[0].set_title(
        "A. Primary Race-Blind Contrasts (N=19,999)", fontweight="bold", fontsize=9.6
    )
    axes5[0].grid(axis="x", alpha=0.3)

    # Panel B: c_query sweep on Race-Blind UCI Diabetes
    sw_d = diab_rb["c_query_sweep_race_blind"]
    cqs_d = [r["c_query"] for r in sw_d]
    axes5[1].plot(
        cqs_d,
        [r["nb_voi_auto"] for r in sw_d],
        "-o",
        color="#2ca02c",
        linewidth=2.0,
        label="Bagged Auto VOI",
    )
    axes5[1].plot(
        cqs_d,
        [r["nb_bu_matched"] for r in sw_d],
        "-.^",
        color="#9467bd",
        linewidth=1.7,
        label="Boundary Unc. (Matched q)",
    )
    axes5[1].plot(
        cqs_d,
        [r["nb_full_100"] for r in sw_d],
        "--s",
        color="#d62728",
        linewidth=1.7,
        label="Full Querying (100%)",
    )
    axes5[1].plot(
        cqs_d,
        [r["nb_stage1_0"] for r in sw_d],
        ":",
        color="#555555",
        linewidth=1.6,
        label="Stage-1 Only (0%)",
    )
    axes5[1].set_xlabel("Assigned Per-Lookup Cost c_query (Units of C_FP)")
    axes5[1].set_ylabel("Net Benefit per 1,000 Encounters")
    axes5[1].set_title("B. Diabetes c_query Sensitivity Sweep", fontweight="bold", fontsize=9.6)
    axes5[1].legend(fontsize=7.3, loc="best")
    axes5[1].grid(alpha=0.3)

    # Panel C: Race-Blind Subgroup Equity (Delta NB vs Full 100% AND vs Stage-1)
    sg_all = (
        diab_rb["subgroup_equity_and_calibration_race_blind"]["race"]["strata"]
        + diab_rb["subgroup_equity_and_calibration_race_blind"]["gender"]["strata"]
    )
    sg_keep = [r for r in sg_all if r["n_enc"] >= 100]
    y5c = np.arange(len(sg_keep))
    mf = [r["delta_nb_vs_full100"] for r in sg_keep]
    lof = [r["delta_nb_vs_full100"] - r["vs_full100_ci95"][0] for r in sg_keep]
    hif = [r["vs_full100_ci95"][1] - r["delta_nb_vs_full100"] for r in sg_keep]
    ms = [r["delta_nb_vs_s1"] for r in sg_keep]
    los = [r["delta_nb_vs_s1"] - r["vs_s1_ci95"][0] for r in sg_keep]
    his = [r["vs_s1_ci95"][1] - r["delta_nb_vs_s1"] for r in sg_keep]
    lbl5c = [f"{r['stratum']} (N={r['n_enc']:,})" for r in sg_keep]

    axes5[2].axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
    axes5[2].errorbar(
        mf,
        y5c - 0.14,
        xerr=[lof, hif],
        fmt="o",
        color="#2ca02c",
        capsize=3.0,
        linewidth=1.5,
        markersize=4.8,
        label="VOI vs. Full 100%",
    )
    axes5[2].errorbar(
        ms,
        y5c + 0.14,
        xerr=[los, his],
        fmt="s",
        color="#1f77b4",
        capsize=3.0,
        linewidth=1.5,
        markersize=4.8,
        label="VOI vs. Stage-1 (0%)",
    )
    axes5[2].set_yticks(y5c)
    axes5[2].set_yticklabels(lbl5c, fontsize=8.0)
    axes5[2].invert_yaxis()
    axes5[2].set_xlabel("Subgroup Delta NB /1,000 (10k Cluster 95% CI)")
    axes5[2].set_title("C. Race-Blind Subgroup Equity Audit", fontweight="bold", fontsize=9.6)
    axes5[2].legend(fontsize=7.2, loc="lower right")
    axes5[2].grid(axis="x", alpha=0.3)

    fig5.tight_layout()
    for ext in (".pdf", ".png"):
        fig5.savefig(out_dir / f"fig5_clinical_triad_and_calibration{ext}", bbox_inches="tight")
    plt.close(fig5)

    for fn in (
        "fig1_discovery_noninferiority_and_stress",
        "fig2_streaming_budget_curves",
        "fig3_conformal_fdr_selection",
        "fig4_pilot_curve_and_flip_decomposition",
        "fig5_clinical_triad_and_calibration",
    ):
        shutil.copy2(out_dir / f"{fn}.pdf", man_dir / f"{fn}.pdf")
        shutil.copy2(out_dir / f"{fn}.png", docs_assets / f"{fn}.png")
    print("Successfully generated and synced all 5 Paper 1 v2.2.0 figures (300 DPI)!")


if __name__ == "__main__":
    main()
