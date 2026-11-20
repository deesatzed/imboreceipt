"""Generate publication figures (Figures 1-4) for the v2-track0 strengthened manuscript."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path("/Users/o2satz/antigravity/imboreceipt")
    out_dir = root / "results" / "two_phase"
    man_dir = root / "manuscript"
    docs_assets = root / "docs" / "assets"

    v2 = json.loads((out_dir / "v2_campaign_summary.json").read_text(encoding="utf-8"))
    cohorts_v2 = v2["cohorts_v2"]

    order = [
        ("UCI_Diabetes_Discharge", "UCI Diabetes (Discharge)", "Diabetes"),
        ("Home_Credit_7Table", "Home Credit (7-Table)", "HomeCredit"),
        ("IEEE_CIS_Fraud", "IEEE-CIS Auth Stream", "IEEE-CIS"),
        ("Scania_APS_Fleet", "Scania APS Fleet", "Scania"),
        ("NeurIPS_BAF", "NeurIPS BAF Stream", "BAF"),
        ("UCI_Taiwan_Credit", "UCI Taiwan Credit", "Taiwan"),
    ]

    # --- Figure 2 (Updated v2): 2x3 Grid of All 6 Cohorts Streaming Frontiers ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(14.4, 7.6), dpi=220)
    pol_plot = [
        ("streaming_voi", "Mean-Centered VOI (P2)", "#2ca02c", "-o"),
        ("streaming_voi_uncentered", "Uncentered VOI (Heur.)", "#17becf", "--v"),
        ("streaming_top_risk", "Streaming Top-Risk", "#1f77b4", "--s"),
        ("streaming_uncertainty", "Boundary Uncertainty", "#9467bd", "-.^"),
        ("mondrian_ambiguity", "Mondrian {0,1}", "#ff7f0e", ":d"),
        ("streaming_random", "Streaming Random", "#888888", ":."),
    ]
    for idx, (ck, title_lbl, _) in enumerate(order):
        r_i, c_i = divmod(idx, 3)
        ax = axes2[r_i, c_i]
        cd = cohorts_v2[ck]
        for pk, plbl, pcol, psty in pol_plot:
            pts = cd["streaming_curves_v2"][pk]
            qs = [100.0 * p["realized_q"]["mean"] for p in pts]
            nbs = [p["net_benefit_per_1k"]["mean"] for p in pts]
            ax.plot(qs, nbs, psty, color=pcol, linewidth=1.55, markersize=4.2, label=plbl)
        qa = 100.0 * cd["voi_autonomous_centered"]["realized_query_rate"]["mean"]
        nba = cd["voi_autonomous_centered"]["net_benefit_per_1k"]["mean"]
        ax.scatter(
            [qa], [nba], color="#d62728", marker="*", s=125, zorder=6, label="Auto VOI > c_q"
        )
        ax.set_title(title_lbl, fontsize=10.2, fontweight="bold")
        if r_i == 1:
            ax.set_xlabel("Realized Streaming Query Rate (%)")
        if c_i == 0:
            ax.set_ylabel("Net Benefit per 1,000 (Net of Query Cost)")
        if idx == 0:
            ax.legend(fontsize=7.2, loc="lower right")
        ax.grid(alpha=0.3)

    fig2.tight_layout()
    for ext in (".pdf", ".png"):
        fig2.savefig(out_dir / f"fig2_streaming_budget_curves{ext}", bbox_inches="tight")
    plt.close(fig2)

    # --- Figure 3 (Updated v2): Conformal FDR Selection (True 20% Tie-Broken VOI + Auto VOI) ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(13.6, 4.7), dpi=220)
    x3 = np.arange(len(order))
    w3 = 0.17
    short_x = [sx for _, _, sx in order]

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
    axes3[0].bar(x3, fdp_v20, w3, label="CfBH + VOI (20% Tie-Broken)", color="#2ca02c")
    axes3[0].bar(x3 + w3, fdp_aut, w3, label="CfBH + Auto VOI > c_q", color="#ff7f0e")
    axes3[0].bar(x3 + 2 * w3, fdp_s2, w3, label="CfBH Full Stage-2 (100%)", color="#1f77b4")
    axes3[0].scatter(
        x3,
        tgt_fdr,
        color="#111111",
        marker="_",
        s=220,
        linewidths=2.2,
        zorder=7,
        label="Target FDR",
    )
    axes3[0].set_xticks(x3)
    axes3[0].set_xticklabels(short_x, fontsize=9.0)
    axes3[0].set_ylabel("Realized False Discovery Proportion (1 - PPV, %)")
    axes3[0].set_title("A. Precision / FDR Control: Mondrian {1} vs. Jin-Candes CfBH")
    axes3[0].legend(fontsize=7.3, ncol=2)
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
    axes3[1].bar(x3 + 0.5 * w3b, rec_aut, w3b, label="CfBH + Auto VOI > c_q", color="#ff7f0e")
    axes3[1].bar(x3 + 1.5 * w3b, rec_s2, w3b, label="CfBH Full Stage-2 (100%)", color="#1f77b4")
    axes3[1].set_xticks(x3)
    axes3[1].set_xticklabels(short_x, fontsize=9.0)
    axes3[1].set_ylabel("Certified Discovery Power / Recall (%) at Target FDR")
    axes3[1].set_title("B. Power Gain of Coherent Selective Acquisition Under CfBH")
    axes3[1].legend(fontsize=7.8)
    axes3[1].grid(axis="y", alpha=0.3)

    fig3.tight_layout()
    for ext in (".pdf", ".png"):
        fig3.savefig(out_dir / f"fig3_conformal_fdr_selection{ext}", bbox_inches="tight")
    plt.close(fig3)

    # --- Figure 4 (NEW v2): Pilot Curves (Neyman vs SRS) & Flip Decomposition ---
    fig4, axes4 = plt.subplots(1, 3, figsize=(14.8, 4.5), dpi=220)

    # Panel A: Pilot-fraction AUPRC retention on low-prevalence cohorts (Scania, IEEE-CIS, BAF)
    fracs = [5, 10, 15, 25, 50, 100]
    ney_keys = ["neyman_05", "neyman_10", "neyman_15", "neyman_25", "neyman_50", "full_100"]
    srs_keys = ["srs_05", "srs_10", "srs_15", "srs_25", "srs_50", "full_100"]
    panel_a_specs = [
        ("Scania_APS_Fleet", "Scania (pi=1.8%)", "#1f77b4"),
        ("IEEE_CIS_Fraud", "IEEE-CIS (pi=3.7%)", "#2ca02c"),
        ("NeurIPS_BAF", "NeurIPS BAF (pi=1.1%)", "#d62728"),
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
    axes4[0].set_ylabel("Test AUPRC Retained (% of 100%-Cohort Seq-CMI)")
    axes4[0].set_title("A. Pilot-Fraction Efficiency: Neyman vs. SRS")
    axes4[0].legend(fontsize=7.0, loc="lower right")
    axes4[0].grid(alpha=0.3)

    # Panel B: Flip-level Net Benefit Decomposition at q = 20% (Mean-Centered VOI vs Top-Risk)
    xb = np.arange(len(order))
    wb = 0.34
    voi_net_delta = []
    rsk_net_delta = []
    rsk_above_tau_waste = []
    voi_above_tau_waste = []
    for ck, _, _ in order:
        v20 = cohorts_v2[ck]["streaming_curves_v2"]["streaming_voi"][3]
        r20 = cohorts_v2[ck]["streaming_curves_v2"]["streaming_top_risk"][3]
        v_d = (
            v20["beneficial_flip_nb_gain_per_1k"]["mean"]
            - v20["harmful_flip_nb_loss_per_1k"]["mean"]
            - v20["query_fee_nb_drag_per_1k"]["mean"]
        )
        r_d = (
            r20["beneficial_flip_nb_gain_per_1k"]["mean"]
            - r20["harmful_flip_nb_loss_per_1k"]["mean"]
            - r20["query_fee_nb_drag_per_1k"]["mean"]
        )
        voi_net_delta.append(v_d)
        rsk_net_delta.append(r_d)
        voi_above_tau_waste.append(100.0 * v20["above_tau_unchanged_queried_rate"]["mean"])
        rsk_above_tau_waste.append(100.0 * r20["above_tau_unchanged_queried_rate"]["mean"])

    axes4[1].bar(xb - wb / 2, voi_net_delta, wb, label="Mean-Centered VOI (q=20%)", color="#2ca02c")
    axes4[1].bar(
        xb + wb / 2, rsk_net_delta, wb, label="Streaming Top-Risk (q=20%)", color="#1f77b4"
    )
    axes4[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes4[1].set_xticks(xb)
    axes4[1].set_xticklabels(short_x, fontsize=8.8)
    axes4[1].set_ylabel("Net Benefit Gain Over Stage 1 (per 1,000)")
    axes4[1].set_title("B. Stage-B Net Benefit Lift at q=20% (E2)")
    axes4[1].legend(fontsize=7.8)
    axes4[1].grid(axis="y", alpha=0.3)

    # Panel C: Autonomous VOI Query-Rate Adaptation Across Per-Query Cost Multiplier (0.25x..4x)
    q_mults = [0.25, 0.5, 1.0, 2.0, 4.0]
    fee_specs = [
        ("UCI_Diabetes_Discharge", "Diabetes", "#1f77b4", "-o"),
        ("Home_Credit_7Table", "Home Credit", "#9467bd", "-s"),
        ("IEEE_CIS_Fraud", "IEEE-CIS", "#2ca02c", "-^"),
        ("Scania_APS_Fleet", "Scania", "#d62728", "-d"),
        ("NeurIPS_BAF", "NeurIPS BAF", "#ff7f0e", "-v"),
    ]
    for ck, clbl, col, sty in fee_specs:
        f_rows = cohorts_v2[ck]["query_cost_sweep"]
        q_ad = [100.0 * r["auto_realized_q"]["mean"] for r in f_rows]
        axes4[2].plot(q_mults, q_ad, sty, color=col, linewidth=1.7, markersize=4.6, label=clbl)
    axes4[2].set_xscale("log", base=2)
    axes4[2].set_xticks(q_mults)
    axes4[2].set_xticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
    axes4[2].set_xlabel("Per-Instance Query Cost Multiplier (c_query)")
    axes4[2].set_ylabel("Autonomous VOI Realized Query Rate (%)")
    axes4[2].set_title("C. Autonomous Stopping Adaptation vs. Query Fee")
    axes4[2].legend(fontsize=7.6)
    axes4[2].grid(alpha=0.3)

    fig4.tight_layout()
    for ext in (".pdf", ".png"):
        fig4.savefig(out_dir / f"fig4_pilot_curve_and_flip_decomposition{ext}", bbox_inches="tight")
    plt.close(fig4)

    # Sync updated PDFs into manuscript/ and PNGs into docs/assets/
    for fn in (
        "fig2_streaming_budget_curves",
        "fig3_conformal_fdr_selection",
        "fig4_pilot_curve_and_flip_decomposition",
    ):
        shutil.copy2(out_dir / f"{fn}.pdf", man_dir / f"{fn}.pdf")
        shutil.copy2(out_dir / f"{fn}.png", docs_assets / f"{fn}.png")
    print("Generated and synced v2 Figures 2, 3, and 4!")


if __name__ == "__main__":
    main()
