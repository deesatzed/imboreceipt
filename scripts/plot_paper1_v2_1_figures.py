"""Generate 5 high-resolution (300 DPI) publication figures for Paper 1 (imboreceipt v2.1).

Figures:
  1. fig1_discovery_noninferiority_and_stress.(pdf|png):
     (A) Pilot-fraction TOST non-inferiority across rho = 15% vs 25% Neyman pilots.
     (B) Leak-free train-fit stress-pool spurious channel waste (null/twin/echo %) across all
         comparators including Chamma Residualized CPI & Baseline-Stratified Perm.
  2. fig2_streaming_budget_curves.(pdf|png):
     2x3 grid of Stage-B streaming Net Benefit frontiers across all 6 cohorts + Auto VOI > c_query
     stars + Covert-AFA (sequential instancewise greedy CMI) canonical markers.
  3. fig3_conformal_fdr_selection.(pdf|png):
     (A) Exchangeable CfBH FDP control vs Mondrian {1}.
     (B) Certified discovery recall lift under CfBH.
     (C) Temporal distribution shift audit (Era 0 -> Era 3 stale calibration vs n=2,500 burn-in).
  4. fig4_pilot_curve_and_flip_decomposition.(pdf|png):
     (A) 5%..100% Neyman vs SRS pilot efficiency frontier.
     (B) Exact decision-flip decomposition at q=20%.
     (C) 16x autonomous query-fee adaptation.
  5. fig5_clinical_triad_and_calibration.(pdf|png):
     UCI Diabetes 130-US clinical discharge audit (99,343 encounters, 69,990 patients):
     (A) R=20 Deployment Estimator Triad (Primary Bagged VOI, Sym. Ens., Per-Split) & Covert-AFA
         with B=10,000 patient-cluster 95% CIs.
     (B) Transitional-care referral exchange-rate profile across tau in {6%, 11.1%, 15%, 20%}.
     (C) Subgroup Delta NB (B=10,000 patient-cluster 95% CIs) across Race/Ethnicity and Sex.
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

    v1 = json.loads((out_dir / "two_phase_campaign_summary.json").read_text(encoding="utf-8"))
    v2 = json.loads((out_dir / "v2_campaign_summary.json").read_text(encoding="utf-8"))
    v21 = json.loads((out_dir / "paper1_v2_1_remediation_results.json").read_text(encoding="utf-8"))

    cohorts_v2 = v2["cohorts_v2"]
    can_h = v21["block3_canonical_holdouts_and_covert"]["canonical_holdouts"]
    stress_lf = v21["block2_stress_and_ablations"]["leak_free_train_fit_stress"]
    clin = v21["block4_clinical_diabetes_audit"]

    order = [
        ("UCI_Diabetes_Discharge", "UCI Diabetes (Discharge)", "Diabetes"),
        ("Home_Credit_7Table", "Home Credit (7-Table)", "HomeCredit"),
        ("IEEE_CIS_Fraud", "IEEE-CIS Auth Stream", "IEEE-CIS"),
        ("Scania_APS_Fleet", "Scania APS Fleet", "Scania"),
        ("NeurIPS_BAF", "NeurIPS BAF Stream", "BAF"),
        ("UCI_Taiwan_Credit", "UCI Taiwan Credit", "Taiwan"),
    ]
    short_x = [sx for _, _, sx in order]

    # -----------------------------------------------------------------------
    # FIGURE 1: Stage-A Pilot Regime Non-Inferiority (15% vs 25%) & Leak-Free Stress Pool
    # -----------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(14.4, 4.9), dpi=300)
    y_pos = np.arange(len(order))

    # Compute 15% and 25% deltas from v2_checkpoints
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
            # Nadeau-Bengio 90% CI (t_{0.95, 19} = 1.729, sqrt(1/20 + 0.15/0.55) = 0.568)
            se_nb = float(np.std(a, ddof=1)) * np.sqrt(1.0 / 20.0 + 0.15 / 0.55)
            ml.append(m_val)
            ll.append(m_val - 1.729 * se_nb)
            hl.append(m_val + 1.729 * se_nb)

    axes1[0].axvline(0.0, color="#222222", linewidth=1.0)
    axes1[0].axvline(
        -0.005, color="#888888", linestyle=":", linewidth=1.1, label="Strict NI (-0.005)"
    )
    axes1[0].axvline(
        -0.015, color="#d62728", linestyle="--", linewidth=1.2, label="Practical NI (-0.015)"
    )
    axes1[0].errorbar(
        d15_m,
        y_pos + 0.13,
        xerr=[np.array(d15_m) - np.array(d15_lo), np.array(d15_hi) - np.array(d15_m)],
        fmt="o",
        color="#1f77b4",
        capsize=3.5,
        linewidth=1.6,
        label="Neyman 15% Pilot (85% savings; 3/6 pass)",
    )
    axes1[0].errorbar(
        d25_m,
        y_pos - 0.13,
        xerr=[np.array(d25_m) - np.array(d25_lo), np.array(d25_hi) - np.array(d25_m)],
        fmt="s",
        color="#2ca02c",
        capsize=3.5,
        linewidth=1.6,
        label="Neyman 25% Pilot (75% savings; 5/6 pass)",
    )
    axes1[0].set_yticks(y_pos)
    axes1[0].set_yticklabels([lbl for _, lbl, _ in order], fontsize=9.2)
    axes1[0].invert_yaxis()
    axes1[0].set_xlim(-0.026, 0.013)
    axes1[0].set_ylim(7.05, -0.45)
    axes1[0].set_xlabel("Test AUPRC Difference vs. 100%-Cohort Seq-CMI (90% TOST CI)")
    axes1[0].set_title("A. Pilot-Fraction Regime & TOST Non-Inferiority", fontweight="bold")
    axes1[0].legend(fontsize=7.5, loc="lower left")
    axes1[0].grid(axis="x", alpha=0.3)

    # Panel B: Spurious channel selection across stress pools (leak-free train-fit verified!)
    x1b = np.arange(len(order))
    wb1 = 0.15
    v1_coh = {c["cohort_key"]: c for c in v1["cohorts"]}

    def _tot_spur_v1(ck: str, arm: str) -> float:
        st = v1_coh[ck]["stress_summary"][arm]
        return float(
            100.0
            * (
                st["null_fraction"]["mean"]
                + st["duplicate_twin_fraction"]["mean"]
                + st["echo_fraction"]["mean"]
            )
        )

    sp_prop = [
        stress_lf[ck]["proposed_ht_holm"]["null_pct"]
        + stress_lf[ck]["proposed_ht_holm"]["twin_pct"]
        + stress_lf[ck]["proposed_ht_holm"]["echo_pct"]
        for ck, _, _ in order
    ]
    sp_one = [_tot_spur_v1(ck, "oneshot_neyman_cmi") for ck, _, _ in order]
    sp_bperm = [_tot_spur_v1(ck, "cpi_pilot") for ck, _, _ in order]
    sp_ko = [_tot_spur_v1(ck, "crt_knockoff_pilot") for ck, _, _ in order]
    sp_unwt = [_tot_spur_v1(ck, "dime_cmi_pilot") for ck, _, _ in order]

    axes1[1].bar(
        x1b - 2 * wb1, sp_prop, wb1, label="Neyman Seq-CMI + Holm LCB (0.0%)", color="#2ca02c"
    )
    axes1[1].bar(x1b - wb1, sp_one, wb1, label="One-Shot CMI (Dup. Twins)", color="#ff7f0e")
    axes1[1].bar(x1b, sp_bperm, wb1, label="Baseline-Strat. Perm. (Dup. Twins)", color="#9467bd")
    axes1[1].bar(x1b + wb1, sp_ko, wb1, label="CRT / Knockoffs (Dup. Twins)", color="#1f77b4")
    axes1[1].bar(x1b + 2 * wb1, sp_unwt, wb1, label="Unweighted Pilot (No HT/LCB)", color="#d62728")
    axes1[1].set_xticks(x1b)
    axes1[1].set_xticklabels(short_x, fontsize=9.0)
    axes1[1].set_ylabel("Spurious Selected Channels (Null + Dup. Twin + Echo, %)")
    axes1[1].set_title("B. Leak-Free Stress-Pool Spurious Selection (M = 25-36)", fontweight="bold")
    axes1[1].legend(fontsize=7.3, loc="upper left")
    axes1[1].grid(axis="y", alpha=0.3)

    fig1.tight_layout()
    for ext in (".pdf", ".png"):
        fig1.savefig(
            out_dir / f"fig1_discovery_noninferiority_and_stress{ext}", bbox_inches="tight"
        )
    plt.close(fig1)

    # -----------------------------------------------------------------------
    # FIGURE 2: 2x3 Grid of All 6 Cohorts Streaming Frontiers + Covert-AFA
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 3, figsize=(14.6, 7.8), dpi=300)
    pol_plot = [
        ("streaming_voi", "Martingale VOI (Prop. 2)", "#2ca02c", "-o"),
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
            ax.plot(qs, nbs, psty, color=pcol, linewidth=1.55, markersize=4.0, label=plbl)
        qa = 100.0 * cd["voi_autonomous_centered"]["realized_query_rate"]["mean"]
        nba = cd["voi_autonomous_centered"]["net_benefit_per_1k"]["mean"]
        ax.scatter(
            [qa], [nba], color="#d62728", marker="*", s=135, zorder=7, label="Auto VOI > c_q"
        )

        # Overlay Covert-AFA canonical q=20% relative position mapped onto split baseline scale
        ch_d = can_h[ck]["net_benefit_and_query_rates"]
        s1_can = ch_d["stage1_0pct"]["nb"]
        s1_r20 = cd["streaming_curves_v2"]["streaming_voi"][0]["net_benefit_per_1k"]["mean"]
        afa20_shift = s1_r20 + (ch_d["covert_afa_20pct"]["nb"] - s1_can)
        q_afa20 = 100.0 * ch_d["covert_afa_20pct"]["q"]
        ax.scatter(
            [q_afa20],
            [afa20_shift],
            color="#8c564b",
            marker="X",
            s=75,
            zorder=6,
            label="Covert-AFA (20% CMI)",
        )

        ax.set_title(title_lbl, fontsize=10.2, fontweight="bold")
        if r_i == 1:
            ax.set_xlabel("Realized Streaming Query Rate (%)")
        if c_i == 0:
            ax.set_ylabel("Net Benefit per 1,000 (Net of Query Cost)")
        if idx == 0:
            ax.legend(fontsize=6.9, loc="lower right")
        ax.grid(alpha=0.3)

    fig2.tight_layout()
    for ext in (".pdf", ".png"):
        fig2.savefig(out_dir / f"fig2_streaming_budget_curves{ext}", bbox_inches="tight")
    plt.close(fig2)

    # -----------------------------------------------------------------------
    # FIGURE 3: Conformal FDR Selection + Temporal Shift & Local Burn-In Audit
    # -----------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 3, figsize=(15.4, 4.6), dpi=300)
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
    axes3[0].set_ylim(0, 118)
    axes3[0].set_ylabel("Realized False Discovery Proportion (%)")
    axes3[0].set_title("A. Exchangeable CfBH vs. Mondrian {1}", fontweight="bold", fontsize=9.8)
    axes3[0].legend(fontsize=6.8, ncol=2, loc="upper left")
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
    axes3[1].set_title("B. Certified Discovery Recall Gain", fontweight="bold", fontsize=9.8)
    axes3[1].legend(fontsize=7.2)
    axes3[1].grid(axis="y", alpha=0.3)

    # Panel C: Temporal Shift on CfBH (Era 0 -> Era 3 stale calibration vs n=2,500 local burn-in)
    shift_d = v21["block3_canonical_holdouts_and_covert"]["disjoint_chrono_and_cfbh_shift"][
        "cfbh_temporal_shift"
    ]
    sh_keys = ["IEEE_CIS_Fraud", "NeurIPS_BAF"]
    sh_lbls = ["IEEE-CIS\n(Era 0 -> Era 3)", "NeurIPS BAF\n(Era 0 -> Era 3)"]
    x_sh = np.arange(2)
    w_sh = 0.26
    stale_fdp = [100.0 * shift_d[k]["stale_cal_fdp"] for k in sh_keys]
    local_fdp = [100.0 * shift_d[k]["local_burnin_2500_fdp"] for k in sh_keys]
    targ_sh = [100.0 * shift_d[k]["target_fdr"] for k in sh_keys]

    axes3[2].bar(x_sh - w_sh / 2, stale_fdp, w_sh, label="Stale Era-0 Calibration", color="#d62728")
    axes3[2].bar(
        x_sh + w_sh / 2, local_fdp, w_sh, label="Recent Burn-In (n=2,500)", color="#2ca02c"
    )
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
    axes3[2].set_xticklabels(sh_lbls, fontsize=9.0)
    axes3[2].set_ylabel("Realized FDP Under Temporal Lag (%)")
    axes3[2].set_title("C. Temporal Drift & Burn-In Recalibration", fontweight="bold", fontsize=9.8)
    axes3[2].legend(fontsize=7.4)
    axes3[2].grid(axis="y", alpha=0.3)

    fig3.tight_layout()
    for ext in (".pdf", ".png"):
        fig3.savefig(out_dir / f"fig3_conformal_fdr_selection{ext}", bbox_inches="tight")
    plt.close(fig3)

    # -----------------------------------------------------------------------
    # FIGURE 4: Pilot Curves, Flip Decomposition, and Query-Fee Adaptation
    # -----------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, 3, figsize=(14.8, 4.5), dpi=300)
    fracs = [5, 10, 15, 25, 50, 100]
    ney_keys = ["neyman_05", "neyman_10", "neyman_15", "neyman_25", "neyman_50", "full_100"]
    srs_keys = ["srs_05", "srs_10", "srs_15", "srs_25", "srs_50", "full_100"]
    panel_a_specs = [
        ("Scania_APS_Fleet", "Scania (1.8%)", "#1f77b4"),
        ("IEEE_CIS_Fraud", "IEEE-CIS (3.7%)", "#2ca02c"),
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
    axes4[0].legend(fontsize=7.0, loc="lower right")
    axes4[0].grid(alpha=0.3)

    xb = np.arange(len(order))
    wb = 0.34
    voi_net_delta, rsk_net_delta = [], []
    for ck, _, _ in order:
        v20 = cohorts_v2[ck]["streaming_curves_v2"]["streaming_voi"][3]
        r20 = cohorts_v2[ck]["streaming_curves_v2"]["streaming_top_risk"][3]
        voi_net_delta.append(
            v20["beneficial_flip_nb_gain_per_1k"]["mean"]
            - v20["harmful_flip_nb_loss_per_1k"]["mean"]
            - v20["query_fee_nb_drag_per_1k"]["mean"]
        )
        rsk_net_delta.append(
            r20["beneficial_flip_nb_gain_per_1k"]["mean"]
            - r20["harmful_flip_nb_loss_per_1k"]["mean"]
            - r20["query_fee_nb_drag_per_1k"]["mean"]
        )
    axes4[1].bar(xb - wb / 2, voi_net_delta, wb, label="Martingale VOI (q=20%)", color="#2ca02c")
    axes4[1].bar(
        xb + wb / 2, rsk_net_delta, wb, label="Streaming Top-Risk (q=20%)", color="#1f77b4"
    )
    axes4[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes4[1].set_xticks(xb)
    axes4[1].set_xticklabels(short_x, fontsize=8.6)
    axes4[1].set_ylabel("Net Benefit Gain Over Stage 1 (/1,000)")
    axes4[1].set_title("B. Stage-B Net Benefit Lift at q=20%", fontweight="bold")
    axes4[1].legend(fontsize=7.8)
    axes4[1].grid(axis="y", alpha=0.3)

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
    axes4[2].set_title("C. Autonomous Stopping vs. Query Fee", fontweight="bold")
    axes4[2].legend(fontsize=7.6)
    axes4[2].grid(alpha=0.3)

    fig4.tight_layout()
    for ext in (".pdf", ".png"):
        fig4.savefig(out_dir / f"fig4_pilot_curve_and_flip_decomposition{ext}", bbox_inches="tight")
    plt.close(fig4)

    # -----------------------------------------------------------------------
    # FIGURE 5 (NEW): Clinical Case Study Deep Audit (Estimator Triad, Exchange Rate, Subgroups)
    # -----------------------------------------------------------------------
    fig5, axes5 = plt.subplots(1, 3, figsize=(15.2, 4.7), dpi=300)

    # Panel A: R=20 Estimator Triad & Covert-AFA paired Delta-NB vs Stage-1 / Reference
    triad = clin["r20_estimator_triad"]
    can_diab = can_h["UCI_Diabetes_Discharge"]["bootstraps_10k"]
    contrasts = [
        (
            "Primary Bagged VOI\nvs. Stage-1 (0%)",
            triad["bagged_voi_primary"]["gh_auto_vs_stage1"],
            "#2ca02c",
        ),
        (
            "Sym. Expected Policy\nvs. Stage-1 (0%)",
            triad["symmetric_expected_policy"]["gh_sym_vs_stage1"],
            "#1f77b4",
        ),
        (
            "Primary Bagged VOI\nvs. Uncentered VOI",
            triad["bagged_voi_primary"]["gh_auto_vs_uncentered"],
            "#17becf",
        ),
        (
            "Bagged VOI (20%)\nvs. Top-Risk (20%)",
            triad["bagged_voi_primary"]["gh_20_vs_toprisk_20"],
            "#9467bd",
        ),
        ("Martingale VOI\nvs. Covert-AFA", can_diab["gh_auto_vs_covertafa_auto"], "#ff7f0e"),
    ]
    y5a = np.arange(len(contrasts))
    m5a = [c[1]["delta_nb"] for c in contrasts]
    lo5a = [c[1]["delta_nb"] - c[1]["ci95_low"] for c in contrasts]
    hi5a = [c[1]["ci95_high"] - c[1]["delta_nb"] for c in contrasts]
    cols5a = [c[2] for c in contrasts]

    axes5[0].axvline(0.0, color="#222222", linewidth=1.0)
    axes5[0].axvline(0.50, color="#d62728", linestyle="--", linewidth=1.1, label="MCID (+0.50/1k)")
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
    axes5[0].set_yticklabels([c[0] for c in contrasts], fontsize=8.4)
    axes5[0].invert_yaxis()
    axes5[0].set_xlim(-0.8, 9.2)
    axes5[0].set_ylim(5.15, -0.45)
    axes5[0].set_xlabel("Paired Delta Net Benefit /1,000 (10k Patient-Cluster 95% CI)")
    axes5[0].set_title("A. UCI Diabetes R=20 Estimator Triad", fontweight="bold", fontsize=9.8)
    axes5[0].legend(fontsize=7.8, loc="lower right")
    axes5[0].grid(axis="x", alpha=0.3)

    # Panel B: Referral Alert-to-Case Exchange Rate across thresholds
    ex_rows = clin["clinical_exchange_rates"]
    x_ex = np.arange(len(ex_rows))
    tau_lbls = [f"{100.0 * r['tau']:.1f}%" for r in ex_rows]
    dtp = [r["delta_tp_per_1k"] for r in ex_rows]
    dtp_lo = [r["delta_tp_per_1k"] - r["delta_tp_ci95"][0] for r in ex_rows]
    dtp_hi = [r["delta_tp_ci95"][1] - r["delta_tp_per_1k"] for r in ex_rows]
    spared = [r["referrals_spared_per_1k"] / 10.0 for r in ex_rows]
    sp_lo = [(r["referrals_spared_per_1k"] - r["referrals_spared_ci95"][0]) / 10.0 for r in ex_rows]
    sp_hi = [(r["referrals_spared_ci95"][1] - r["referrals_spared_per_1k"]) / 10.0 for r in ex_rows]

    w_ex = 0.34
    axes5[1].bar(
        x_ex - w_ex / 2,
        dtp,
        w_ex,
        yerr=[dtp_lo, dtp_hi],
        capsize=3.0,
        color="#2ca02c",
        label="Extra Readmissions Caught (/1,000)",
    )
    axes5[1].bar(
        x_ex + w_ex / 2,
        spared,
        w_ex,
        yerr=[sp_lo, sp_hi],
        capsize=3.0,
        color="#1f77b4",
        label="False Referrals Spared (/100)",
    )
    axes5[1].axhline(0.0, color="#222222", linewidth=0.9)
    axes5[1].set_xticks(x_ex)
    axes5[1].set_xticklabels(tau_lbls, fontsize=9.0)
    axes5[1].set_ylim(-6.8, 15.8)
    axes5[1].set_xlabel("Discharge Action Threshold tau")
    axes5[1].set_ylabel("Cases Caught (/1k) | Referrals Spared (/100)")
    axes5[1].set_title("B. Transitional-Care Exchange Profile", fontweight="bold", fontsize=9.8)
    axes5[1].legend(fontsize=7.3, loc="upper left")
    axes5[1].grid(axis="y", alpha=0.3)

    # Panel C: Subgroup Delta-NB vs Stage-1 with 10k Patient-Cluster 95% CIs
    sg_all = (
        clin["subgroup_equity_and_calibration"]["race"]["strata"]
        + clin["subgroup_equity_and_calibration"]["gender"]["strata"]
    )
    sg_keep = [r for r in sg_all if r["n_enc"] >= 100]
    y5c = np.arange(len(sg_keep))
    m5c = [r["delta_nb_vs_s1"] for r in sg_keep]
    lo5c = [r["delta_nb_vs_s1"] - r["ci95_low"] for r in sg_keep]
    hi5c = [r["ci95_high"] - r["delta_nb_vs_s1"] for r in sg_keep]
    lbl5c = [f"{r['stratum']} (N={r['n_enc']:,})" for r in sg_keep]

    axes5[2].axvline(0.0, color="#222222", linewidth=1.0)
    axes5[2].errorbar(
        m5c,
        y5c,
        xerr=[lo5c, hi5c],
        fmt="s",
        color="#2ca02c",
        capsize=3.2,
        linewidth=1.6,
        markersize=5.2,
    )
    axes5[2].set_yticks(y5c)
    axes5[2].set_yticklabels(lbl5c, fontsize=8.2)
    axes5[2].invert_yaxis()
    axes5[2].set_xlabel("Subgroup Delta NB vs. Stage-1 (10k Cluster 95% CI)")
    axes5[2].set_title("C. Subgroup Equity & Cluster CIs", fontweight="bold", fontsize=9.8)
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
    print("Successfully generated and synced all 5 Paper 1 v2.1 figures (300 DPI)!")


if __name__ == "__main__":
    main()
