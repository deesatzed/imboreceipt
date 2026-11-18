# Unified 9-Cohort + 10-Dataset `CREDO-Bench` Journal-Ready Results Summary

* **Total Leak-Free 4-Way Split Evaluations:** `95` (`50` Tier-1 Fixed-Table runs + `45` Multi-Channel / Multi-Table runs across up to `1,000,000` rows per dataset)
* **LLM Tokens Used:** `0` (`100%` deterministic local CPU math)
* **Automated Invariance, Leakage, & Journal-Editor Audit Tests:** `24 / 24` passing (`tests/test_professor_audit.py`)
* **Primary Receipts:** `artifacts/bench/journal_ready/PAPER_CAMPAIGN_REPORT.md` & `artifacts/bench/journal_ready/campaign_summary.json`

---

## 1. Theorem 1 — Fixed-Table Decoder Saturation (`10` Datasets, Including `Webpage 35k`, `KDD Protein Homology 146k`, & `CLIMB` Ensembles)

Across all `10` canonical benchmarks (`1,109` to `145,751` rows), **0 / 10** show a statistically significant AUPRC gain over natural-prevalence calibrated `LightGBM` (`max_delta_step=1.0`) for **`SMOTE+GBDT` ($p_{\text{NB}} \in [0.088, 0.961]$)**, **`EasyEnsemble` ($p_{\text{NB}} \in [0.068, 1.000]$)**, or **`RUSBoost` (`0 / 10`)**:
* **Webpage (`34,780` rows, `n_+^test = 147`):** Calibrated `LightGBM` **`0.8165 ± 0.0155`** vs. `SMOTE` `0.7704`, `EasyEnsemble` `0.3675`, `RUSBoost` `0.1007`.
* **KDD Protein Homology (`145,751` rows, `n_+^test = 194`):** Calibrated `LightGBM` **`0.8951 ± 0.0236`** vs. `SMOTE` `0.8849`, `EasyEnsemble` `0.8513`, `RUSBoost` `0.7772`.

---

## 2. Theorem 2 — `5π` Score-Matched Case-Cohort `CREDO REG` vs. `100%`-Row `Global MI` & `5π` `Unmatched Case-Control MI` (`Prentice 1986`)

| Cohort (`N`, Prevalence) | `A0` Base GBDT | `A1` `SMOTE` | `A4` Global MI (`100%` rows) | `A4b` Unmatched CC MI (`5π`) | `A5` `CREDO REG` (`5π`) | `A5b` `6`-Ens | `A7` Full `CREDO` | Pilot Rows Measured | NB $p$ (`REG` vs `MI` / `UnmCC`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gate 0.5 Ground Truth** (`12k`, `4.1%`) | `0.2104` | `0.1553` | `0.6469` | `0.5188` | **0.7869** | `0.8106` | **0.8143** | **20.5%** | `0.017 / 0.002` |
| **UCI AI4I Maintenance** (`10k`, `3.4%`) | `0.3606` | `0.3552` | `0.5125` | `0.5125` | **0.7433** | `0.7250` | `0.7230` | **16.9%** | `0.000 / 0.000` |
| **UCI Taiwan Credit** (`26.9k`, `18.7%`) | `0.2474` | `0.2468` | `0.4453` | `0.4453` | `0.4445` | `0.4471` | **0.4504** | `93.3%` | `0.725 / 0.725` |
| **UCI Diabetes 130-US** (`101.8k`, `11.3%`) | `0.1305` | `0.1295` | `0.2203` | `0.1979` | **0.2203** | `0.2226` | **0.2231** | **55.9%** | `1.000 / 0.030` |
| **ULB Fraud (PCA)** (`284.8k`, `0.17%`) | `0.4821` | `0.5150` | `0.7348` | `0.7288` | `0.6977` | **0.7380** | **0.7374** | **0.90%** | `0.845 / 0.843` |
| **Scania APS Trucks** (`76.0k`, `1.81%`) | `0.5260` | `0.4978` | `0.8483` | `0.8482` | **0.8930** | **0.8939** | **0.8937** | **9.04%** | `0.001 / 0.001` |
| **Home Credit (`7` Tables)** (`307.5k`, `8.07%`) | `0.1678` | `0.1394` | `0.2449` | `0.2449` | **0.2557** | `0.2586` | **0.2591** | **40.4%** | `0.003 / 0.003` |
| **IEEE-CIS Fraud (`2` Tables)** (`590.5k`, `3.50%`) | `0.1974` | `0.1874` | `0.4841` | `0.4798` | **0.5015** | `0.5021` | **0.5031** | **18.0%** | `0.048 / 0.012` |
| **NeurIPS BAF Onboarding** (`1.00M`, `1.10%`) | `0.0795` | `0.0637` | `0.1101` | `0.1101` | **0.1574** | `0.1638` | **0.1660** | **4.93%** | `0.010 / 0.010` |

---

## 3. Theorem 3 — Rare-Event Conformal Singleton Paradox (`CREDO Deficit Foveation` vs. `Classical Mid-Band Conformal {0,1}`)

When $\pi \ll q_{\text{target}}$, Stage-1's Mondrian `{1}` singleton (`only_1`) still has a high false-positive rate ($D_q(x) > 0$). Classical mid-band conformal abstention (`{0, 1}`, skipping `{1}`) leaves those top-scoring false positives unverified at Stage 2, whereas **CREDO Evidence-Deficit Upper-Tail Foveation (`in_1 & D_q(x) > 0`)** verifies them while pruning `~in_1` at the **exact same Stage-2 query budget (`k_queries`)**:

| Cohort | Stage-2 Queries Saved | Stage-1 Only AUPRC | Classical Mid-Band Conformal `{0,1}` | **CREDO Deficit Foveation (`in_1 & D_q>0`)** | 100% Stage-2 AUPRC | Foveation vs. Mid-Band Holm $p_{\text{FWER}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gate 0.5 Ground Truth** | **62.9%** | `0.2104` | `0.4623` | **0.6785** (`+0.2162`) | `0.8143` | `0.00249` |
| **UCI AI4I Maintenance** | **77.1%** | `0.3606` | `0.4508` | **0.6335** (`+0.1827`) | `0.7230` | `0.07098` |
| **UCI Taiwan Credit** | **34.6%** | `0.2474` | `0.3964` | **0.4200** (`+0.0236`) | `0.4504` | `0.16500` |
| **UCI Diabetes 130-US** | **26.0%** | `0.1305` | `0.1981` | **0.2091** (`+0.0110`) | `0.2231` | `0.16500` |
| **ULB Credit Card Fraud** | **79.0%** | `0.4821` | `0.4818` | **0.7129** (`+0.2311`) | `0.7374` | `0.06416` |
| **Scania APS Trucks** | **78.6%** | `0.5260` | `0.5505` | **0.8869** (`+0.3364`) | `0.8937` | `0.00241` |
| **Home Credit (`7` Tables)** | **41.4%** | `0.1678` | `0.2040` | **0.2501** (`+0.0461`) | `0.2591` | `0.00227` |
| **IEEE-CIS Fraud (`2` Tables)** | **62.0%** | `0.1974` | `0.2294` | **0.4783** (`+0.2489`) | `0.5031` | `0.00014` |
| **NeurIPS BAF Onboarding** | **61.9%** | `0.0795` | `0.0850` | **0.1612** (`+0.0762`) | `0.1660` | `0.00241` |
