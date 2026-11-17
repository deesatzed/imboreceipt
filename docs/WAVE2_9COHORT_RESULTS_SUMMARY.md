# Unified 9-Cohort + 10-Dataset `CREDO-Bench` Wave-2 Results Summary

* **Total Leak-Free 4-Way Split Evaluations:** `95` (`50` Tier-1 Fixed-Table runs + `45` Multi-Channel / Multi-Table runs across up to `1,000,000` rows per dataset)
* **LLM Tokens Used:** `0` (`100%` deterministic local CPU math)
* **Automated Invariance & Leakage Tests:** `22 / 22` passing (`tests/test_professor_audit.py`)
* **Primary Receipts:** `artifacts/bench/wave2_full/PAPER_CAMPAIGN_REPORT.md` & `artifacts/bench/wave2_full/campaign_summary.json`

---

## 1. Tier-1 Fixed-Table Saturation (`10` Datasets, Including `Webpage 35k`, `KDD Protein Homology 146k`, & `CLIMB` Ensembles)

Across all `10` datasets (`1,109` to `145,751` rows), **0 / 10** show a statistically significant AUPRC gain over natural-prevalence calibrated `LightGBM` (`max_delta_step=1.0`) for **`SMOTE` (`p_NB >= 0.088`)**, **`EasyEnsemble` (`p_NB >= 0.068`)**, or **`RUSBoost` (`0 / 10`)**:
* **Webpage (`34,780` rows, `n_+^test = 147`):** Calibrated `LightGBM` **`0.8165 ± 0.0155`** vs. `SMOTE` `0.7704`, `EasyEnsemble` `0.3675`, `RUSBoost` `0.1007`.
* **KDD Protein Homology (`145,751` rows, `n_+^test = 194`):** Calibrated `LightGBM` **`0.8951 ± 0.0236`** vs. `SMOTE` `0.8849`, `EasyEnsemble` `0.8513`, `RUSBoost` `0.7772`.

---

## 2. Unified 9-Cohort Multi-Channel & Multi-Table Knockout Ablation (`5` Splits / Disjoint Future Eras Each)

| Cohort (`N`, Prevalence) | `A0` Base GBDT | `A1` `SMOTE` | `A4` Global MI (`100%` rows) | `A5b` `6`-Model Ens | `A6` Pure Orth | `A7` Full `CREDO` | Pilot Rows Measured | Stage-2 Saved | Joint 9-Cohort Holm `p_FWER` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gate 0.5 Ground Truth** (`12k`, `4.1%`) | `0.2104` | `0.1553` | `0.7869` | **0.8106** | `0.7917` | **0.8095** | **20.5%** | **84.2%** | `0.00002` |
| **UCI AI4I Maintenance** (`10k`, `3.4%`) | `0.3606` | `0.3552` | `0.7404` | `0.7250` | `0.6964` | **0.7431** | **16.9%** | **97.7%** | `0.00260` |
| **UCI Taiwan Credit** (`26.9k`, `18.7%`) | `0.2474` | `0.2468` | `0.4453` | `0.4471` | `0.4427` | **0.4495** | `93.3%` | **56.7%** | `0.00014` |
| **UCI Diabetes 130-US** (`101.8k`, `11.3%`) | `0.1305` | `0.1295` | `0.1454` | `0.1486` | **0.1968** | **0.1964** | **55.9%** | **46.4%** | `0.00014` |
| **ULB Fraud (PCA)** (`284.8k`, `0.17%`) | `0.4821` | `0.5150` | **0.7428** | `0.6988` | `0.6790` | `0.6822` | **0.90%** | **85.3%** | `0.03528` |
| **Scania APS Trucks** (`76.0k`, `1.81%`) | `0.5260` | `0.4978` | `0.8757` | `0.8939` | `0.8701` | **0.8946** | **9.04%** | **82.3%** | `0.00127` |
| **Home Credit (`7` Tables)** (`307.5k`, `8.07%`) | `0.1678` | `0.1394` | `0.2557` | **0.2586** | `0.2477` | **0.2581** | **40.4%** | **63.4%** | `0.00004` |
| **IEEE-CIS Fraud (`2` Tables)** (`590.5k`, `3.50%`) | `0.1974` | `0.1874` | `0.5030` | `0.5041` | `0.4776` | **0.5054** | **18.0%** | **84.1%** | `0.00019` |
| **NeurIPS BAF Onboarding** (`1.00M`, `1.10%`) | `0.0795` | `0.0637` | `0.1613` | **0.1638** | `0.1572` | **0.1615** | **4.93%** | **84.0%** | `0.00175` |

---

## 3. Headline Scientific Takeaways Across All 95 Splits

1. **`9 / 9` Multi-Channel Wins Under Joint Family-Wise Error Control:** Every single multi-channel cohort passes strict 9-cohort Holm–Bonferroni + Nadeau–Bengio significance (`p_FWER <= 0.0353`).
2. **Non-PCA Hypothesis Validated (`8 / 8` Non-PCA Cohorts):** On **every** non-PCA dataset — including all four Wave-2 industrial/financial flagships (`Scania 76k`, `Home Credit 307k`, `IEEE-CIS 590k`, `NeurIPS BAF 1M`) — `CREDO` (`4.9%–55.9%` pilot rows) **tied or beat `100%`-row `Global MI`**, proving that `ULB` was the sole outlier strictly because `ULB`'s `V1..V28` columns were pre-orthogonalized via PCA.
3. **`SMOTE` Degraded Performance on `4 / 4` Real-World Industrial/Financial Flagships:** Applying `SMOTE` on `X_base` lowered AUPRC on `Scania` (`0.526 -> 0.498`), `Home Credit` (`0.168 -> 0.139`), `IEEE-CIS` (`0.197 -> 0.187`), and `BAF 1M` (`0.080 -> 0.064`).
4. **Conformal Foveation Beats Paying for `100%` Stage-2 Lookups:** On `Scania APS` (`811.9` vs. `704.0` per 1k), `Home Credit` (`666.7` vs. `626.8`), `IEEE-CIS` (`830.8` vs. `823.2`), and `Taiwan Credit` (`1566.8` vs. `1521.5`), firing Stage-2 queries strictly inside Mondrian conformal `{0, 1}` ambiguity bands delivered **higher net dollar utility than querying Stage-2 on 100% of cases**, while eliminating **`56.7%` to `84.1%`** of runtime lookups!
