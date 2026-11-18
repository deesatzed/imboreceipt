# 1-Page Executive Summary: Solving Rare-Event Prediction by Fixing the Data Bottleneck (`CREDO`)

**Author:** Wayne A. Satz, MD (Department of Emergency Medicine, Temple University)  
**Scope:** 95 leak-free 4-way split evaluations across 10 standard benchmarks ($N = 1.1\text{k}\text{--}146\text{k}$) and 9 multi-table clinical, industrial, and financial cohorts ($N = 10\text{k}\text{--}1.0\text{M}$ rows).

---

### The Big Picture: Why Standard "Imbalanced Data" Tricks Hit a Wall
When predicting rare events ($\pi = 0.1\%\text{--}10\%$, such as 30-day hospital readmissions, equipment breakdowns, or payment fraud), data science teams almost always try to fix low Precision–Recall (`AUPRC`) by tweaking the **algorithm** on a fixed intake table—generating synthetic minority rows (`SMOTE`, `ADASYN`), training undersampled ensembles (`EasyEnsemble`, `RUSBoost`), or reweighting the loss function.

Across 50 splits on 10 standard benchmarks, we prove that once a modern gradient-boosted tree (`LightGBM`) is properly stabilized and calibrated on real-world prevalence, **every post-hoc resampling and ensemble method achieves 0 out of 10 statistically significant AUPRC wins ($p \ge 0.068$)**—and actively *lowers* accuracy on 4 out of 4 large-scale datasets (`76k` to `1M` rows). Models fail on rare events not because the algorithm is weak, but because **the baseline intake table runs out of information**.

---

### The Three Core Principles (With Plain-English Formulas)

#### Principle 1: You Can't Re-Sort a Tie — Why `SMOTE` & Reweighting Cannot Beat the Intake Table
* **The Plain-English Idea:** Suppose 100 cases look **identical** on your baseline intake table (same initial vitals, demographics, or summary stats), and 15 of them actually experience the rare event ($15\%$ risk). If your operational threshold requires **$50\%$ confidence** to justify an intervention, you have a $35$-point confidence gap.
* **Why `SMOTE` Fails:** Making synthetic copies of the 15 positive rows (`SMOTE`) or multiplying their weight by $5\times$ simply inflates **all 100 identical-looking rows together** from $15\%$ to $47\%$. Because multiplying everyone's odds by a constant **never changes who ranks above whom**, the 85 false alarms remain mixed right in with the 15 true cases.
* **The Formula (The Information Gap):**
  $$\text{Information Gap} = \text{Target Confidence Needed} - \text{Baseline Intake Confidence}$$
  *(Measured on a log-odds scale so evidence from separate data sources adds together cleanly.)*
* **Actionable Rule:** When $\text{Information Gap} > 0$, stop tweaking the algorithm on the intake table. The only way to separate the 15 true cases from the 85 look-alike false alarms is to pull a **second, independent data source ("Stage-2 data")**.

#### Principle 2: The "Look-Alike Stress Test" — Picking the Best Stage-2 Data Using Only `5%` of the Cohort
* **The Plain-English Problem:** Suppose you have 5 candidate Stage-2 data sources (e.g., secondary tables, longitudinal histories, or third-party vendor scores). Pulling all 5 sources on $1,000,000$ rows is too expensive. Worse, standard feature selection (`Global Mutual Information`) gets tricked by **"Copycat Channels"**—sources that simply repackage your intake variables in a new format and score high globally only because they repeat what the intake table already knows.
* **The Formula (`5\pi` Look-Alike Matching & Relative Evidence Gain `REG`):**
  $$\text{Pilot Sample Size} = 5 \times \text{Rare Event Rate } (\pi) \quad \text{(100\% of Positives } + 4\text{ "Look-Alike False Alarms" per Positive)}$$
  $$\text{Relative Evidence Gain (REG)} = \frac{\text{\% of True Positives flagged by the new Stage-2 data}}{\text{\% of Look-Alike False Alarms flagged by the new Stage-2 data}}$$
* **Why It Works:** Instead of comparing positive cases against obviously low-risk controls, we match every true positive against 4 false alarms that got the **exact same high risk score on the intake table**. Inside this matched group, a "Copycat Channel" flags both groups equally ($\text{REG} = 1.0 \implies \text{reject}$), whereas a genuinely independent data source flags True Positives far more often than Look-Alikes ($\text{REG} \gg 1.0 \implies \text{keep}$).
* **Result:** Measuring as little as **$0.9\%\text{--}18\%$ of rows**, `CREDO REG` beats equal-size unmatched sampling on **7 / 9 cohorts** ($p \le 0.030$) and beats $100\%$-dataset `Global MI` on **6 / 9 cohorts** ($p \le 0.048$).

#### Principle 3: Double-Checking the "Red-Flag" Bucket — Cutting Live Stage-2 Lookups by `41%–79%`
* **The Plain-English Problem:** Even after you find a great Stage-2 data source, pulling it on $100\%$ of live arrivals is costly. Standard two-stage screening ("Conformal Prediction") splits Stage-1 intake predictions into three buckets: **Green `{0}`** (clearly safe $\to$ skip Stage-2), **Yellow `{0, 1}`** (uncertain middle $\to$ order Stage-2), and **Red `{1}`** (top alerts $\to$ standard methods assume these are "already certain" and **skip Stage-2**).
* **Why Skipping the Red Bucket Fails in Rare Events (The Base-Rate Trap):**
  $$\text{True Precision Inside the Red Bucket} = \frac{\text{Caught Positives}}{\text{Caught Positives} + \text{False Alarms}}$$
  If a rare event occurs in **$1\%$** of arrivals ($10$ per $1,000$), even a strong Stage-1 screen with $90\%$ sensitivity ($9$ caught) and a $5\%$ false-alarm rate ($50$ false alarms) puts **$9$ true cases and $50$ false alarms** into the Red Bucket—meaning **$85\%$ of the "confident" Red Bucket are false alarms ($\frac{9}{9+50} = 15\%$ precision)!**
* **Our Fix ("Upper-Tail Foveation"):** Leaving those 50 false alarms unverified at the top of the list ruins precision. **CREDO Foveation** safely skips the huge Green Bucket (`60%–80%` of arrivals) and **routes Stage-2 lookups to verify the Red Bucket + top of the Yellow Bucket**.
* **Result:** At the **exact same Stage-2 lookup budget**, verifying the Red Bucket beats standard Yellow-Bucket conformal routing on **9 / 9 cohorts**, recovering **$95\%\text{--}99\%$ of full Stage-2 accuracy while eliminating $41\%\text{--}79\%$ of lookups on rare events**.

| Representative Cohort ($N$, Rare Event Rate) | Intake Only | `SMOTE` | Global MI (`100%`) | Unmatched (`5π`) | **Full `CREDO` (`5π`)** | Standard Yellow-Only | **`CREDO` Red+Yellow (Saved)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI Hospital Readmission** (`101.8k`, `11.2%`) | `0.1305` | `0.1295` | `0.2203` | `0.1979` | **0.2231** | `0.1981` | **0.2091** (`26.0%` saved) |
| **Scania APS Truck Fleet** (`76.0k`, `1.81%`) | `0.5260` | `0.4978` | `0.8483` | `0.8482` | **0.8937** | `0.5505` | **0.8869** (**78.6%** saved) |
| **Home Credit 7-Table** (`307.5k`, `8.07%`) | `0.1678` | `0.1394` | `0.2449` | `0.2449` | **0.2591** | `0.2040` | **0.2501** (**41.4%** saved) |
| **IEEE-CIS Fraud 2-Table** (`590.5k`, `3.50%`) | `0.1974` | `0.1874` | `0.4841` | `0.4798` | **0.5031** | `0.2294` | **0.4783** (**62.0%** saved) |
| **NeurIPS BAF Onboarding** (`1.00M`, `1.10%`) | `0.0795` | `0.0637` | `0.1101` | `0.1101` | **0.1660** | `0.0850` | **0.1612** (**61.9%** saved) |
