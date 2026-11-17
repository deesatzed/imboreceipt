# Critique of Original Freshman Explainer + Expanded Post-Audit Guide

> Original verbatim explainer preserved in [FRESHMAN_EXPLAINER_ORIGINAL.md](file:///Users/o2satz/antigravity/imboreceipt/docs/FRESHMAN_EXPLAINER_ORIGINAL.md). Full statistical audit in [PROFESSOR_AUDIT_AND_MITIGATION_REPORT.md](file:///Users/o2satz/antigravity/imboreceipt/docs/PROFESSOR_AUDIT_AND_MITIGATION_REPORT.md).

---

## Part 1: Critique of Our Last Display (Pedagogical + Scientific)

Before reusing or publishing our previous freshman explanation, a strict self-critique exposes **4 weaknesses** — two in *how we taught the concepts*, and two in *underlying benchmark artifacts* that our Senior Professor Audit caught and fixed:

1. **Scientific Overstatement #1 (`ULB Credit Card Fraud` `0.22` vs. `0.76` Claim):**
   * *What our previous display said:* Standard `100%`-row feature selection "got confused by normal customer drift and dropped AUPRC from `0.27` down to `0.22`," while `CREDO` jumped "`3x` higher to `0.76`."
   * *Why that was flawed:* When we audited `LightGBM` at `0.17%` fraud prevalence, we discovered standard `LightGBM` without Newton-step clipping (`max_delta_step=1.0`) suffers Hessian division blowup (`1 / 0.0017`), AND `1e-6` probability clipping created ties in the top fraud tail! Once we gave *every* baseline `max_delta_step=1.0` and `1e-15` clipping across 5 strictly non-overlapping future test windows, `100%`-row `Global MI` actually achieved **`0.743` AUPRC** (because ULB's `V1..V28` columns were *already mathematically orthogonalized by PCA*!), while `CREDO` achieved **`0.682–0.699` AUPRC** (`+0.20` above baseline `0.482`) by scanning **only `0.9%` of training rows (`111x` cheaper)** and saving **`85.3%` of runtime lookups**. That is a *cost-efficiency* win on PCA data, not a `3x` accuracy win over `Global MI` — whereas the genuine accuracy win over `Global MI` (`+35.1%`) happens on messy hospital data (`Diabetes 130-US`) where features are *not* pre-orthogonalized!
2. **Scientific Flaw #2 (`Taiwan Credit` Label Definition):**
   * *What our previous display missed:* The earlier `Taiwan Credit` loader defined "severe default" using month-1 utilization (`BILL_AMT1 / LIMIT_BAL > 0.55`) while also testing `BILL_AMT1` as an acquirable feature — a circular target definition, plus pre-split feature normalization. Fixing both (restricting strictly at month-6 baseline `PAY_6 <= 0` and predicting raw unmodified UCI default with train-only compilers) eliminated the artifact completely: `CREDO` (`0.450` AUPRC) now cleanly beats baseline (`0.247`) and slightly edges `Global MI` (`0.445`), while saving `56.7%` of Stage-2 queries.
3. **Pedagogical Gap #1 ("Why Can't One Big Tree Model Just Figure Everything Out?"):**
   * A sharp freshman who knows AUPRC will immediately ask: *"Once you buy those 2 extra columns, why use `CREDO`'s fancy Confirm-and-Rescue Lattice instead of throwing all columns into one big `LightGBM` or a 6-model ensemble?"* Our previous display never answered that! Now we have the exact controlled experiment (`A5b`, an equal-capacity 6-model ensemble): on messy hospital records (`Diabetes 130-US`), `CREDO`'s lattice beats the 6-model ensemble by **`+32.2%` (`0.196` vs. `0.149` AUPRC)** because separate confirmation and rescue branches stop noisy labels (`ADI = 0.519`) from drowning out distinct clinical pathways.
4. **Pedagogical Gap #2 (`kappa_0` Left as Mysterious Jargon):**
   * Mentioning "`kappa_0` coincidence check" in passing without a concrete everyday analogy ("two smoke alarms placed right above the same toaster") leaves a freshman guessing how `CREDO` knows when two signals are truly independent vs. echoing the same false alarm.

---

## Part 2: Expanded Freshman Guide (Post-Audit Hardened Edition)

### Why Rare-Event Machine Learning Hits a Wall — and How to Break Through It

Suppose your job is to spot **1-in-577 credit card frauds** or predict which **hospital patients will land back in the ER within 30 days**. Because you already know **AUPRC** (Area Under the Precision-Recall Curve), you know why rare events humble ordinary models: when `99%` of rows are negative, even a tiny `1%` false-alarm trickle buries your true positives and drags precision into the mud.

For two decades, the standard reflex in imbalanced machine learning has been: **"Synthesize fake minority rows (`SMOTE` / `ADASYN`) or reweight the loss function on the spreadsheet you already have."**

We built an open, leak-free testing harness ([imboreceipt](file:///Users/o2satz/antigravity/imboreceipt/README.md)) and ran **65 rigorous 4-way split experiments** (`train / calibrate / validate / test`, audited against data leakage, target circularity, and optimizer bugs) to test what *actually* moves AUPRC and operational utility.

---

### Lesson 1: You Can't Squeeze Blood from a Turnip (`0 / 8` Wins for Oversampling)
We took **8 classic imbalanced benchmarks** (`Mammography`, `Satimage`, `Thyroid`, `COIL 2000`, `Ozone`, `PC1`, `Yeast`, `Abalone`) and tested `SMOTE`, `Borderline-SMOTE`, `ADASYN`, cost-sensitive boosting, and Gaussian-copula synthetic data against a plain, properly calibrated `LightGBM` model.
* **What happened:** **0 out of 8** datasets showed a statistically significant AUPRC gain (overlap-corrected Nadeau–Bengio `p >= 0.088` on every single dataset).
* **Two hidden traps we uncovered:**
  1. Uncalibrated `SMOTE` warped predicted probabilities, making the model think `5%` risks were `40%` risks.
  2. Half of those famous benchmarks had only **`5` to `12` positive examples** in their test sets! With `8` positives, flipping *two* predictions swings recall by `25%`. Decades of tiny leaderboard "wins" on small imbalanced datasets sit entirely inside coin-flip noise.
* **The takeaway:** If a healthy borrower and a defaulting borrower have the *exact same numbers* in your current columns, drawing synthetic dots between them cannot separate them. You haven't hit a *modeling* problem; you've hit an **information ceiling**.

---

### Lesson 2: Fix "Receipts" Before You Blame Your Model
In textbook datasets, `Y = 0` means *"nothing bad happened."* In real hospitals, banks, and factories, `Y = 0` is just an **administrative receipt** meaning *"nobody recorded it inside our cutoff window."*
* In the **UCI 130-US Hospitals Diabetes cohort (`101,766` hospital visits across `71,518` patients)**, our automated **Receipt Audit** discovered severe label distortion (`ADI = 0.519`):
  * **`39.1%`** of patients labeled `"no <30-day readmission"` (`Y = 0`) were actually readmitted *after* day 30 (horizon censoring).
  * **`2.5%`** of `"negatives"` had **died in the hospital or entered hospice** — meaning they *could never* be readmitted, yet naive models treat them as "successful discharges"!
* **The takeaway:** Repairing censored and selective labels on the training set (`Stage R`) stops your model from learning backwards rules (like *"sicker hospice patients never come back, so extreme illness = low risk"*).

---

### Lesson 3: Buy Smart Data Using a Tiny "Detective Sample" (`CREDO REG`)
Once your current spreadsheet is maxed out, the only real way to lift AUPRC is to add **new channels** — extra clinical history, raw sensor physics, or 6-month payment dynamics.

In the real world, candidate data vendors often sell **redundant echoes** of data you already own (like a second credit bureau score that strongly correlates with default *only* because it copies the baseline limit you already have).
* **Why standard feature selection gets tricked:** Standard **Global Mutual Information (`Global MI`)** scans `100%` of your dataset and asks, *"Does channel `Z` correlate with `Y`?"* A redundant copy of your baseline data correlates with `Y`, so `Global MI` happily wastes your budget buying duplicate information!
* **How `CREDO REG` works:** Instead of measuring `100%` of rows, `CREDO` pulls a small **Case-Cohort Detective Sample** (all rare positives + `4x` as many look-alike baseline false alarms) and asks: *"Inside cases where my current model is confused, does `Z` add **new** log-odds separation beyond my baseline?"*
* **What happened across real datasets:**
  * **Hospital Readmissions (`101,766` visits):** `Global MI` scanned `100%` of visits, fell for a redundant intake-summary proxy (`50%` channel precision), and stalled at **`0.145` AUPRC**. `CREDO` scanned `55.9%` of visits, rejected every decoy (`100%` precision), and reached **`0.196` AUPRC (`+35.1%` relative gain over `Global MI`)**.
  * **Credit Card Fraud (`284,807` transactions, `0.17%` fraud):** Here, the public features (`V1..V28`) were *already* mathematically decorrelated by Principal Component Analysis (`PCA`), so `100%`-row `Global MI` was in its ideal textbook environment (`0.743` AUPRC). Even so, `CREDO` measured **only `0.9%` of training rows (`730` rows instead of `81,500` — `111x` cheaper!)** and still reached **`0.682–0.699` AUPRC** (up from `0.482` baseline), capturing **`>92%` of full-scan accuracy for `<1%` of the pilot data cost**.

---

### Lesson 4: The "Two Smoke Alarms Above One Toaster" Check (`kappa_0` Lattice)
Once you acquire new channels, how should you combine them?
* Imagine your baseline model is a **kitchen smoke alarm** (`A`) and your new feature is a **second sensor** (`B`):
  * **Independent sensor (`kappa_0 ≈ 1`):** If `B` is a **heat/carbon-monoxide sensor**, burnt toast sets off `A` *without* setting off `B`. Requiring `A AND B` slashes false alarms multiplicatively!
  * **Coupled sensor (`kappa_0 >> 1`):** If `B` is just a **second optical smoke alarm taped next to `A`**, burnt toast sets off *both* at the same time (`kappa_0 = 3.89` on factory milling machines because rotational `RPM` drives both baseline speed and mechanical power). Pretending they are independent double-counts false alarms!
* **`CREDO`'s LIGO Coincidence Gate:** Borrowed from gravitational-wave physics, `CREDO` shifts negative examples out of alignment on the calibration fold to measure `kappa_0`:
  * On **Hospital Readmissions (`101.8k`)**, separating confirmation and rescue pathways beat an **equal-size 6-model ensemble on the exact same features by `+32.2%` (`0.196` vs. `0.149` AUPRC)**!
  * On **Factory Milling Failures (`10,000` runs)**, `kappa_0 = 3.89` automatically detected shared `RPM` coupling, blended coupled + orthogonal branches to hit **`0.743` AUPRC** (beating pure-orthogonal's `0.696` and the 6-model ensemble's `0.725`), and caught **`96.5%` of overstrain failures** and **`92.5%` of power failures** while flagging **`0.0%` of memoryless random shocks**.

---

### Lesson 5: Only Pay for Stage-2 Lookups in the "Gray Zone" (`Conformal Foveation`)
Even after you build a great Stage-2 model, pulling extra sensors or bureau reports for *every* live customer costs money and latency.
* `CREDO` wraps Stage 1 in **Mondrian Conformal Prediction** (guaranteed statistical error bounds for both classes):
  * Clear `safe` (`{0}`) and clear `danger` (`{1}`) cases are decided immediately at Stage 1 for free.
  * Only ambiguous **`{0, 1}` gray-zone cases** trigger Stage-2 feature lookups ("foveation," like your eye focusing high-resolution vision only where something moved).
* **Runtime savings:** Across our 4 public cohorts, Conformal Foveation saved **`46.4%` (`Diabetes`), `56.7%` (`Taiwan Credit`), `85.3%` (`ULB Fraud`), and `97.7%` (`AI4I Maintenance`)** of Stage-2 runtime queries — and on `Taiwan Credit`, actually achieved **higher net profit (`1,566.8` per 1k) than paying for Stage-2 on everyone (`1,521.5` per 1k)**!

---

### Honest Bottom-Line Verdict on Usefulness (`8.5 / 10`)

* **Use `CREDO` when:** You operate a multi-stage screening pipeline (fraud, clinical triage, predictive maintenance, trust & safety, underwriting) where labels are censored/selective, candidate vendor features may echo data you already own, and pilot/runtime feature queries cost real money or latency.
* **Skip `CREDO` (just use calibrated `LightGBM` with `max_delta_step=1.0`) when:** You have a single closed CSV with clean labels, zero ability to engineer trajectories or acquire extra signals, or moderate-prevalence pre-cleaned PCA features where scanning `100%` of rows is already free.
