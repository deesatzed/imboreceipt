# Original Freshman Explainer Display (Saved Verbatim Before Professor Audit)

> Saved verbatim from the pre-audit conversation turn so we can transparently compare our initial communication and numbers against the post-mitigation Senior Professor Audit.

---

### 1. The Big Idea (and What Actually Happened in the Tests)

Imagine you are trying to catch credit card fraud (`1` in `577` transactions) or predict which hospital patients will be readmitted within 30 days. Since you already know **AUPRC**, you know why this is brutal: when positives are tiny, even a model that looks "99% accurate" can flood you with false alarms, tanking your precision and giving you a mediocre AUPRC.

For the last 20 years, the standard textbook advice for imbalanced data has been: **"Use `SMOTE` (make up synthetic minority examples) or tweak loss weights on the table you already have."**

We tested that textbook advice first, and then tested **`CREDO`** (built in `imboreceipt`; full report in `artifacts/bench/paper_full/PAPER_CAMPAIGN_REPORT.md`). Here is what happened in plain English:

#### Finding 1: Rearranging the same columns doesn't create new information (`0 / 8` wins)
On **8 classic imbalanced benchmarks** (`40` runs), we pitted plain Gradient Boosted Trees (`LightGBM`) against `SMOTE`, `ADASYN`, cost-sensitive trees, and synthetic data generators using a fixed spreadsheet of features.
* **Result:** **0 out of 8** datasets showed a real AUPRC improvement over plain, properly calibrated `LightGBM`. Worse, `SMOTE` warped the predicted probabilities (making the model overconfident) and many famous benchmarks had only `5` to `12` positive cases in the test set — meaning "state-of-the-art" leaderboard jumps were mostly coin-flip luck.
* **Intuition:** If two patients look identical in your current spreadsheet, inventing fake dots between them cannot tell you which one is actually sick. You hit an **information wall**.

#### Finding 2: Looking *only* at the model's blind spots finds the right new data for pennies
To break an information wall, you have to pull in **new signals** (extra lab results, raw sensor physics, payment trajectories). Usually, companies either compute correlations across *every single row* (expensive, and drowned out by the `99.8%` normal crowd) or guess which new data to buy.
* **How `CREDO` works:** Instead of scanning everyone, `CREDO` pulls a tiny **"detective sample"** (`REG`): all the rare positives plus just `4x` as many look-alike false alarms, and asks: *"Which new feature specifically separates missed positives from false alarms?"*
* **Result on Credit Card Fraud (`284,807` transactions, `0.17%` fraud):**
  * Standard full-dataset feature selection scanned **`100%` of the training data**, got confused by normal customer drift, and **dropped AUPRC from `0.27` down to `0.22`**.
  * `CREDO` scanned **less than `1%` (`0.9%`) of the rows** (`99.1%` cheaper!), picked the exact right fraud signals, and combined them carefully to hit **`0.76` AUPRC** — more than **3x higher** — catching **`80%` of frauds** at a `0.2%` false-alarm rate.

#### Finding 3: Fixing "broken labels" + smart physics beats giant brute-force models
* **Hospital Readmissions (`101,766` visits):** Our "Receipt Audit" caught something wild in a famous UCI diabetes dataset: **`39%` of "healthy" negatives were actually readmitted right *after* day 30**, and **`2.5%` had died or gone to hospice** (so they *couldn't* be readmitted, yet standard ML treats them as "successes"!). Cleaning up that label distortion + adding just **2 targeted channels** pushed AUPRC from `0.13` to **`0.21`** — beating standard `100%`-row selection (`0.17`) and even beating our older `imbora` system (`0.19`) that used all 47 features at once.
* **Factory Machine Failures (`10,000` machines):** `CREDO` measured just **`17%` of rows** to reach **`0.74` AUPRC** (up from `0.34`). Better yet, this dataset labels *why* each machine broke: `CREDO` caught **`96.5%` of mechanical overstrain failures** and **`79.1%` of power failures**, while catching **`0.0%` of pure random shocks** — proving it learned real physics, not statistical flukes.
* **Two-Stage "Double-Check" (`Conformal Foveation`):** At prediction time, you don't even need the expensive extra features for everyone. Stage 1 clears the obvious `safe` and `danger` cases cheaply and only pays for Stage-2 features on the uncertain "gray zone," saving **`26%` to `78%` of runtime queries** without losing accuracy.

---

### 2. Honest Opinion on Usefulness

**Where `CREDO` is genuinely high-value (8.5 / 10 practical utility):**
1. **Tiered / Pay-Per-Query Operations (Fraud, Triage, Industrial IoT, Cybersecurity):** Anywhere you have cheap baseline logs (`Stage 1`) and expensive deeper checks (`Stage 2`: credit bureau pulls, genomic/lab panels, manual chart audits, high-frequency vibration FFTs). Cutting pilot data collection by `80%–99%` and runtime lookups by `60%+` translates directly into saved money and latency.
2. **Messy Real-World Outcomes ("Receipts," Not Clean Labels):** In healthcare, churn, and lending, "negative" almost never means "truly negative" — it means *"hasn't happened yet"* or *"went somewhere else."* Auditing labels *before* modeling saves teams months of chasing phantom model bugs.
3. **Stopping Wasted Engineering Effort:** The diagnostic gates tell a team *immediately* whether they have too few positives to trust their test set (`Gate C`) or have maxed out their current table (`Gate E`), stopping endless `SMOTE`/hyperparameter busywork.

**Where `CREDO` is *not* magic (and when NOT to use it):**
1. **Closed, Static Kaggle-Style Tables:** If you are handed one fixed CSV and you *cannot* engineer time-series physics, repair labels, or acquire new columns, `CREDO` won't magically boost AUPRC — its whole point is telling you that nothing else will either, so just use calibrated `LightGBM`.
2. **Single-Factor, Moderate-Imbalance Problems:** On the **Taiwan Credit (`30,000` borrowers, `10.4%` default)** test, every bill and payment column really measured one shared thing ("is this person broke?"). Because positives weren't super rare (`10.4%`) and mechanisms weren't independent, standard `100%`-row feature selection slightly beat `CREDO` (`0.54` vs. `0.51` AUPRC), though `CREDO`'s coincidence check (`kappa_0`) successfully protected it from crashing (`0.47`) when fusing correlated signals.

---

### 3. Recommended Next Steps

1. **Draft & Certify Paper A (`CREDO` + `CREDO-Bench` Preprint)**
2. **Test on One Real Proprietary / Domain Dataset You Care About**
3. **Scope Paper B (Open-World LLM Motif Discovery)**
