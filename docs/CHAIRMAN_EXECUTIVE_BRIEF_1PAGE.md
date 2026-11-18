# Executive Brief: Reframing Rare-Event Prediction from Algorithmic Tweaking to Active Evidence Acquisition (`CREDO`)

**Target Audience:** Department Leadership & Data Science Faculty  
**Empirical Scope:** 95 leak-free 4-way split evaluations across 10 standard benchmarks ($N = 1.1\text{k}\text{--}146\text{k}$) and 9 multi-table clinical, industrial, and financial cohorts ($N = 10\text{k}\text{--}1.0\text{M}$, including $101,766$ hospital encounters).

---

### The Core Premise: Why Standard Imbalanced Learning Hits a Wall
When predicting rare events ($\pi = 0.1\%\text{--}10\%$, such as 30-day unplanned readmissions, occult ED decompensation, equipment failure, or fraud), standard data science treats low Precision–Recall (`AUPRC`) as an **algorithmic problem** on a fixed intake table $X_{\text{base}}$—applying synthetic oversampling (`SMOTE`, `ADASYN`), undersampled ensembles (`EasyEnsemble`, `RUSBoost`), or cost reweighting.

Across 50 splits on 10 benchmarks, we show that once gradient-boosted trees (`LightGBM`) are Newton-stabilized against small-probability step blowup (`max_delta_step=1.0`) and calibrated on natural prevalence, **every post-hoc resampling and ensemble method achieves 0 / 10 statistically significant AUPRC wins ($p_{\text{NB}} \ge 0.068$)** and actively degrades AUPRC on 4 / 4 large-scale flagships. Rare-event screening stalls not because the classifier is weak, but because the baseline intake table $X_{\text{base}}$ runs out of information.

---

### Three Statistical Principles & Plain-English Formulas

#### 1. The Fixed-Table Information Ceiling & Evidence Deficit (Theorem 1)
Reweighting or oversampling a fixed feature table from prevalence $\pi$ to $\tilde{\pi}$ only multiplies every case's posterior odds by a constant factor $\gamma = \frac{\tilde{\pi}(1-\pi)}{\pi(1-\tilde{\pi})}$. Because multiplying everyone's odds by a constant never changes *who ranks above whom*, it cannot improve the Precision–Recall curve. Instead, for any patient or arrival $x$, we compute the **Log-Evidence Deficit** $D_q(x)$:

$$D_q(x) = \max\left(0,\; \underbrace{\ln\frac{q_{\text{target}}}{1 - q_{\text{target}}}}_{\text{Target Log-Odds Needed}} - \underbrace{\ln\frac{p_{\text{base}}(x)}{1 - p_{\text{base}}(x)}}_{\text{Baseline Intake Log-Odds}}\right)$$

* **Operational Meaning:** $D_q(x)$ measures **how much additional diagnostic evidence is missing**. If an intervention requires $q_{\text{target}} = 50\%$ precision (even odds, $\ln 1 = 0$), but a patient's baseline intake variables $X_{\text{base}}$ only reach $p_{\text{base}}(x) = 15\%$ ($\ln(0.15/0.85) = -1.73$), then $D_q(x) = 1.73\text{ nats} > 0$. No algorithm on $X_{\text{base}}$ can close that gap; crossing the threshold requires acquiring an orthogonal **Stage-2 channel** $Z$ (e.g., a laboratory panel, utilization trajectory, or secondary table).

#### 2. Discovering Stage-2 Channels Without Buying "Baseline Echoes" (Theorem 2)
Measuring candidate Stage-2 channels $Z_1, \dots, Z_M$ across $100\%$ of a large cohort is expensive. Worse, standard feature selection (Global Mutual Information) gets fooled by **collinear echoes**—candidate feeds that repackage baseline intake variables ($Z \approx f(X_{\text{base}})$) and appear predictive globally only because $X_{\text{base}}$ predicts $Y$. `CREDO` solves both problems using a **$5\pi$ Score-Matched Case-Cohort Pilot**:

$$\text{Pilot Size} = 5\pi \quad (\text{100\% of positives } + 4\text{ controls matched on baseline risk } p_{\text{base}}(x))$$

$$\text{REG}(Z \mid X_{\text{base}}) = \frac{\mathbb{P}(Z \text{ flags high risk} \mid Y=1,\; \text{Matched High } p_{\text{base}})}{\mathbb{P}(Z \text{ flags high risk} \mid Y=0,\; \text{Matched High } p_{\text{base}})}$$

* **Operational Meaning:** Instead of comparing cases against obviously healthy controls, we match every true positive against 4 false-alarm controls that **looked identical on the baseline intake model**. Inside this matched group, a redundant echo of $X_{\text{base}}$ has $\text{REG} = 1.0$ (no new lift), whereas a genuinely independent diagnostic channel has $\text{REG} \gg 1.0$.
* **Result:** Measuring as little as **$0.9\%\text{--}18\%$ of rows**, `CREDO REG` beats equal-size unmatched case-control MI on **7 / 9 cohorts** ($p_{\text{NB}} \le 0.030$, including $+0.0224$ AUPRC on $101,766$ hospital readmissions) and beats $100\%$-row Global MI on **6 / 9 cohorts** ($p_{\text{NB}} \le 0.048$).

#### 3. The Rare-Event Conformal Paradox & Runtime "Foveation" (Theorem 3)
At runtime, ordering Stage-2 tests on $100\%$ of arrivals is too costly. Classical two-stage conformal prediction splits Stage-1 predictions into `{0}` (safe), `{0, 1}` (uncertain mid-band), and `{1}` (flagged positive)—and only orders Stage-2 tests on `{0, 1}`, assuming `{1}` needs no verification. **In rare events, that assumption fails:**

$$\text{Precision Inside Stage-1 } \{1\} = \frac{\pi \cdot \text{TPR}_1}{\pi \cdot \text{TPR}_1 + (1 - \pi) \cdot \text{FPR}_1}$$

* **Operational Meaning:** When prevalence is rare ($\pi = 1\%$), even a strong Stage-1 screen with $\text{TPR}_1 = 90\%$ sensitivity and $\text{FPR}_1 = 5\%$ false positives yields a `{1}` bucket precision of only $\frac{0.009}{0.009 + 0.0495} = \mathbf{15.4\%}$—meaning **$84.6\%$ of the "confident" `{1}` bucket are false alarms!** Leaving `{1}` unverified ruins top-tail precision. **CREDO Evidence-Deficit Foveation** instead routes Stage-2 workups to the upper tail ($\{1\}$ and $\{0,1\}$ cases where $D_q(x) > 0$) while pruning `{0}`.
* **Result:** At the exact same Stage-2 query budget, `CREDO Foveation` beats classical `{0, 1}` conformal routing on **9 / 9 cohorts** (e.g., `0.8869` vs. `0.5505` AUPRC on `Scania 76k`; `0.4783` vs. `0.2294` on `IEEE-CIS 590k`; `0.7129` vs. `0.4818` on `ULB 285k`), recovering **$95\%\text{--}99\%$ of full Stage-2 AUPRC while eliminating $41\%\text{--}79\%$ of rare-event lookups**.

---

### Immediate Clinical Translation: Emergency Department `ESI-3` Triage
Roughly half of Emergency Department arrivals are triaged as **ESI Level 3**—a classic high-volume ambiguity zone where clinicians must decide who needs expensive/time-intensive Stage-2 diagnostic workups (lactate, troponin, imaging) to catch rare occult decompensation (`sepsis`, `ICU transfer`) without gridlocking the ED. `CREDO`'s $5\pi$ score-matched discovery and Evidence-Deficit Foveation provide the exact mathematical engine to optimize selective ED workup ordering.
