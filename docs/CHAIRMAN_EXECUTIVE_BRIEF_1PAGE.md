# 1-Page Executive Walkthrough: How `CREDO` Works (A Step-by-Step Thought Experiment on 101,766 Hospital Readmissions)

**Author:** Wayne A. Satz, MD (Department of Emergency Medicine, Temple University)  
**Cohort:** UCI 30-Day Hospital Readmission ($101,766$ hospital stays across $71,518$ diabetic patients, where **$11.2\%$** suffer an unplanned readmission within 30 days).

---

### Three Predicates of Our Thought Experiment
Instead of treating imbalanced data as an **algorithmic** problem (`SMOTE`, reweighting), we frame it around three information predicates:
1. **Define the Latent-Space Information Gap:** Measure the exact gap between our current feature space and the ideal confidence needed to act.
2. **Exploit "High-Signal False Positives":** Aside from the rare True Positives themselves, where is the richest information in the current data about what we are missing? In the **High-Signal False Positives**—the look-alike false alarms that max out the intake score but do not have the event.
3. **Two-Stage Screening to Rescue Signal "Lost in True-Negative Abundance":** When secondary features are evaluated globally across the whole dataset, they get drowned out by the overwhelming sea of easy True Negatives (`89%–99%` of rows). A 2-stage screen first sweeps away the easy True Negatives, then applies secondary features **specifically inside the upper-tail screening set** where True Positives and High-Signal False Positives collide.

---

### The Setup: Why Isn't Everything in the Stage-1 Intake Screen from Day 1?
* **What if you DO put 100% of available columns into Stage 1?** We tested that first across **10 full-feature benchmarks (50 splits)** in Table 1 of the paper: even with 100% of columns included upfront, once a table's information is exhausted, `SMOTE`, `ADASYN`, `EasyEnsemble`, and `RUSBoost` achieve **0 out of 10 statistically significant AUPRC wins** over calibrated `LightGBM`.
* **Why Stage 1 vs. Stage 2 exists in real life:** In hospitals, banks, and fleets, **Stage 1** is what is free and instant at arrival (7 basic registration fields, a 1-page loan form, or a card-swipe header), whereas **Stage-2 channels** live in separate EHR modules, lab orders, or paid external Credit/Identity APIs (`Home Credit`'s 7 tables; `IEEE-CIS`'s identity table). Using only the **7 free Stage-1 registration fields** on UCI Readmission (`Age`, `Gender`, `Race`, `Admission Type/Source`, `Length of Stay`, `Lab Count`), calibrated `LightGBM` gets just **`0.1305` AUPRC** (barely above random `0.112`) because thousands of patients look **identical** on those 7 numbers.

---

### Step 1: Defining the Information Gap (*Principle 1: You Can't Re-Sort a Tie*)
* **Picture 100 patients** who share the **exact same 7 intake numbers** (e.g., 65-year-old female, 4-day stay, 45 labs). In reality, **15 will be readmitted ($15\%$ true risk)** and **85 will recover fine ($85\%$ High-Signal False Positives)**. Suppose our care team requires at least **$28\%$ confidence (precision)** before paying for a home-health nurse:

$$\text{Information Gap} = \text{Target Confidence Needed } (28\%) - \text{Current Space Confidence } (15\%) = +13\% \text{ gap}$$

* **Why `SMOTE` Hits a Brick Wall:** Generating fake copies of the 15 readmitted patients (`SMOTE`) or multiplying their weight by $3\times$ simply inflates **all 100 identical-looking patients together** from $15\%$ to $35\%$. It **never changes who ranks above whom**—the 85 High-Signal False Positives stay mixed right in with the 15 True Positives (`SMOTE` gets **`0.1295` AUPRC**).
* **Rule 1:** When $\text{Information Gap} > 0$, stop tweaking the algorithm on the intake space; you must bring in **Stage-2 features** to break the tie.

---

### Step 2: Using "High-Signal False Positives" to Pick Stage-2 Features (*Principle 2: Rescuing Signal Lost in True-Negative Abundance*)
* **The Choice:** The hospital EHR has **7 candidate Stage-2 modules** (*1: Prior Inpatient/ER Visits*, *2: Discharge Disposition & Specialty*, *3: Diagnosis Count*, *4: Medication Changes*, *5: HbA1c/Glucose*, *6: A "Copycat Acuity Index" that secretly repackages Length-of-Stay + Lab Count*, and *7: Ward Noise*). Our budget allows **2 modules**.
* **Why Global Selection Fails (Drowned in True-Negative Abundance):** If you score features globally or against random healthy patients (Unmatched Case-Control MI), the 89% sea of easy True Negatives dominates the math—making a "Copycat" or general-illness feature look great because readmitted patients stayed longer than random healthy patients! Unmatched MI picks a suboptimal channel on 4 of 5 splits and stalls at **`0.1979` AUPRC**.
* **What `CREDO` Does Instead:** We pit every True Positive directly against **$m(\pi)$ "High-Signal False Positives"** ($m=2$ at $11.2\%$ prevalence, measuring only $33.5\%$ of charts; $m=4\text{--}8$ for rarer events)—patients who got the **exact same high risk score on the 7 intake fields**, but were *not* readmitted:

$$\text{Relative Evidence Gain (REG)} = \frac{\text{\% of True Positives flagged by the Stage-2 feature}}{\text{\% of High-Signal False Positives flagged by the Stage-2 feature}}$$

* **Result:** Inside this matched upper-tail set, the "Copycat Index" flags both groups equally ($\text{REG} = \frac{30\%}{30\%} = \mathbf{1.0} \to \text{Rejected}$), while *Prior Visits* + *Discharge Routing* rescue the signal that was lost in True-Negative abundance ($\text{REG} \gg \mathbf{1.0} \to \text{Selected on 5/5 splits}$), lifting AUPRC from **`0.1305` $\to$ `0.2231`** ($p = 0.030$).

---

### Step 3: Two-Stage Screening at Discharge (*Principle 3: Verifying the Upper-Tail Screening Set*)
* **The Live Triage Problem:** Standard two-stage "Conformal Prediction" splits Stage-1 predictions into **Green `{0}`** (easy True Negatives $\to$ skip Stage-2), **Yellow `{0, 1}`** (uncertain middle $\to$ pull Stage-2), and **Red `{1}`** (top alerts $\to$ standard methods assume these are "already certain" and **skip Stage-2**).
* **Why Skipping the Red Bucket Fails:** Look inside that Stage-1 **Red Bucket `{1}`**:

$$\text{True Precision Inside the Red Bucket} = \frac{\text{True Positives}}{\text{True Positives} + \text{High-Signal False Positives}}$$

* Because readmission is $11.2\%$ overall and the 7 intake fields are weak, **roughly $80\%$ of the Stage-1 Red Bucket are High-Signal False Positives**! Standard Yellow-Only routing leaves those unverified false alarms at the top of the list (**`0.1981` AUPRC**). **`CREDO` Two-Stage Foveation** sweeps away the Green Bucket of easy True Negatives (`26.0%` saved on Diabetes; **`41%–79%` saved** on rarer cohorts) and **applies Stage-2 features directly to the Red + Yellow upper-tail screening set**, lifting AUPRC to **`0.2091`** at the exact same query budget (winning on **9 / 9 cohorts**, including `0.55 -> 0.89` on `76k` trucks and `0.23 -> 0.48` on `590k` transactions).
