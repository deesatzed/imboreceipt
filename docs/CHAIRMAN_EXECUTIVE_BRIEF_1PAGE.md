# 1-Page Executive Walkthrough: How `CREDO` Works (A Step-by-Step Thought Experiment on 101,766 Hospital Readmissions)

**Author:** Wayne A. Satz, MD (Department of Emergency Medicine, Temple University)  
**Context:** A plain-English walkthrough of our three statistical principles using a single experiment from our paper (**UCI 30-Day Hospital Readmission**: $101,766$ hospital stays across $71,518$ diabetic patients, where **$11.2\%$** bounce back within 30 days).

---

### The Thought Experiment Setup: Why Basic Intake Data Stalls
Imagine a patient is being discharged, and we want to predict if they will suffer an unplanned **30-day readmission** ($11.2\%$ base rate) so our care-transition team can intervene.
* **What we have for free on every patient ("Stage-1 Intake Table"):** Just **7 basic registration fields**: `Age`, `Gender`, `Race`, `Admission Type`, `Admission Source`, `Length of Stay`, and `Total Lab Count`.
* **The Bottleneck:** Using only those 7 basic fields, even our best calibrated gradient-boosted tree (`LightGBM`) achieves a Precision–Recall score (`AUPRC`) of just **`0.1305`** (barely above random guessing at `0.112`). Why? Because thousands of patients look **identical** on those 7 basic numbers.

---

### Step 1: Why `SMOTE` and Algorithm Tweaking Fail (*Principle 1: You Can't Re-Sort a Tie*)
* **What a standard data science team tries first:** *"Our score is only `0.1305` because readmissions are the minority class (`11.2%`). Let's run `SMOTE` (synthetic minority oversampling) or increase the loss weight on readmitted patients!"*
* **Watch what happens in our thought experiment:**  
  Picture **100 patients** who share the **exact same 7 intake numbers** (e.g., 65-year-old female, 4-day stay, 45 labs). In reality, **15 of them will be readmitted (`15%` true risk)** and **85 will go home and recover fine (`85%` false alarms)**.  
  Suppose our care-transition team requires at least **$28\%$ confidence (precision)** before paying for a home-health nurse:

$$\text{Information Gap} = \text{Target Confidence Needed } (28\%) - \text{Intake Confidence } (15\%) = +13\% \text{ gap}$$

* **Why `SMOTE` hits a brick wall:** If you generate synthetic copies of the 15 readmitted patients (`SMOTE`) or multiply their training weight by $3\times$, the algorithm simply inflates the predicted score of **all 100 identical-looking patients together** from $15\%$ to $35\%$. It **never changes who ranks above whom**—the 85 false alarms are still mixed right in with the 15 true readmissions! Sure enough, `SMOTE` gets **`0.1295` AUPRC** (zero gain over `0.1305`; and across 10 standard benchmarks in our paper, `SMOTE`, `EasyEnsemble`, and `RUSBoost` win **0 out of 10 times**).
* **Rule 1:** When $\text{Information Gap} > 0$, **stop tweaking the algorithm on the intake table**. The only way to separate the 15 true readmissions from the 85 look-alike false alarms is to pull a **second, independent data source ("Stage-2 data")**.

---

### Step 2: Picking the Right Stage-2 Data Without Buying a "Copycat" (*Principle 2: The Look-Alike Stress Test*)
* **The Choice:** Suppose the hospital EHR has **7 candidate Stage-2 data feeds** we could extract (e.g., *Channel 1: Prior 12-Month Inpatient/ER Visits*, *Channel 2: Discharge Disposition & Specialty*, *Channel 3: Diagnosis Count*, *Channel 4: Medication Changes*, *Channel 5: HbA1c/Glucose*, *Channel 6: A "Hospital Acuity Index" that secretly just repackages Length-of-Stay + Lab Count*, and *Channel 7: Random Ward Codes*). Our budget allows us to add **2 channels**.
* **Why Standard Feature Selection Gets Tricked:** If you test candidate channels by comparing readmitted patients against **random non-readmitted patients** (classical Unmatched Case-Control Mutual Information), a "Copycat" or general-illness channel looks great—because readmitted patients had longer stays than random healthy patients! In our experiment, Unmatched Mutual Information picks a suboptimal channel on 4 out of 5 splits and stalls at **`0.1979` AUPRC**.
* **What `CREDO` Does Instead:** We match every true readmission against **4 "Look-Alike False Alarms"**—patients who got the **exact same high risk score on the 7 intake fields**, but were *not* readmitted:

$$\text{Relative Evidence Gain (REG)} = \frac{\text{\% of True Readmissions flagged by the candidate Stage-2 channel}}{\text{\% of Look-Alike False Alarms flagged by the candidate Stage-2 channel}}$$

* **Watch the Look-Alike Stress Test in action:**
  1. **Channel 6 ("Copycat Acuity Index"):** Because the True Readmissions and the Look-Alike False Alarms *already* have the exact same Length-of-Stay and Lab Count, the Copycat Index flags both groups at the exact same rate: $\text{REG} = \frac{30\%}{30\%} = \mathbf{1.0}$ (**Zero new information $\to$ Rejected!**).
  2. **Channels 1 & 2 (*Prior Inpatient/ER Visits* + *Discharge Routing*):** Among patients who looked identical at admission, those with prior hospitalizations and skilled-nursing discharges bounce back far more often than their look-alikes: $\text{REG} \gg \mathbf{1.0}$ (**Selected on 5 out of 5 splits!**).
* **Result:** Testing only the matched pilot (`55.9%` of patients—saving `44.1%` of discovery queries on Diabetes, and saving **`82%–99%`** on rarer `0.2%–3.5%` cohorts), `CREDO` lifts AUPRC from **`0.1305` $\to$ `0.2231`** ($p = 0.030$ vs. Unmatched MI).

---

### Step 3: At Discharge, Who Actually Needs the Stage-2 Lookup? (*Principle 3: Double-Checking the Red Bucket*)
* **The Live Triage Problem:** Now our system is live. Every patient gets the free 7-variable **Stage-1 Intake Screen** instantly. Pulling the **Stage-2 Channels** on $100\%$ of patients is wasteful, so standard two-stage "Conformal Prediction" splits patients into three buckets based on their Stage-1 score:
  * **Green Bucket `{0}` (Low Score):** Clearly safe $\to$ *Skip Stage-2.*
  * **Yellow Bucket `{0, 1}` (Middle Score):** Uncertain $\to$ *Pull Stage-2 data.*
  * **Red Bucket `{1}` (Highest Intake Score):** Standard methods call this *"Confident Positive"* and **skip Stage-2**, assuming these patients are guaranteed readmissions.
* **Why Skipping the Red Bucket Fails (The Base-Rate Trap):** Look inside that Stage-1 **Red Bucket `{1}`**:

$$\text{True Precision Inside the Red Bucket} = \frac{\text{Caught Readmissions}}{\text{Caught Readmissions} + \text{False Alarms}}$$

  Because readmission is only $11.2\%$ overall—and the 7 intake fields are weak (`0.1305`)—**roughly $80\%$ of the patients in the Stage-1 Red Bucket are actually False Alarms** (patients who had a 6-day stay and 50 labs, but have *zero* prior hospitalizations and are going home healthy)!
  * **Standard Conformal Routing (Yellow Bucket Only):** Skips the Red Bucket, leaving all those unverified Stage-1 false alarms sitting at the very top of the final risk list $\to$ drops to **`0.1981` AUPRC**.
  * **Our Fix (`CREDO` Red + Yellow Verification):** Safely skips the Green Bucket (`26.0%` of discharges saved on Diabetes; **`41%–79%` saved** on rarer cohorts) and **uses Stage-2 data to double-check the Red Bucket + top of the Yellow Bucket**, weeding out the Red-Bucket false alarms and lifting AUPRC to **`0.2091`** at the exact same query budget (and winning on **9 out of 9 cohorts** in the paper, including `0.55 -> 0.89` on `76k` trucks and `0.23 -> 0.48` on `590k` transactions).
