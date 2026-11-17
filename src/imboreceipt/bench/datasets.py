"""Reproducible public dataset loaders (Tier 1 OpenML/imblearn & Tier 2 CREDO-Bench UCI)."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from sklearn.datasets import fetch_openml

from imboreceipt.bench.gate05_sandbox import MultiChannelCohort
from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.types import SemanticType
from imboreceipt.discover.compilers import (
    ConservationMismatchCompiler,
    ConservationRule,
    CriticalSlowingDownCompiler,
    MissingLineSilenceCompiler,
)
from imboreceipt.discover.selector import CandidateChannelSpec

DEFAULT_CACHE_DIR = Path("data/cache")


@dataclass(frozen=True, slots=True)
class Tier1DatasetSpec:
    """Descriptor for a standard Tier-1 imbalanced tabular benchmark."""

    key: str
    display_name: str
    source: str  # "imblearn" or "openml"
    source_id: str | int
    positive_label: str | int | None = None
    target_precision: float = 0.40
    max_fpr_budget: float = 0.01


TIER1_PUBLIC_SPECS: tuple[Tier1DatasetSpec, ...] = (
    Tier1DatasetSpec("mammography", "Mammography", "openml", 310, "1", 0.45, 0.01),
    Tier1DatasetSpec("coil_2000", "COIL 2000 Insurance", "imblearn", "coil_2000", 1, 0.25, 0.02),
    Tier1DatasetSpec("satimage", "Satimage", "imblearn", "satimage", 1, 0.60, 0.01),
    Tier1DatasetSpec("thyroid_sick", "Thyroid Sick", "imblearn", "thyroid_sick", 1, 0.65, 0.01),
    Tier1DatasetSpec("ozone_level", "Ozone Level", "imblearn", "ozone_level", 1, 0.35, 0.02),
    Tier1DatasetSpec("pc1_defects", "PC1 Software Defects", "openml", 1068, "true", 0.35, 0.02),
    Tier1DatasetSpec("yeast_me2", "Yeast ME2", "imblearn", "yeast_me2", 1, 0.35, 0.02),
    Tier1DatasetSpec("abalone_19", "Abalone 19", "imblearn", "abalone_19", 1, 0.20, 0.02),
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_fetch_openml(data_id: int, cache_dir: Path):
    """Fetch OpenML dataset with automatic pandas-3.1 boolean-ARFF fallback."""
    data_home = str(cache_dir / "raw_openml")
    try:
        return fetch_openml(
            data_id=data_id,
            as_frame=True,
            parser="auto",
            data_home=data_home,
        )
    except TypeError:
        ds = fetch_openml(
            data_id=data_id,
            as_frame=True,
            parser="pandas",
            data_home=data_home,
            read_csv_kwargs={"dtype": str},
        )
        for col in ds.frame.columns:
            cleaned = ds.frame[col].astype(str).str.strip().str.strip("'")
            try:
                ds.frame[col] = pd.to_numeric(cleaned)
            except (ValueError, TypeError):
                ds.frame[col] = cleaned
        return ds


def load_tier1_dataset(
    spec: Tier1DatasetSpec,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Fetch (or load from local cache) a Tier-1 public imbalanced dataset with SHA-256 receipt."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"tier1_{spec.key}.csv"

    if not csv_path.exists():
        if spec.source == "imblearn":
            bunch = fetch_datasets(data_home=str(cache_dir / "raw_imblearn"))[str(spec.source_id)]
            df = pd.DataFrame(
                bunch.data,
                columns=[f"f{i}" for i in range(bunch.data.shape[1])],
            )
            df["__target__"] = (bunch.target == 1).astype(int)
        elif spec.source == "openml":
            ds = _safe_fetch_openml(int(spec.source_id), cache_dir)
            df = ds.frame.copy()
            t_col = ds.target_names[0] if ds.target_names else df.columns[-1]
            raw_t = df.pop(t_col).astype(str).str.strip().str.lower()
            pos_str = str(spec.positive_label).strip().lower()
            if (raw_t == pos_str).sum() == 0:
                # Fallback to minority class automatically if OpenML string encoding differs
                vc = raw_t.value_counts()
                minority_val = vc.index[-1]
                df["__target__"] = (raw_t == minority_val).astype(int)
            else:
                df["__target__"] = (raw_t == pos_str).astype(int)
        else:
            raise ValueError(f"Unsupported Tier-1 source: {spec.source}")
        df.to_csv(csv_path, index=False)

    digest = file_sha256(csv_path)
    loaded = pd.read_csv(csv_path)
    y = loaded.pop("__target__").to_numpy(dtype=int)
    return loaded, y, digest


def _fetch_openml_cached(data_id: int, cache_name: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_name}.csv"
    if not path.exists():
        ds = _safe_fetch_openml(data_id, cache_dir)
        df = ds.frame.copy()
        df.to_csv(path, index=False)
    return pd.read_csv(path, low_memory=False)


def load_tier2_ai4i_maintenance(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seed: int = 42,
) -> MultiChannelCohort:
    """Load UCI AI4I 2020 Predictive Maintenance (OpenML 42890) with ground-truth subtypes!

    Contains 10,000 industrial milling runs, 339 rare failures (~3.39%), and explicit ground-truth
    mechanism columns: TWF (Tool Wear), HDF (Heat Dissipation), PWF (Power), OSF (Overstrain),
    and RNF (Random Shock Failure - genuinely memoryless!).
    """
    df = _fetch_openml_cached(42890, "tier2_ai4i_2020", cache_dir)
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)

    # Locate columns flexibly across OpenML naming variants
    def _find(substr: str) -> str:
        for c in df.columns:
            if substr.lower() in c.lower():
                return c
        raise KeyError(f"Column matching '{substr}' not found in AI4I columns: {list(df.columns)}")

    c_type = _find("Type")
    c_air = _find("Air")
    c_proc = _find("Process")
    c_rpm = _find("Rotational")
    c_torq = _find("Torque")
    c_wear = _find("Tool")
    c_fail = _find("Machine")

    y_true = pd.to_numeric(df[c_fail], errors="coerce").fillna(0).to_numpy(dtype=int)

    # Extract ground-truth failure subtype labels
    subtype = np.full(len(df), "none", dtype=object)
    for tag in ("RNF", "TWF", "HDF", "PWF", "OSF"):
        matches = [c for c in df.columns if c.upper() == tag]
        if matches:
            flag = pd.to_numeric(df[matches[0]], errors="coerce").fillna(0).to_numpy(dtype=int)
            subtype[flag == 1] = tag

    X_base = pd.DataFrame(
        {
            "product_type": df[c_type].astype(str),
            "air_temp_k": pd.to_numeric(df[c_air], errors="coerce"),
            "rotational_rpm": pd.to_numeric(df[c_rpm], errors="coerce"),
        }
    )

    proc_temp = pd.to_numeric(df[c_proc], errors="coerce")
    torque = pd.to_numeric(df[c_torq], errors="coerce")
    wear = pd.to_numeric(df[c_wear], errors="coerce")
    rpm_rad = X_base["rotational_rpm"] * (2.0 * np.pi / 60.0)

    # Physical conservation & mechanism features (no target leakage!)
    mech_power_w = torque * rpm_rad
    temp_delta = proc_temp - X_base["air_temp_k"]
    overstrain_nm_min = wear * torque

    rng = np.random.default_rng(seed)
    Z_pool = pd.DataFrame(
        {
            "process_temp_k": proc_temp,
            "thermal_delta_k": temp_delta,
            "torque_nm": torque,
            "mechanical_power_w": mech_power_w,
            "tool_wear_min": wear,
            "overstrain_product": overstrain_nm_min,
            "decoy_vibration_1": rng.normal(0.0, 1.0, size=len(df)),
            "decoy_ambient_humidity": rng.normal(50.0, 10.0, size=len(df)),
        }
    )

    candidates = [
        CandidateChannelSpec(
            "ch_thermal_dissipation",
            ("process_temp_k", "thermal_delta_k"),
            acquisition_cost=1.0,
            mechanism_tag="HDF_thermal",
        ),
        CandidateChannelSpec(
            "ch_power_conservation",
            ("torque_nm", "mechanical_power_w"),
            acquisition_cost=1.0,
            mechanism_tag="PWF_power",
        ),
        CandidateChannelSpec(
            "ch_wear_overstrain",
            ("tool_wear_min", "overstrain_product"),
            acquisition_cost=1.0,
            mechanism_tag="TWF_OSF_wear",
        ),
        CandidateChannelSpec(
            "ch_decoy_vibration",
            ("decoy_vibration_1",),
            acquisition_cost=1.0,
            mechanism_tag="decoy",
        ),
        CandidateChannelSpec(
            "ch_decoy_humidity",
            ("decoy_ambient_humidity",),
            acquisition_cost=1.0,
            mechanism_tag="decoy",
        ),
    ]

    return MultiChannelCohort(
        name="UCI_AI4I_2020_Maintenance",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_thermal_dissipation",
            "ch_power_conservation",
            "ch_wear_overstrain",
        },
        y_true=y_true,
        y_tilde=y_true.copy(),
        contract=DecisionContract(
            task_name="AI4I_Predictive_Maintenance",
            target_precision=0.60,
            max_fpr_budget=0.01,
            cost_fp=1.0,
            cost_fn=30.0,
            graded_action_cost_fp=0.20,
        ),
        acquisition_budget=2.0,
        base_semantic_types={
            "product_type": SemanticType.NOMINAL,
            "air_temp_k": SemanticType.CONTINUOUS,
            "rotational_rpm": SemanticType.CONTINUOUS,
        },
        mechanism_subtype=subtype,
    )


def load_tier2_taiwan_credit_panel(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seed: int = 42,
) -> MultiChannelCohort:
    """Load UCI Taiwan Credit Default 6-month panel (OpenML 42477, N=30,000).

    Target: Severe rare default (default payment next month on accounts without prior chronic
    3+ month delinquency, or severe high-balance default tail ~3.8% prevalence), allowing clean
    testing of Critical Slowing Down, Missing-Line Payment Silence, and Flow Conservation!
    """
    df = _fetch_openml_cached(42477, "tier2_taiwan_credit", cache_dir)
    # Standardize column names to uppercase and map OpenML X1..X23 schema if present
    df.columns = [c.strip().upper() for c in df.columns]
    if "X1" in df.columns and "LIMIT_BAL" not in df.columns:
        df = df.rename(
            columns={
                "X1": "LIMIT_BAL",
                "X2": "SEX",
                "X3": "EDUCATION",
                "X4": "MARRIAGE",
                "X5": "AGE",
                "X6": "PAY_1",
                "X7": "PAY_2",
                "X8": "PAY_3",
                "X9": "PAY_4",
                "X10": "PAY_5",
                "X11": "PAY_6",
                "X12": "BILL_AMT1",
                "X13": "BILL_AMT2",
                "X14": "BILL_AMT3",
                "X15": "BILL_AMT4",
                "X16": "BILL_AMT5",
                "X17": "BILL_AMT6",
                "X18": "PAY_AMT1",
                "X19": "PAY_AMT2",
                "X20": "PAY_AMT3",
                "X21": "PAY_AMT4",
                "X22": "PAY_AMT5",
                "X23": "PAY_AMT6",
            }
        )

    # Rename PAY_0 -> PAY_1 if needed; ordered oldest (t6) to newest (t1)
    if "PAY_0" in df.columns and "PAY_1" not in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})

    target_candidates = [c for c in df.columns if "DEFAULT" in c or c == "Y" or c == "CLASS"]
    t_col = target_candidates[-1] if target_candidates else df.columns[-1]
    raw_default = pd.to_numeric(df[t_col], errors="coerce").fillna(0).to_numpy(dtype=int)

    limit_bal = pd.to_numeric(df["LIMIT_BAL"], errors="coerce").fillna(1.0)
    bill_cols = ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]
    pay_amt_cols = ["PAY_AMT6", "PAY_AMT5", "PAY_AMT4", "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]
    pay_stat_cols = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_1"]

    for c in bill_cols + pay_amt_cols + pay_stat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Rare high-exposure default event (default == 1 AND utilization > 55% of limit)
    high_util = (df["BILL_AMT1"] / np.maximum(1000.0, limit_bal)) > 0.55
    y_true = ((raw_default == 1) & high_util).to_numpy(dtype=int)

    X_base = pd.DataFrame(
        {
            "limit_bal": limit_bal,
            "sex": df["SEX"].astype(str),
            "education": df["EDUCATION"].astype(str),
            "marriage": df["MARRIAGE"].astype(str),
            "age": pd.to_numeric(df["AGE"], errors="coerce"),
            "initial_bill_m6": df["BILL_AMT6"],
            "initial_status_m6": df["PAY_6"],
        }
    )

    csd_df = CriticalSlowingDownCompiler({"bill": bill_cols}).fit(df).transform(df)
    eesh_df = MissingLineSilenceCompiler({"payment": pay_amt_cols}).fit(df).transform(df)
    inv_df = (
        ConservationMismatchCompiler(
            [
                ConservationRule(
                    "net_debt_surge",
                    "flow_balance",
                    ("BILL_AMT1", "PAY_AMT2", "PAY_AMT1"),
                    ("BILL_AMT3",),
                )
            ]
        )
        .fit(df)
        .transform(df)
    )

    rng = np.random.default_rng(seed)
    Z_pool = pd.concat(
        [
            csd_df,
            eesh_df,
            inv_df,
            df[["PAY_3", "PAY_2", "PAY_1", "BILL_AMT1", "PAY_AMT1"]].rename(
                columns={
                    "PAY_3": "precursor_pay_m3",
                    "PAY_2": "precursor_pay_m2",
                    "PAY_1": "precursor_pay_m1",
                    "BILL_AMT1": "latest_bill_m1",
                    "PAY_AMT1": "latest_pay_m1",
                }
            ),
            pd.DataFrame(
                {
                    "decoy_macro_idx": rng.normal(0, 1, len(df)),
                    "decoy_branch_noise": rng.normal(0, 1, len(df)),
                }
            ),
        ],
        axis=1,
    )

    candidates = [
        CandidateChannelSpec(
            "ch_recent_repayment_status",
            ("precursor_pay_m3", "precursor_pay_m2", "precursor_pay_m1"),
            acquisition_cost=1.0,
            mechanism_tag="precursor_status",
        ),
        CandidateChannelSpec(
            "ch_bill_csd_trajectory",
            tuple(csd_df.columns) + ("latest_bill_m1",),
            acquisition_cost=1.0,
            mechanism_tag="critical_slowing_down",
        ),
        CandidateChannelSpec(
            "ch_payment_silence_conservation",
            tuple(eesh_df.columns) + tuple(inv_df.columns) + ("latest_pay_m1",),
            acquisition_cost=1.0,
            mechanism_tag="missing_line_conservation",
        ),
        CandidateChannelSpec(
            "ch_decoy_macro",
            ("decoy_macro_idx",),
            acquisition_cost=1.0,
            mechanism_tag="decoy",
        ),
        CandidateChannelSpec(
            "ch_decoy_branch",
            ("decoy_branch_noise",),
            acquisition_cost=1.0,
            mechanism_tag="decoy",
        ),
    ]

    return MultiChannelCohort(
        name="UCI_Taiwan_Credit_6m_Panel",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_recent_repayment_status",
            "ch_bill_csd_trajectory",
            "ch_payment_silence_conservation",
        },
        y_true=y_true,
        y_tilde=y_true.copy(),
        contract=DecisionContract(
            task_name="Taiwan_Severe_Default",
            target_precision=0.45,
            max_fpr_budget=0.02,
            cost_fp=1.0,
            cost_fn=20.0,
            graded_action_cost_fp=0.20,
        ),
        acquisition_budget=2.0,
        base_semantic_types={
            "limit_bal": SemanticType.CONTINUOUS,
            "sex": SemanticType.NOMINAL,
            "education": SemanticType.NOMINAL,
            "marriage": SemanticType.NOMINAL,
            "age": SemanticType.COUNT,
            "initial_bill_m6": SemanticType.CONTINUOUS,
            "initial_status_m6": SemanticType.ORDINAL,
        },
    )


def load_tier2_diabetes130(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_patients: int | None = 25000,
    seed: int = 42,
) -> MultiChannelCohort:
    """Load UCI Diabetes 130-US Hospitals (OpenML 4541) with patient groups & receipt censoring.

    Explicitly models:
    - ``competing_censor_mask``: expired/hospice discharges (disposition IDs 11, 13, 14, 19, 20, 21)
      that cannot be readmitted within 30 days.
    - ``future_positive_mask``: ``>30`` day readmissions censored by the 30-day binary cutoff.
    - ``group_ids``: ``patient_nbr`` (zero patient leakage across splits).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "tier2_diabetes130.csv"
    if not csv_path.exists():
        uci_url = (
            "https://archive.ics.uci.edu/static/public/296/"
            "diabetes+130-us+hospitals+for+years+1999-2008.zip"
        )
        with urllib.request.urlopen(uci_url, timeout=60) as resp:
            outer_bytes = resp.read()
        with zipfile.ZipFile(io.BytesIO(outer_bytes)) as zf:
            names = zf.namelist()
            if "diabetic_data.csv" in names:
                raw_csv = zf.read("diabetic_data.csv")
            else:
                inner_zips = [n for n in names if n.endswith(".zip")]
                if not inner_zips:
                    raise RuntimeError(f"diabetic_data.csv not found in UCI archive: {names}")
                with zipfile.ZipFile(io.BytesIO(zf.read(inner_zips[0]))) as inner_zf:
                    csv_matches = [
                        n for n in inner_zf.namelist() if n.endswith("diabetic_data.csv")
                    ]
                    raw_csv = inner_zf.read(csv_matches[0])
        csv_path.write_bytes(raw_csv)

    df = pd.read_csv(csv_path, low_memory=False).replace("?", np.nan)
    df.columns = [c.strip() for c in df.columns]

    # Deterministic patient-grouped subsample if requested for fast reproducible benchmark execution
    if max_patients is not None and "patient_nbr" in df.columns:
        uniq_p = df["patient_nbr"].unique()
        if len(uniq_p) > max_patients:
            rng_sub = np.random.default_rng(seed)
            keep_p = set(rng_sub.choice(uniq_p, size=max_patients, replace=False).tolist())
            df = df[df["patient_nbr"].isin(keep_p)].reset_index(drop=True)

    raw_readm = df["readmitted"].astype(str).str.strip().str.upper()
    y_tilde = (raw_readm == "<30").to_numpy(dtype=int)
    future_pos = (raw_readm == ">30").to_numpy(dtype=bool)

    disp_num = pd.to_numeric(df["discharge_disposition_id"], errors="coerce").fillna(-1).astype(int)
    competing_censor = disp_num.isin({11, 13, 14, 19, 20, 21}).to_numpy(dtype=bool)

    groups = (
        df["patient_nbr"].to_numpy()
        if "patient_nbr" in df.columns
        else np.arange(len(df), dtype=int)
    )

    X_base = pd.DataFrame(
        {
            "race": df["race"].astype(str),
            "gender": df["gender"].astype(str),
            "age_bucket": df["age"].astype(str),
            "admission_type_id": df["admission_type_id"].astype(str),
            "admission_source_id": df["admission_source_id"].astype(str),
            "time_in_hospital": pd.to_numeric(df["time_in_hospital"], errors="coerce"),
            "num_lab_procedures": pd.to_numeric(df["num_lab_procedures"], errors="coerce"),
        }
    )

    rng = np.random.default_rng(seed)
    Z_pool = pd.DataFrame(
        {
            "number_inpatient": pd.to_numeric(df["number_inpatient"], errors="coerce"),
            "number_emergency": pd.to_numeric(df["number_emergency"], errors="coerce"),
            "number_outpatient": pd.to_numeric(df["number_outpatient"], errors="coerce"),
            "discharge_disposition_id": df["discharge_disposition_id"].astype(str),
            "medical_specialty": df["medical_specialty"].astype(str),
            "number_diagnoses": pd.to_numeric(df["number_diagnoses"], errors="coerce"),
            "diag_1_chapter": df["diag_1"].astype(str).str.slice(0, 3),
            "num_medications": pd.to_numeric(df["num_medications"], errors="coerce"),
            "insulin": df["insulin"].astype(str),
            "change": df["change"].astype(str),
            "diabetesMed": df["diabetesMed"].astype(str),
            "A1Cresult": df["A1Cresult"].astype(str),
            "max_glu_serum": df["max_glu_serum"].astype(str),
            "decoy_ward_code": rng.integers(1, 15, size=len(df)).astype(str),
            "decoy_ chart_noise": rng.normal(0, 1, size=len(df)),
        }
    )

    candidates = [
        CandidateChannelSpec(
            "ch_prior_utilization",
            ("number_inpatient", "number_emergency", "number_outpatient"),
            acquisition_cost=1.0,
            mechanism_tag="utilization_history",
        ),
        CandidateChannelSpec(
            "ch_discharge_routing",
            ("discharge_disposition_id", "medical_specialty"),
            acquisition_cost=1.0,
            mechanism_tag="care_transition",
        ),
        CandidateChannelSpec(
            "ch_diagnosis_burden",
            ("number_diagnoses", "diag_1_chapter"),
            acquisition_cost=1.0,
            mechanism_tag="comorbidity",
        ),
        CandidateChannelSpec(
            "ch_medication_intensification",
            ("num_medications", "insulin", "change", "diabetesMed"),
            acquisition_cost=1.0,
            mechanism_tag="pharmacotherapy",
        ),
        CandidateChannelSpec(
            "ch_glycemic_biomarkers",
            ("A1Cresult", "max_glu_serum"),
            acquisition_cost=1.0,
            mechanism_tag="glycemic_panel",
        ),
        CandidateChannelSpec(
            "ch_decoy_ward",
            ("decoy_ward_code", "decoy_ chart_noise"),
            acquisition_cost=1.0,
            mechanism_tag="decoy",
        ),
    ]

    return MultiChannelCohort(
        name="UCI_Diabetes_130US_Hospitals",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_prior_utilization",
            "ch_discharge_routing",
            "ch_diagnosis_burden",
            "ch_medication_intensification",
        },
        y_true=y_tilde,
        y_tilde=y_tilde,
        contract=DecisionContract(
            task_name="Diabetes130_Readmission_30d",
            target_precision=0.28,
            max_fpr_budget=0.05,
            cost_fp=1.0,
            cost_fn=10.0,
            graded_action_cost_fp=0.20,
        ),
        acquisition_budget=2.0,
        base_semantic_types={
            "race": SemanticType.NOMINAL,
            "gender": SemanticType.NOMINAL,
            "age_bucket": SemanticType.ORDINAL,
            "admission_type_id": SemanticType.NOMINAL,
            "admission_source_id": SemanticType.NOMINAL,
            "time_in_hospital": SemanticType.COUNT,
            "num_lab_procedures": SemanticType.RECEIPT_FLAG,
        },
        group_ids=groups,
        future_positive_mask=future_pos,
        competing_censor_mask=competing_censor,
        receipt_cols=["num_lab_procedures"],
    )


def load_tier2_credit_card_fraud(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_negatives: int | None = 60000,
    window_idx: int | None = None,
    n_windows: int = 5,
    seed: int = 42,
) -> MultiChannelCohort:
    """Load ULB Credit Card Fraud (OpenML 1597) with strict chronological ordering on Time!

    - When ``max_negatives`` is None and ``window_idx`` is provided (0..n_windows-1),
      slices a contiguous 76% chronological horizon window across the full 284,807-row
      transaction stream so multi-window replication evaluates genuinely distinct
      out-of-time regimes without negative subsampling!
    """
    df = _fetch_openml_cached(1597, "tier2_ulb_creditcard", cache_dir)
    df.columns = [c.strip() for c in df.columns]
    df["__seq_order__"] = (
        pd.to_numeric(df["Time"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "Time" in df.columns
        else np.arange(len(df), dtype=float)
    )

    raw_cls = df["Class"].astype(str).str.strip().str.replace("'", "")
    y_all = (raw_cls == "1").to_numpy(dtype=int)
    df["__y__"] = y_all
    df = df.sort_values("__seq_order__", kind="mergesort").reset_index(drop=True)

    if window_idx is not None and n_windows > 1:
        span_frac = 0.76
        step_frac = (1.0 - span_frac) / float(n_windows - 1)
        start_f = step_frac * float(window_idx)
        lo_i = int(round(start_f * len(df)))
        hi_i = min(len(df), int(round((start_f + span_frac) * len(df))))
        df = df.iloc[lo_i:hi_i].reset_index(drop=True)

    if max_negatives is not None:
        pos_df = df[df["__y__"] == 1]
        neg_df = df[df["__y__"] == 0]
        if len(neg_df) > max_negatives:
            rng = np.random.default_rng(seed)
            keep_neg = rng.choice(neg_df.index.to_numpy(), size=max_negatives, replace=False)
            df = pd.concat([pos_df, df.loc[keep_neg]], axis=0)
            df = df.sort_values("__seq_order__", kind="mergesort").reset_index(drop=True)

    y_true = df.pop("__y__").to_numpy(dtype=int)
    time_arr = df.pop("__seq_order__").to_numpy(dtype=float)

    X_base = pd.DataFrame(
        {
            "Amount": pd.to_numeric(df["Amount"], errors="coerce"),
            **{f"V{i}": pd.to_numeric(df[f"V{i}"], errors="coerce") for i in range(1, 6)},
        }
    )

    rng = np.random.default_rng(seed + 7)
    Z_pool = pd.DataFrame(
        {
            **{f"V{i}": pd.to_numeric(df[f"V{i}"], errors="coerce") for i in range(6, 29)},
            "decoy_merchant_hash": rng.normal(0, 1, len(df)),
            "decoy_latency_ms": rng.exponential(20.0, len(df)),
        }
    )

    candidates = [
        CandidateChannelSpec(
            "ch_behavior_block_A",
            tuple(f"V{i}" for i in range(6, 13)),
            acquisition_cost=1.0,
            mechanism_tag="pca_block_A",
        ),
        CandidateChannelSpec(
            "ch_device_block_B",
            tuple(f"V{i}" for i in range(13, 20)),
            acquisition_cost=1.0,
            mechanism_tag="pca_block_B",
        ),
        CandidateChannelSpec(
            "ch_velocity_block_C",
            tuple(f"V{i}" for i in range(20, 29)),
            acquisition_cost=1.0,
            mechanism_tag="pca_block_C",
        ),
        CandidateChannelSpec(
            "ch_decoy_gateway",
            ("decoy_merchant_hash", "decoy_latency_ms"),
            acquisition_cost=1.0,
            mechanism_tag="decoy",
        ),
    ]

    return MultiChannelCohort(
        name="ULB_CreditCard_Fraud_Chronological",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_behavior_block_A",
            "ch_device_block_B",
            "ch_velocity_block_C",
        },
        y_true=y_true,
        y_tilde=y_true.copy(),
        contract=DecisionContract(
            task_name="ULB_CreditCard_Fraud",
            target_precision=0.75,
            max_fpr_budget=0.002,
            cost_fp=1.0,
            cost_fn=50.0,
            graded_action_cost_fp=0.10,
        ),
        acquisition_budget=2.0,
        time_order=time_arr,
    )
