"""Fine-grained multi-channel cohort loaders (M = 15..26 native; M = 25..36 with stress pools).

Addresses editorial requirements for realistic channel dimensionality, explicit X_base and channel
definitions, discharge-landmark clinical temporal validity with fairness/missingness auditing,
pure-noise null channels, Stage-2-to-Stage-2 redundant twins, and baseline echoes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imboreceipt.bench.datasets import DEFAULT_CACHE_DIR
from imboreceipt.bench.wave2_datasets import (
    _prepare_home_credit_7table_cache,
    _prepare_ieee_cis_cache,
)
from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.types import SemanticType
from imboreceipt.discover.selector import CandidateChannelSpec


@dataclass
class FineGrainedCohort:
    """Self-describing multi-channel cohort with X_base, channel pool, and audit metadata."""

    name: str
    display_name: str
    domain: str
    provenance: str
    landmark_description: str
    X_base: pd.DataFrame
    Z_pool: pd.DataFrame
    candidates: list[CandidateChannelSpec]
    y: np.ndarray
    contract: DecisionContract
    acquisition_budget: float
    per_instance_query_cost: float
    base_semantic_types: dict[str, SemanticType]
    group_ids: np.ndarray | None = None
    time_order: np.ndarray | None = None
    fairness_subgroups: dict[str, np.ndarray] = field(default_factory=dict)
    missingness_rates: dict[str, float] = field(default_factory=dict)
    channel_Fixed_interface_cost: dict[str, float] = field(default_factory=dict)
    eligibility_audit: dict[str, Any] = field(default_factory=dict)


def _staggered_chrono_window(
    df: pd.DataFrame,
    window_idx: int | None,
    n_windows: int,
    max_window_rows: int = 160000,
) -> pd.DataFrame:
    """Extract a causal forward chronological window across ``n_windows`` staggered eras."""
    if window_idx is None or n_windows <= 1:
        if len(df) > max_window_rows:
            return df.iloc[:max_window_rows].reset_index(drop=True)
        return df
    n_tot = len(df)
    win_len = min(max_window_rows, max(40000, int(round(0.45 * n_tot))))
    if win_len >= n_tot:
        return df.reset_index(drop=True)
    step = (n_tot - win_len) / float(max(1, n_windows - 1))
    lo = int(round(window_idx * step))
    hi = min(n_tot, lo + win_len)
    return df.iloc[lo:hi].reset_index(drop=True)


TWIN_SOURCE_IDS_BY_COHORT: dict[str, tuple[str, ...]] = {
    "UCI_Diabetes_Discharge": ("hie_prior_inpatient", "icd_primary_grouper", "lab_hba1c_panel"),
    "Home_Credit_7Table": ("ext_bureau_vendor_2", "ext_bureau_vendor_3", "inst_dpd_delinquency"),
    "IEEE_CIS_Fraud": ("cnt_card_links_C1_4", "cnt_device_links_C12_14", "vesta_block_01"),
    "Scania_APS_Fleet": ("hist_sensor_ag", "hist_sensor_ee", "hist_sensor_ba"),
    "NeurIPS_BAF": ("credit_bureau_score", "device_email_graph_8w", "other_cards_registry"),
    "UCI_Taiwan_Credit": ("repay_status_m1", "repay_status_m2", "cash_payment_m1"),
}


def _append_stress_channels(
    X_base: pd.DataFrame,
    z_dict: dict[str, Any],
    candidates: list[CandidateChannelSpec],
    twin_source_ids: tuple[str, ...],
    seed: int,
    train_idx: np.ndarray | None = None,
) -> None:
    """Append 4 pure-noise null channels, 3 Stage-2 redundant twins, and 3 baseline echoes.

    When ``train_idx`` is provided, all location/scale statistics (medians, means, SDs) used to
    construct noisy Stage-2 twins and baseline echoes are estimated strictly on ``train_idx``.
    """
    n = len(X_base)
    rng = np.random.default_rng(seed + 9103)
    fit_idx = np.arange(n, dtype=int) if train_idx is None else np.asarray(train_idx, dtype=int)

    # 1. Four pure-noise null channels Z_null _|_ (Y, X_base, Z_native)
    for k in range(1, 5):
        c1 = f"null_gauss_{k}"
        c2 = f"null_heavy_{k}"
        c3 = f"null_count_{k}"
        z_dict[c1] = rng.normal(0.0, 1.0, n).astype(np.float32)
        z_dict[c2] = rng.standard_t(df=4.0, size=n).astype(np.float32)
        z_dict[c3] = rng.poisson(lam=1.5, size=n).astype(np.float32)
        candidates.append(
            CandidateChannelSpec(
                f"null_channel_{k}",
                (c1, c2, c3),
                acquisition_cost=1.0,
                mechanism_tag="pure_noise_null",
            )
        )

    # 2. Three Stage-2-to-Stage-2 redundant twin channels (noisy copies of native Stage-2 channels)
    spec_by_id = {c.channel_id: c for c in candidates if c.mechanism_tag != "pure_noise_null"}
    chosen_twins = [cid for cid in twin_source_ids if cid in spec_by_id][:3]
    for t_idx, src_id in enumerate(chosen_twins, start=1):
        src_spec = spec_by_id[src_id]
        twin_cols: list[str] = []
        for j, col in enumerate(src_spec.columns[:3]):
            raw_s = z_dict[col]
            s_tmp = raw_s if isinstance(raw_s, pd.Series) else pd.Series(raw_s)
            if s_tmp.dtype != object and str(s_tmp.dtype) != "category":
                arr_v = pd.to_numeric(s_tmp, errors="coerce").to_numpy(dtype=float)
            else:
                # Fit category code vocabulary strictly on fit_idx
                fit_cats = pd.Index(s_tmp.iloc[fit_idx].dropna().astype(str).unique())
                cat_type = pd.CategoricalDtype(categories=fit_cats, ordered=False)
                arr_v = s_tmp.astype(str).astype(cat_type).cat.codes.to_numpy(dtype=float)
                arr_v[arr_v < 0] = np.nan
            fit_sub = arr_v[fit_idx]
            med_fit = float(np.nanmedian(fit_sub)) if np.any(~np.isnan(fit_sub)) else 0.0
            arr_clean = np.nan_to_num(arr_v, nan=med_fit)
            sd_v = max(1e-3, float(np.std(arr_clean[fit_idx])))
            t_col = f"s2_twin_{t_idx}_{j}"
            # Correlated noisy twin (Corr ~ 0.94) of native Stage-2 channel src_id
            z_dict[t_col] = (arr_clean + rng.normal(0.0, 0.35 * sd_v, n)).astype(np.float32)
            twin_cols.append(t_col)
        candidates.append(
            CandidateChannelSpec(
                f"s2_twin_of_{src_id}",
                tuple(twin_cols),
                acquisition_cost=src_spec.acquisition_cost,
                mechanism_tag="stage2_twin",
            )
        )

    # 3. Three baseline-echo channels Z_echo = h(X_base) + eps
    num_base_arrays: list[np.ndarray] = []
    for col in X_base.columns[:6]:
        s = X_base[col]
        if s.dtype == object or str(s.dtype) == "category":
            fit_cats = pd.Index(s.iloc[fit_idx].dropna().astype(str).unique())
            cat_type = pd.CategoricalDtype(categories=fit_cats, ordered=False)
            v = s.astype(str).astype(cat_type).cat.codes.to_numpy(dtype=float)
            v[v < 0] = 0.0
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            fit_v = v[fit_idx]
            med_b = float(np.nanmedian(fit_v)) if np.any(~np.isnan(fit_v)) else 0.0
            v = np.nan_to_num(v, nan=med_b)
        mu_fit = float(np.mean(v[fit_idx]))
        sd = max(1e-6, float(np.std(v[fit_idx])))
        num_base_arrays.append((v - mu_fit) / sd)

    while len(num_base_arrays) < 3:
        num_base_arrays.append(rng.normal(0.0, 1.0, n))

    b0, b1, b2 = num_base_arrays[0], num_base_arrays[1], num_base_arrays[2]
    z_dict["base_echo_lin_a"] = (0.8 * b0 + 0.6 * b1 + rng.normal(0.0, 0.25, n)).astype(np.float32)
    z_dict["base_echo_lin_b"] = (0.7 * b1 - 0.5 * b2 + rng.normal(0.0, 0.25, n)).astype(np.float32)
    z_dict["base_echo_quad_a"] = (0.5 * (b0**2) + 0.6 * b2 + rng.normal(0.0, 0.30, n)).astype(
        np.float32
    )
    z_dict["base_echo_inter_a"] = (0.7 * b0 * b1 + 0.5 * b0 + rng.normal(0.0, 0.30, n)).astype(
        np.float32
    )
    candidates.extend(
        [
            CandidateChannelSpec(
                "base_echo_linear",
                ("base_echo_lin_a", "base_echo_lin_b"),
                acquisition_cost=1.0,
                mechanism_tag="baseline_echo",
            ),
            CandidateChannelSpec(
                "base_echo_quadratic",
                ("base_echo_quad_a",),
                acquisition_cost=1.0,
                mechanism_tag="baseline_echo",
            ),
            CandidateChannelSpec(
                "base_echo_interaction",
                ("base_echo_inter_a",),
                acquisition_cost=1.0,
                mechanism_tag="baseline_echo",
            ),
        ]
    )


def materialize_leak_free_stress_pool(
    cohort: FineGrainedCohort,
    train_idx: np.ndarray,
    seed: int = 42,
    twin_source_ids: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[CandidateChannelSpec]]:
    """Construct augmented stress pool with all twin/echo moments fit strictly on ``train_idx``."""
    z_dict: dict[str, Any] = {col: cohort.Z_pool[col].copy() for col in cohort.Z_pool.columns}
    cands = list(cohort.candidates)
    t_ids = (
        twin_source_ids
        if twin_source_ids is not None
        else TWIN_SOURCE_IDS_BY_COHORT.get(cohort.name, tuple(c.channel_id for c in cands[:3]))
    )
    _append_stress_channels(
        cohort.X_base,
        z_dict,
        cands,
        twin_source_ids=t_ids,
        seed=seed,
        train_idx=np.asarray(train_idx, dtype=int),
    )
    return pd.DataFrame(z_dict), cands


def _icd9_organ_grouper(series: pd.Series) -> pd.Series:
    """Map raw ICD-9 codes to Strack et al. (2014) 9 clinical organ-system categories."""
    s = series.fillna("MISSING").astype(str).str.strip()
    out = np.full(len(s), "Other", dtype=object)
    is_miss = s == "MISSING"
    is_ve = s.str.startswith(("V", "E"))
    num = pd.to_numeric(s, errors="coerce")

    out[is_miss] = "Missing"
    out[is_ve] = "External_Supplemental"
    out[(num >= 390) & (num <= 459) | (num == 785)] = "Circulatory"
    out[(num >= 460) & (num <= 519) | (num == 786)] = "Respiratory"
    out[(num >= 520) & (num <= 579) | (num == 787)] = "Digestive"
    out[(num >= 250.0) & (num < 251.0)] = "Diabetes"
    out[(num >= 800) & (num <= 999)] = "Injury"
    out[(num >= 710) & (num <= 739)] = "Musculoskeletal"
    out[(num >= 580) & (num <= 629) | (num == 788)] = "Genitourinary"
    out[(num >= 140) & (num <= 239)] = "Neoplasms"
    return pd.Series(out, index=series.index, dtype=str)


def load_fine_diabetes_discharge(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    include_stress_channels: bool = False,
    include_race_in_base: bool = True,
    max_patients: int | None = None,
    seed: int = 42,
) -> FineGrainedCohort:
    """UCI Diabetes 130-US Hospitals at the Hospital Discharge Landmark (M=16 native; M=26 stress).

    Clinical landmark validity:
    - Upfront outcome-independent eligibility excludes inpatient deaths & hospice transfers
      (disposition IDs {11,13,14,19,20,21}: 2,423 of 101,766 encounters, leaving 99,343 stays).
    - X_base contains all 13 local inpatient ADT & stay-summary variables already recorded in the
      hospital's local encounter header at discharge (demographics, admission/discharge codes,
      length of stay, cumulative inpatient lab/procedure/medication/diagnosis counts).
    - Stage-2 candidate channels (M = 16) represent external HIE prior-utilization queries,
      ICD-9 clinical organ-system groupers, specialty service feeds, glycemic lab interfaces,
      and class-specific medication reconciliation modules—all of which genuinely exist at the
      discharge decision landmark!
    """
    csv_path = cache_dir / "tier2_diabetes130.csv"
    df_raw = pd.read_csv(csv_path, low_memory=False).replace("?", np.nan)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    n_raw = len(df_raw)

    disp_num = (
        pd.to_numeric(df_raw["discharge_disposition_id"], errors="coerce").fillna(-1).astype(int)
    )
    inelig = disp_num.isin({11, 13, 14, 19, 20, 21}).to_numpy(dtype=bool)
    raw_pos = (df_raw["readmitted"].astype(str).str.strip().str.upper() == "<30").to_numpy(
        dtype=bool
    )
    n_excl = int(inelig.sum())
    n_excl_pos = int((inelig & raw_pos).sum())

    df = df_raw.loc[~inelig].reset_index(drop=True)
    if max_patients is not None:
        uniq_p = df["patient_nbr"].unique()
        if len(uniq_p) > max_patients:
            rng_p = np.random.default_rng(seed)
            keep_p = set(rng_p.choice(uniq_p, size=max_patients, replace=False).tolist())
            df = df[df["patient_nbr"].isin(keep_p)].reset_index(drop=True)

    missingness_rates = {
        "race": float(df["race"].isna().mean()),
        "payer_code": float(df["payer_code"].isna().mean()),
        "weight": float(df["weight"].isna().mean()),
        "medical_specialty": float(df["medical_specialty"].isna().mean()),
        "diag_1": float(df["diag_1"].isna().mean()),
        "diag_2": float(df["diag_2"].isna().mean()),
        "diag_3": float(df["diag_3"].isna().mean()),
    }

    y = (df["readmitted"].astype(str).str.strip().str.upper() == "<30").to_numpy(dtype=int)
    groups = df["patient_nbr"].to_numpy()
    race_clean = df["race"].fillna("Missing").astype(str).to_numpy()
    gender_clean = df["gender"].fillna("Missing").astype(str).to_numpy()

    base_cols_dict: dict[str, Any] = {
        "gender": df["gender"].fillna("__MISSING__").astype(str),
        "age_bucket": df["age"].fillna("__MISSING__").astype(str),
        "admission_type_id": df["admission_type_id"].astype(str),
        "admission_source_id": df["admission_source_id"].astype(str),
        "discharge_disposition_id": df["discharge_disposition_id"].astype(str),
        "payer_code": df["payer_code"].fillna("__MISSING__").astype(str),
        "payer_missing_flag": df["payer_code"].isna().astype(np.float32),
        "weight_recorded_flag": (~df["weight"].isna()).astype(np.float32),
        "time_in_hospital": pd.to_numeric(df["time_in_hospital"], errors="coerce").astype(
            np.float32
        ),
        "num_lab_procedures": pd.to_numeric(df["num_lab_procedures"], errors="coerce").astype(
            np.float32
        ),
        "num_procedures": pd.to_numeric(df["num_procedures"], errors="coerce").astype(np.float32),
        "num_medications": pd.to_numeric(df["num_medications"], errors="coerce").astype(np.float32),
        "number_diagnoses": pd.to_numeric(df["number_diagnoses"], errors="coerce").astype(
            np.float32
        ),
    }
    sem_types: dict[str, SemanticType] = {
        "gender": SemanticType.NOMINAL,
        "age_bucket": SemanticType.ORDINAL,
        "admission_type_id": SemanticType.NOMINAL,
        "admission_source_id": SemanticType.NOMINAL,
        "discharge_disposition_id": SemanticType.NOMINAL,
        "payer_code": SemanticType.NOMINAL,
        "payer_missing_flag": SemanticType.BINARY,
        "weight_recorded_flag": SemanticType.BINARY,
        "time_in_hospital": SemanticType.COUNT,
        "num_lab_procedures": SemanticType.COUNT,
        "num_procedures": SemanticType.COUNT,
        "num_medications": SemanticType.COUNT,
        "number_diagnoses": SemanticType.COUNT,
    }
    if include_race_in_base:
        base_cols_dict["race"] = df["race"].fillna("__MISSING__").astype(str)
        sem_types["race"] = SemanticType.NOMINAL

    X_base = pd.DataFrame(base_cols_dict)

    z_dict: dict[str, Any] = {
        "number_inpatient": pd.to_numeric(df["number_inpatient"], errors="coerce").astype(
            np.float32
        ),
        "inpatient_any_flag": (
            pd.to_numeric(df["number_inpatient"], errors="coerce").fillna(0) > 0
        ).astype(np.float32),
        "number_emergency": pd.to_numeric(df["number_emergency"], errors="coerce").astype(
            np.float32
        ),
        "emergency_any_flag": (
            pd.to_numeric(df["number_emergency"], errors="coerce").fillna(0) > 0
        ).astype(np.float32),
        "number_outpatient": pd.to_numeric(df["number_outpatient"], errors="coerce").astype(
            np.float32
        ),
        "diag_1_organ": _icd9_organ_grouper(df["diag_1"]),
        "diag_1_prefix": df["diag_1"].fillna("__MISSING__").astype(str).str.slice(0, 3),
        "diag_2_organ": _icd9_organ_grouper(df["diag_2"]),
        "diag_2_prefix": df["diag_2"].fillna("__MISSING__").astype(str).str.slice(0, 3),
        "diag_3_organ": _icd9_organ_grouper(df["diag_3"]),
        "diag_3_prefix": df["diag_3"].fillna("__MISSING__").astype(str).str.slice(0, 3),
        "medical_specialty": df["medical_specialty"].fillna("__MISSING__").astype(str),
        "A1Cresult": df["A1Cresult"].fillna("None").astype(str),
        "max_glu_serum": df["max_glu_serum"].fillna("None").astype(str),
    }

    med_cols = [
        "insulin",
        "change",
        "diabetesMed",
        "metformin",
        "glyburide-metformin",
        "glipizide-metformin",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
        "glipizide",
        "glyburide",
        "glimepiride",
        "chlorpropamide",
        "tolazamide",
        "tolbutamide",
        "acetohexamide",
        "pioglitazone",
        "rosiglitazone",
        "troglitazone",
        "glimepiride-pioglitazone",
        "repaglinide",
        "nateglinide",
        "acarbose",
        "miglitol",
    ]
    for mc in med_cols:
        z_dict[mc] = df[mc].fillna("No").astype(str)

    candidates = [
        CandidateChannelSpec(
            "hie_prior_inpatient", ("number_inpatient", "inpatient_any_flag"), 1.2, "external_hie"
        ),
        CandidateChannelSpec(
            "hie_prior_emergency", ("number_emergency", "emergency_any_flag"), 1.1, "external_hie"
        ),
        CandidateChannelSpec("hie_prior_outpatient", ("number_outpatient",), 1.0, "external_hie"),
        CandidateChannelSpec(
            "icd_primary_grouper", ("diag_1_organ", "diag_1_prefix"), 1.0, "clinical_coding"
        ),
        CandidateChannelSpec(
            "icd_secondary_grouper", ("diag_2_organ", "diag_2_prefix"), 1.0, "clinical_coding"
        ),
        CandidateChannelSpec(
            "icd_tertiary_grouper", ("diag_3_organ", "diag_3_prefix"), 1.0, "clinical_coding"
        ),
        CandidateChannelSpec(
            "attending_specialty", ("medical_specialty",), 0.8, "specialty_registry"
        ),
        CandidateChannelSpec("lab_hba1c_feed", ("A1Cresult",), 0.9, "lab_interface"),
        CandidateChannelSpec("lab_glucose_feed", ("max_glu_serum",), 0.9, "lab_interface"),
        CandidateChannelSpec(
            "rx_insulin_titration", ("insulin", "change", "diabetesMed"), 1.0, "pharmacy_rec"
        ),
        CandidateChannelSpec(
            "rx_biguanide_metformin",
            (
                "metformin",
                "glyburide-metformin",
                "glipizide-metformin",
                "metformin-rosiglitazone",
                "metformin-pioglitazone",
            ),
            0.8,
            "pharmacy_rec",
        ),
        CandidateChannelSpec(
            "rx_sulfonylureas_gen2", ("glipizide", "glyburide", "glimepiride"), 0.8, "pharmacy_rec"
        ),
        CandidateChannelSpec(
            "rx_sulfonylureas_gen1",
            ("chlorpropamide", "tolazamide", "tolbutamide", "acetohexamide"),
            0.8,
            "pharmacy_rec",
        ),
        CandidateChannelSpec(
            "rx_thiazolidinediones",
            ("pioglitazone", "rosiglitazone", "troglitazone", "glimepiride-pioglitazone"),
            0.8,
            "pharmacy_rec",
        ),
        CandidateChannelSpec(
            "rx_meglitinides", ("repaglinide", "nateglinide"), 0.8, "pharmacy_rec"
        ),
        CandidateChannelSpec("rx_alpha_glucosidase", ("acarbose", "miglitol"), 0.8, "pharmacy_rec"),
    ]

    if include_stress_channels:
        _append_stress_channels(
            X_base,
            z_dict,
            candidates,
            twin_source_ids=("hie_prior_inpatient", "hie_prior_emergency", "icd_primary_grouper"),
            seed=seed,
        )

    fixed_interface_costs = {
        c.channel_id: (150.0 if "hie" in c.channel_id else 60.0) for c in candidates
    }

    return FineGrainedCohort(
        name="UCI_Diabetes_Discharge",
        display_name="UCI Diabetes 130-US (Discharge Landmark)",
        domain="Clinical Hospital Readmission",
        provenance="real_observational",
        landmark_description=(
            "Hospital discharge landmark on 99,343 discharged-alive non-hospice stays across "
            "69,984 patients; X_base holds 13 local encounter/stay summary fields, and M=16 "
            "candidate channels represent external HIE, coding, specialty, lab, and Rx feeds."
        ),
        X_base=X_base,
        Z_pool=pd.DataFrame(z_dict),
        candidates=candidates,
        y=y,
        contract=DecisionContract(
            task_name="Diabetes_Discharge_Readmission_30d",
            target_precision=0.30,
            max_fpr_budget=0.05,
            cost_fp=1.0,
            cost_fn=8.0,
            graded_action_cost_fp=0.20,
        ),
        acquisition_budget=4.5,
        per_instance_query_cost=0.04,
        base_semantic_types=sem_types,
        group_ids=groups,
        fairness_subgroups={"race": race_clean, "gender": gender_clean},
        missingness_rates=missingness_rates,
        channel_Fixed_interface_cost=fixed_interface_costs,
        eligibility_audit={
            "n_raw_encounters": n_raw,
            "n_excluded_death_or_hospice": n_excl,
            "n_excluded_recorded_lt30": n_excl_pos,
            "n_eligible_encounters": int(len(df)),
            "n_unique_patients": int(pd.Series(groups).nunique()),
            "prevalence": float(np.mean(y)),
        },
    )


def load_fine_home_credit(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    include_stress_channels: bool = False,
    max_rows: int | None = None,
    seed: int = 42,
) -> FineGrainedCohort:
    """Home Credit Default Risk across 7 relational tables partitioned into M=26 native channels."""
    pkl_path = _prepare_home_credit_7table_cache(cache_dir)
    df = pd.read_pickle(pkl_path)
    if max_rows is not None and len(df) > max_rows:
        rng_s = np.random.default_rng(seed)
        keep_i = np.sort(rng_s.choice(len(df), size=max_rows, replace=False))
        df = df.iloc[keep_i].reset_index(drop=True)

    y = df["TARGET"].to_numpy(dtype=int)

    base_cat = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
    ]
    base_num = [
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
    ]
    X_base = pd.DataFrame(
        {
            **{c: df[c].fillna("__MISSING__").astype(str) for c in base_cat},
            **{c: pd.to_numeric(df[c], errors="coerce").astype(np.float32) for c in base_num},
        }
    )
    sem_types = {
        **{c: SemanticType.NOMINAL for c in base_cat},
        **{c: SemanticType.CONTINUOUS for c in base_num},
    }

    channel_defs: list[tuple[str, list[str], float, str]] = [
        ("ext_bureau_vendor_1", ["EXT_SOURCE_1"], 1.2, "external_score_api"),
        ("ext_bureau_vendor_2", ["EXT_SOURCE_2"], 1.0, "external_score_api"),
        ("ext_bureau_vendor_3", ["EXT_SOURCE_3"], 1.0, "external_score_api"),
        (
            "bur_active_exposure",
            ["bur_count", "bur_active_frac", "bur_credit_sum"],
            1.0,
            "bureau_table",
        ),
        ("bur_debt_leverage", ["bur_debt_sum", "bur_debt_to_credit_ratio"], 1.0, "bureau_table"),
        (
            "bur_overdue_severity",
            ["bur_max_overdue", "bur_overdue_sum", "bur_bb_dpd_months"],
            1.0,
            "bureau_table",
        ),
        (
            "bur_credit_recency",
            ["bur_days_credit_max", "bur_days_credit_mean"],
            0.9,
            "bureau_table",
        ),
        (
            "bur_inquiry_cadence",
            [c for c in df.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")],
            0.8,
            "bureau_inquiries",
        ),
        (
            "prev_refusal_history",
            ["prev_refused_frac", "prev_refused_count"],
            1.0,
            "previous_app_table",
        ),
        (
            "prev_approval_terms",
            ["prev_approved_frac", "prev_cred_to_app_mean"],
            1.0,
            "previous_app_table",
        ),
        (
            "prev_app_velocity",
            ["prev_app_count", "prev_days_decision_max"],
            0.9,
            "previous_app_table",
        ),
        (
            "inst_dpd_delinquency",
            ["inst_dpd_mean", "inst_dpd_max", "inst_underpaid_frac"],
            1.0,
            "installments_table",
        ),
        (
            "inst_payment_volume",
            ["inst_count", "inst_pay_ratio_mean", "inst_amt_due_sum", "inst_amt_paid_sum"],
            1.0,
            "installments_table",
        ),
        (
            "inst_traj_early_9m",
            ["inst_pay_t0", "inst_pay_t1", "inst_pay_t2"],
            0.9,
            "installments_table",
        ),
        (
            "inst_traj_recent_9m",
            ["inst_pay_t3", "inst_pay_t4", "inst_pay_t5"],
            0.9,
            "installments_table",
        ),
        (
            "cc_utilization_summary",
            ["cc_months", "cc_util_mean", "cc_util_max"],
            1.0,
            "credit_card_table",
        ),
        (
            "cc_drawings_and_dpd",
            ["cc_drawings_sum", "cc_payments_sum", "cc_dpd_max"],
            1.0,
            "credit_card_table",
        ),
        ("cc_util_traj_6m", [f"cc_util_t{t}" for t in range(6)], 0.9, "credit_card_table"),
        (
            "pos_cash_delinquency",
            ["pos_months", "pos_future_inst_min", "pos_dpd_max", "pos_dpd_def_mean"],
            1.0,
            "pos_cash_table",
        ),
        (
            "employer_occupation",
            ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"],
            0.8,
            "employment_registry",
        ),
        (
            "region_ratings",
            ["REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"],
            0.8,
            "geo_registry",
        ),
        (
            "address_mismatch_check",
            [
                "REG_REGION_NOT_LIVE_REGION",
                "REG_REGION_NOT_WORK_REGION",
                "LIVE_REGION_NOT_WORK_REGION",
                "REG_CITY_NOT_LIVE_CITY",
                "REG_CITY_NOT_WORK_CITY",
                "LIVE_CITY_NOT_WORK_CITY",
            ],
            0.8,
            "address_verification",
        ),
        (
            "social_circle_default",
            [
                "OBS_30_CNT_SOCIAL_CIRCLE",
                "DEF_30_CNT_SOCIAL_CIRCLE",
                "OBS_60_CNT_SOCIAL_CIRCLE",
                "DEF_60_CNT_SOCIAL_CIRCLE",
            ],
            0.9,
            "social_network",
        ),
        (
            "bldg_physical_avg",
            [
                "APARTMENTS_AVG",
                "BASEMENTAREA_AVG",
                "YEARS_BEGINEXPLUATATION_AVG",
                "YEARS_BUILD_AVG",
                "ELEVATORS_AVG",
                "ENTRANCES_AVG",
                "FLOORSMAX_AVG",
                "LIVINGAREA_AVG",
            ],
            0.8,
            "property_inspection",
        ),
        (
            "contact_phone_flags",
            [
                "FLAG_MOBIL",
                "FLAG_EMP_PHONE",
                "FLAG_WORK_PHONE",
                "FLAG_CONT_MOBILE",
                "FLAG_PHONE",
                "FLAG_EMAIL",
                "DAYS_LAST_PHONE_CHANGE",
            ],
            0.8,
            "telecom_check",
        ),
        ("kyc_document_flags", [f"FLAG_DOCUMENT_{i}" for i in range(2, 12)], 0.7, "kyc_documents"),
    ]

    z_dict: dict[str, Any] = {}
    candidates: list[CandidateChannelSpec] = []
    for cid, cols, cost, tag in channel_defs:
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue
        for c in valid_cols:
            if df[c].dtype == object or str(df[c].dtype) == "category":
                z_dict[c] = df[c].fillna("__MISSING__").astype(str)
            else:
                z_dict[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        candidates.append(CandidateChannelSpec(cid, tuple(valid_cols), cost, tag))

    if include_stress_channels:
        _append_stress_channels(
            X_base,
            z_dict,
            candidates,
            twin_source_ids=("ext_bureau_vendor_2", "ext_bureau_vendor_3", "inst_dpd_delinquency"),
            seed=seed,
        )

    return FineGrainedCohort(
        name="Home_Credit_7Table",
        display_name="Home Credit Default Risk (7 Tables, M=26)",
        domain="Consumer Lending Underwriting",
        provenance="real_observational",
        landmark_description=(
            "Loan application underwriting across 307,511 borrowers; X_base holds 18 submitted "
            "application fields, and M=26 channels span 3 external score APIs, 6 relational "
            "history tables (20.8M child records), property inspection, and KYC checks."
        ),
        X_base=X_base,
        Z_pool=pd.DataFrame(z_dict),
        candidates=candidates,
        y=y,
        contract=DecisionContract(
            task_name="HomeCredit_Underwriting",
            target_precision=0.32,
            max_fpr_budget=0.03,
            cost_fp=1.0,
            cost_fn=12.0,
            graded_action_cost_fp=0.20,
        ),
        acquisition_budget=5.0,
        per_instance_query_cost=0.04,
        base_semantic_types=sem_types,
        fairness_subgroups={"gender": df["CODE_GENDER"].fillna("Missing").astype(str).to_numpy()},
    )


def load_fine_ieee_cis(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    window_idx: int | None = None,
    n_windows: int = 20,
    max_window_rows: int = 150000,
    include_stress_channels: bool = False,
    seed: int = 42,
) -> FineGrainedCohort:
    """IEEE-CIS Fraud Detection partitioned into M=25 transaction & identity channels."""
    pkl_path = _prepare_ieee_cis_cache(cache_dir)
    df_all = pd.read_pickle(pkl_path)
    df = _staggered_chrono_window(df_all, window_idx, n_windows, max_window_rows=max_window_rows)

    y = df["isFraud"].to_numpy(dtype=int)
    time_arr = df["TransactionDT"].to_numpy(dtype=np.float64)

    X_base = pd.DataFrame(
        {
            "TransactionAmt": df["TransactionAmt"].astype(np.float32),
            "ProductCD": df["ProductCD"].astype(str),
            "card1": df["card1"].astype(np.float32),
            "card2": df["card2"].astype(np.float32),
            "card3": df["card3"].astype(np.float32),
            "card4": df["card4"].astype(str),
            "card5": df["card5"].astype(np.float32),
            "card6": df["card6"].astype(str),
            "addr1": df["addr1"].astype(np.float32),
            "addr2": df["addr2"].astype(np.float32),
            "P_emaildomain": df["P_emaildomain"].astype(str),
        }
    )
    sem_types = {
        "TransactionAmt": SemanticType.CONTINUOUS,
        "ProductCD": SemanticType.NOMINAL,
        "card1": SemanticType.CONTINUOUS,
        "card2": SemanticType.CONTINUOUS,
        "card3": SemanticType.CONTINUOUS,
        "card4": SemanticType.NOMINAL,
        "card5": SemanticType.CONTINUOUS,
        "card6": SemanticType.NOMINAL,
        "addr1": SemanticType.CONTINUOUS,
        "addr2": SemanticType.CONTINUOUS,
        "P_emaildomain": SemanticType.NOMINAL,
    }

    v_all = [c for c in df.columns if c.startswith("V")]
    v_chunks = np.array_split(v_all, 10)

    channel_defs: list[tuple[str, list[str], float, str]] = [
        ("cnt_card_links_C1_4", [f"C{i}" for i in range(1, 5)], 1.0, "entity_graph"),
        ("cnt_addr_links_C5_8", [f"C{i}" for i in range(5, 9)], 1.0, "entity_graph"),
        ("cnt_email_links_C9_11", [f"C{i}" for i in range(9, 12)], 1.0, "entity_graph"),
        ("cnt_device_links_C12_14", [f"C{i}" for i in range(12, 15)], 1.0, "entity_graph"),
        ("delta_card_age_D1_4", [f"D{i}" for i in range(1, 5)], 1.0, "temporal_deltas"),
        ("delta_email_recency_D5_9", [f"D{i}" for i in range(5, 10)], 1.0, "temporal_deltas"),
        ("delta_addr_tenure_D10_15", [f"D{i}" for i in range(10, 16)], 1.0, "temporal_deltas"),
        ("avs_match_M1_5", [f"M{i}" for i in range(1, 6)], 0.8, "avs_rules"),
        (
            "routing_dist_recipient",
            [f"M{i}" for i in range(6, 10)] + ["dist1", "dist2", "R_emaildomain"],
            0.9,
            "routing_geo",
        ),
    ]
    for b_idx, v_grp in enumerate(v_chunks, start=1):
        if len(v_grp) > 0:
            channel_defs.append(
                (f"vesta_block_{b_idx:02d}", [str(x) for x in v_grp], 1.0, "vesta_risk_engine")
            )

    channel_defs.extend(
        [
            ("id_numeric_scores_01_06", [f"id_0{i}" for i in range(1, 7)], 1.1, "identity_table"),
            (
                "id_numeric_scores_07_11",
                [f"id_{i:02d}" for i in range(7, 12)],
                1.1,
                "identity_table",
            ),
            ("id_network_proxy_12_20", [f"id_{i}" for i in range(12, 21)], 1.1, "identity_table"),
            ("id_ip_asn_21_29", [f"id_{i}" for i in range(21, 30)], 1.1, "identity_table"),
            ("id_browser_os_30_38", [f"id_{i}" for i in range(30, 39)], 1.1, "identity_table"),
            (
                "id_device_hardware",
                ["DeviceType", "DeviceInfo", "__has_id_receipt__"],
                1.0,
                "identity_table",
            ),
        ]
    )

    z_dict: dict[str, Any] = {}
    candidates: list[CandidateChannelSpec] = []
    for cid, cols, cost, tag in channel_defs:
        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            continue
        for c in valid_cols:
            if str(df[c].dtype) == "category" or df[c].dtype == object:
                z_dict[c] = df[c].astype(str)
            else:
                z_dict[c] = df[c].astype(np.float32)
        candidates.append(CandidateChannelSpec(cid, tuple(valid_cols), cost, tag))

    if include_stress_channels:
        _append_stress_channels(
            X_base,
            z_dict,
            candidates,
            twin_source_ids=("cnt_card_links_C1_4", "cnt_device_links_C12_14", "vesta_block_01"),
            seed=seed,
        )

    return FineGrainedCohort(
        name="IEEE_CIS_Fraud",
        display_name="IEEE-CIS Payment Fraud (2 Tables, M=25)",
        domain="Real-Time Payment Authorization",
        provenance="real_observational",
        landmark_description=(
            "Chronological payment authorization stream (590,540 transactions); X_base holds 11 "
            "authorization header fields, and M=25 channels span entity graphs, temporal deltas, "
            "10 Vesta feature blocks, and 6 Identity device/network tables."
        ),
        X_base=X_base,
        Z_pool=pd.DataFrame(z_dict),
        candidates=candidates,
        y=y,
        contract=DecisionContract(
            task_name="IEEE_CIS_Realtime_Auth",
            target_precision=0.60,
            max_fpr_budget=0.005,
            cost_fp=1.0,
            cost_fn=25.0,
            graded_action_cost_fp=0.15,
        ),
        acquisition_budget=5.0,
        per_instance_query_cost=0.04,
        base_semantic_types=sem_types,
        time_order=time_arr,
    )


def load_fine_scania_aps(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    include_stress_channels: bool = False,
    seed: int = 42,
) -> FineGrainedCohort:
    """Scania APS Heavy Truck Fleet (76,000 trucks, 170 sensors -> M=22 subsystem channels)."""
    csv_path = cache_dir / "tier2_scania_aps.csv"
    df = pd.read_csv(csv_path, low_memory=False).replace("na", np.nan)
    raw_cls = df.pop("class").astype(str).str.strip().str.lower().str.strip("'")
    y = (raw_cls == "pos").to_numpy(dtype=int)

    num_df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    base_cols = ["aa_000", "ab_000", "ac_000", "ad_000"]
    X_base = num_df[base_cols].copy()
    X_base["missing_sensor_count"] = num_df.isna().sum(axis=1).to_numpy(dtype=np.float32)
    sem_types = {c: SemanticType.CONTINUOUS for c in X_base.columns}

    hist_prefixes = ("ag", "ay", "az", "ba", "cn", "cs", "ee")
    candidates: list[CandidateChannelSpec] = []
    used = set(base_cols)
    z_dict: dict[str, Any] = {}

    for hp in hist_prefixes:
        h_cols = [c for c in num_df.columns if c.startswith(f"{hp}_")]
        used.update(h_cols)
        for c in h_cols:
            z_dict[c] = num_df[c]
        candidates.append(
            CandidateChannelSpec(f"hist_sensor_{hp}", tuple(h_cols), 1.0, "histogram_sensor")
        )

    rem_cols = [c for c in num_df.columns if c not in used]
    rem_groups = np.array_split(rem_cols, 15)
    for g_idx, grp in enumerate(rem_groups, start=1):
        g_list = [str(x) for x in grp]
        if not g_list:
            continue
        for c in g_list:
            z_dict[c] = num_df[c]
        p_lbl = f"{g_list[0].split('_')[0]}_{g_list[-1].split('_')[0]}"
        candidates.append(
            CandidateChannelSpec(
                f"counter_mod_{g_idx:02d}_{p_lbl}", tuple(g_list), 1.0, "counter_module"
            )
        )

    if include_stress_channels:
        _append_stress_channels(
            X_base,
            z_dict,
            candidates,
            twin_source_ids=("hist_sensor_ag", "hist_sensor_ee", "hist_sensor_ba"),
            seed=seed,
        )

    return FineGrainedCohort(
        name="Scania_APS_Fleet",
        display_name="Scania APS Heavy Trucks (170 Sensors, M=22)",
        domain="Industrial Fleet Maintenance",
        provenance="real_observational",
        landmark_description=(
            "76,000 heavy trucks with official 50:1 failure-vs-inspection cost contract; X_base "
            "holds 5 primary odometer/load counters, and M=22 candidate channels span 7 10-bin "
            "operating-range histograms and 15 telemetry counter modules."
        ),
        X_base=X_base,
        Z_pool=pd.DataFrame(z_dict),
        candidates=candidates,
        y=y,
        contract=DecisionContract(
            task_name="Scania_50to1_Maintenance",
            target_precision=0.50,
            max_fpr_budget=0.01,
            cost_fp=1.0,
            cost_fn=50.0,
            graded_action_cost_fp=0.15,
        ),
        acquisition_budget=4.0,
        per_instance_query_cost=0.08,
        base_semantic_types=sem_types,
    )


def load_fine_neurips_baf(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    window_idx: int | None = None,
    n_windows: int = 20,
    max_window_rows: int = 160000,
    include_stress_channels: bool = False,
    seed: int = 42,
) -> FineGrainedCohort:
    """NeurIPS 2022 Bank Account Fraud (1M rows, chronological eras, M=18 native channels)."""
    csv_path = cache_dir / "kaggle" / "BAF" / "Base.csv"
    df_all = pd.read_csv(csv_path, low_memory=False)
    df_all["__seq_order__"] = df_all["month"].to_numpy(dtype=np.float64) * 1e7 + np.arange(
        len(df_all), dtype=np.float64
    )
    df_all = df_all.sort_values("__seq_order__", kind="mergesort").reset_index(drop=True)
    df = _staggered_chrono_window(df_all, window_idx, n_windows, max_window_rows=max_window_rows)

    y = pd.to_numeric(df["fraud_bool"], errors="coerce").fillna(0).to_numpy(dtype=int)
    time_arr = df["__seq_order__"].to_numpy(dtype=np.float64)

    for neg_col in (
        "prev_address_months_count",
        "current_address_months_count",
        "bank_months_count",
        "session_length_in_minutes",
        "intended_balcon_amount",
    ):
        if neg_col in df.columns:
            v = pd.to_numeric(df[neg_col], errors="coerce")
            df[neg_col] = v.where(v >= 0, np.nan).astype(np.float32)

    base_num = [
        "income",
        "customer_age",
        "proposed_credit_limit",
        "intended_balcon_amount",
        "days_since_request",
    ]
    base_cat = ["employment_status", "housing_status", "payment_type", "source"]
    X_base = pd.DataFrame(
        {
            **{c: pd.to_numeric(df[c], errors="coerce").astype(np.float32) for c in base_num},
            **{c: df[c].astype(str) for c in base_cat},
        }
    )
    sem_types = {
        **{c: SemanticType.CONTINUOUS for c in base_num},
        **{c: SemanticType.NOMINAL for c in base_cat},
    }

    channel_defs: list[tuple[str, list[str], float, str]] = [
        ("credit_bureau_score", ["credit_risk_score"], 1.2, "credit_bureau"),
        ("other_cards_registry", ["has_other_cards"], 0.9, "credit_bureau"),
        ("bank_tenure_history", ["bank_months_count"], 0.9, "internal_history"),
        ("prev_address_tenure", ["prev_address_months_count"], 0.9, "address_bureau"),
        ("curr_address_tenure", ["current_address_months_count"], 0.9, "address_bureau"),
        ("velocity_6h_stream", ["velocity_6h"], 1.0, "consortium_velocity"),
        ("velocity_24h_stream", ["velocity_24h"], 1.0, "consortium_velocity"),
        ("velocity_4w_stream", ["velocity_4w"], 1.0, "consortium_velocity"),
        ("zip_density_4w", ["zip_count_4w"], 0.9, "geo_velocity"),
        ("branch_density_8w", ["bank_branch_count_8w"], 0.9, "branch_velocity"),
        ("dob_email_graph_4w", ["date_of_birth_distinct_emails_4w"], 1.0, "identity_ring"),
        ("name_email_lexical", ["name_email_similarity"], 0.9, "email_intelligence"),
        ("email_domain_provider", ["email_is_free"], 0.8, "email_intelligence"),
        ("landline_phone_verify", ["phone_home_valid"], 0.8, "telecom_verify"),
        ("mobile_phone_verify", ["phone_mobile_valid"], 0.8, "telecom_verify"),
        (
            "session_behavior",
            ["session_length_in_minutes", "keep_alive_session"],
            0.9,
            "session_telemetry",
        ),
        ("device_email_graph_8w", ["device_distinct_emails_8w"], 1.1, "device_fingerprint"),
        ("device_os_foreign_ip", ["device_os", "foreign_request"], 1.0, "device_fingerprint"),
    ]

    z_dict: dict[str, Any] = {}
    candidates: list[CandidateChannelSpec] = []
    for cid, cols, cost, tag in channel_defs:
        for c in cols:
            if c == "device_os" or df[c].dtype == object:
                z_dict[c] = df[c].astype(str)
            else:
                z_dict[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        candidates.append(CandidateChannelSpec(cid, tuple(cols), cost, tag))

    if include_stress_channels:
        _append_stress_channels(
            X_base,
            z_dict,
            candidates,
            twin_source_ids=(
                "credit_bureau_score",
                "device_email_graph_8w",
                "other_cards_registry",
            ),
            seed=seed,
        )

    age_grp = np.where(df["customer_age"].to_numpy(dtype=float) >= 50.0, "Age>=50", "Age<50")

    return FineGrainedCohort(
        name="NeurIPS_BAF",
        display_name="NeurIPS Bank Account Fraud (1M, M=18)",
        domain="Digital Banking Onboarding",
        provenance="privacy_synthetic_from_real",
        landmark_description=(
            "Chronological bank onboarding stream (1,000,000 applications, CTGAN-synthesized "
            "from real bank logs by Jesus et al., 2022); X_base holds 9 applicant form fields, "
            "and M=18 channels represent external bureau, telecom, email, velocity, & device APIs."
        ),
        X_base=X_base,
        Z_pool=pd.DataFrame(z_dict),
        candidates=candidates,
        y=y,
        contract=DecisionContract(
            task_name="BAF_Onboarding_Screen",
            target_precision=0.25,
            max_fpr_budget=0.01,
            cost_fp=1.0,
            cost_fn=30.0,
            graded_action_cost_fp=0.15,
        ),
        acquisition_budget=4.5,
        per_instance_query_cost=0.03,
        base_semantic_types=sem_types,
        time_order=time_arr,
        fairness_subgroups={"age_50plus": age_grp},
    )


def load_fine_taiwan_credit(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    restrict_baseline_current: bool = False,
    include_stress_channels: bool = False,
    seed: int = 42,
) -> FineGrainedCohort:
    """UCI Taiwan Credit 6-Month Panel (full N=30,000 by default; optional PAY_6<=0 sensitivity)."""
    csv_path = cache_dir / "tier2_taiwan_credit.csv"
    df_raw = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df_raw)

    # OpenML 42477 maps x1=LIMIT_BAL, x2=SEX, x3=EDUCATION, x4=MARRIAGE, x5=AGE,
    # x6..x11=PAY_0..PAY_6 (m1..m6), x12..x17=BILL_AMT1..6, x18..x23=PAY_AMT1..6
    pay6 = pd.to_numeric(df_raw["x11"], errors="coerce").fillna(0)
    raw_y = pd.to_numeric(df_raw["y"], errors="coerce").fillna(0).to_numpy(dtype=int)
    if restrict_baseline_current:
        inelig = (pay6 > 0).to_numpy(dtype=bool)
        df = df_raw.loc[~inelig].reset_index(drop=True)
        excl_n = int(inelig.sum())
        excl_pos = int((inelig & (raw_y == 1)).sum())
    else:
        df = df_raw.copy()
        excl_n = 0
        excl_pos = 0

    y = pd.to_numeric(df["y"], errors="coerce").fillna(0).to_numpy(dtype=int)
    X_base = pd.DataFrame(
        {
            "LIMIT_BAL": pd.to_numeric(df["x1"], errors="coerce").astype(np.float32),
            "SEX": df["x2"].astype(str),
            "EDUCATION": df["x3"].astype(str),
            "MARRIAGE": df["x4"].astype(str),
            "AGE": pd.to_numeric(df["x5"], errors="coerce").astype(np.float32),
            "PAY_M6_BASE": pd.to_numeric(df["x11"], errors="coerce").astype(np.float32),
            "BILL_AMT_M6_BASE": pd.to_numeric(df["x17"], errors="coerce").astype(np.float32),
            "PAY_AMT_M6_BASE": pd.to_numeric(df["x23"], errors="coerce").astype(np.float32),
        }
    )
    sem_types = {
        "LIMIT_BAL": SemanticType.CONTINUOUS,
        "SEX": SemanticType.NOMINAL,
        "EDUCATION": SemanticType.NOMINAL,
        "MARRIAGE": SemanticType.NOMINAL,
        "AGE": SemanticType.COUNT,
        "PAY_M6_BASE": SemanticType.ORDINAL,
        "BILL_AMT_M6_BASE": SemanticType.CONTINUOUS,
        "PAY_AMT_M6_BASE": SemanticType.CONTINUOUS,
    }

    z_dict: dict[str, Any] = {}
    candidates: list[CandidateChannelSpec] = []
    # 5 monthly repayment channels (m5..m1), 5 bill channels, 5 payment channels = 15 channels
    for idx_m, m_lbl in enumerate(("m5", "m4", "m3", "m2", "m1")):
        pay_col = f"x{10 - idx_m}"
        bill_col = f"x{16 - idx_m}"
        amt_col = f"x{22 - idx_m}"
        z_dict[f"repay_{m_lbl}"] = pd.to_numeric(df[pay_col], errors="coerce").astype(np.float32)
        z_dict[f"bill_{m_lbl}"] = pd.to_numeric(df[bill_col], errors="coerce").astype(np.float32)
        z_dict[f"payamt_{m_lbl}"] = pd.to_numeric(df[amt_col], errors="coerce").astype(np.float32)
        candidates.append(
            CandidateChannelSpec(
                f"repay_status_{m_lbl}", (f"repay_{m_lbl}",), 1.0, "repayment_ledger"
            )
        )
        candidates.append(
            CandidateChannelSpec(
                f"statement_bill_{m_lbl}", (f"bill_{m_lbl}",), 0.8, "statement_ledger"
            )
        )
        candidates.append(
            CandidateChannelSpec(
                f"cash_payment_{m_lbl}", (f"payamt_{m_lbl}",), 0.8, "payment_ledger"
            )
        )

    if include_stress_channels:
        _append_stress_channels(
            X_base,
            z_dict,
            candidates,
            twin_source_ids=("repay_status_m1", "repay_status_m2", "cash_payment_m1"),
            seed=seed,
        )

    return FineGrainedCohort(
        name="UCI_Taiwan_Credit",
        display_name="UCI Taiwan Credit Panel (M=15)",
        domain="Revolving Credit Servicing",
        provenance="real_observational",
        landmark_description=(
            f"UCI Taiwan Credit ({len(df):,} accounts; "
            f"restrict_baseline_current={restrict_baseline_current}); "
            "X_base holds demographics, credit limit, and oldest month-6 billing snapshot; M=15 "
            "channels represent subsequent monthly repayment, statement, and payment ledgers."
        ),
        X_base=X_base,
        Z_pool=pd.DataFrame(z_dict),
        candidates=candidates,
        y=y,
        contract=DecisionContract(
            task_name="Taiwan_Credit_Default",
            target_precision=0.50,
            max_fpr_budget=0.05,
            cost_fp=1.0,
            cost_fn=5.0,
            graded_action_cost_fp=0.20,
        ),
        acquisition_budget=3.5,
        per_instance_query_cost=0.03,
        base_semantic_types=sem_types,
        eligibility_audit={
            "n_raw": n_raw,
            "restrict_baseline_current": restrict_baseline_current,
            "n_excluded": excl_n,
            "n_excluded_positives": excl_pos,
            "n_evaluated": int(len(df)),
            "prevalence": float(np.mean(y)),
        },
    )


__all__ = [
    "FineGrainedCohort",
    "TWIN_SOURCE_IDS_BY_COHORT",
    "load_fine_diabetes_discharge",
    "load_fine_home_credit",
    "load_fine_ieee_cis",
    "load_fine_neurips_baf",
    "load_fine_scania_aps",
    "load_fine_taiwan_credit",
    "materialize_leak_free_stress_pool",
]
