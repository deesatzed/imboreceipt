"""Wave-2 Flagship Multi-Table & High-Scale Public Cohorts (Scania APS, BAF, IEEE-CIS, Home Credit).

All loaders obey Senior-Professor inductive hygiene:
- Zero cross-row or cross-entity statistics are ever computed prior to ``freeze_split``.
- Relational aggregations in Home Credit (7 tables) and IEEE-CIS (2 tables) operate strictly
  within each independent entity's historical pre-application / pre-authorization records.
- Stateful trajectory, silence, and conservation compilers are stored as ``compiler_factories``
  on ``raw_aux_df`` and fitted strictly on ``train_idx`` inside ``materialize_z_splits``.
- Chronological streams (BAF 1M, IEEE-CIS 590k) partition multi-window campaigns into
  100% mutually disjoint out-of-time test blocks.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from imboreceipt.bench.datasets import DEFAULT_CACHE_DIR
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


def _disjoint_chrono_slice(
    df: pd.DataFrame,
    window_idx: int | None,
    n_windows: int,
) -> tuple[pd.DataFrame, bool]:
    """Slice a contiguous causal window whose final 15% test block is disjoint across windows."""
    if window_idx is None or n_windows <= 1:
        return df, False
    span_frac = min(0.52, 1.0 / (1.0 + 0.23 * float(n_windows - 1)))
    step_frac = (1.0 - span_frac) / float(n_windows - 1)
    start_f = step_frac * float(window_idx)
    lo_i = int(round(start_f * len(df)))
    hi_i = min(len(df), int(round((start_f + span_frac) * len(df))))
    return df.iloc[lo_i:hi_i].reset_index(drop=True), True


def load_wave2_scania_aps(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seed: int = 42,
    include_synthetic_proxies: bool = False,
) -> MultiChannelCohort:
    """Load Scania APS Failure at Heavy Trucks (76,000 trucks, 1.81% failure, 170 sensors).

    NOTE on feature semantics: Scania histogram groups (ag, ay, az, ba, cn, cs, ee) represent
    operating-range histogram bins rather than chronological time steps; summary dispersion and
    adjacent-bin gradient features capture histogram shape rather than temporal slowing down,
    and remaining anonymized counters carry no undocumented subsystem labels.
    """
    csv_path = cache_dir / "tier2_scania_aps.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run scripts/stage_wave2_datasets.py first.")

    df = pd.read_csv(csv_path, low_memory=False).replace("na", np.nan)
    raw_cls = df.pop("class").astype(str).str.strip().str.lower().str.strip("'")
    y_true = (raw_cls == "pos").to_numpy(dtype=int)

    num_df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    miss_count = num_df.isna().sum(axis=1).to_numpy(dtype=np.float32)

    base_cols = ["aa_000", "ab_000", "ac_000", "ad_000", "ae_000", "af_000"]
    X_base = num_df[base_cols].copy()
    X_base["telemetry_missing_count"] = miss_count

    hist_a_groups = {
        "ag": [f"ag_00{i}" for i in range(10)],
        "ay": [f"ay_00{i}" for i in range(10)],
        "az": [f"az_00{i}" for i in range(10)],
    }
    hist_b_groups = {
        "ba": [f"ba_00{i}" for i in range(10)],
        "cn": [f"cn_00{i}" for i in range(10)],
        "cs": [f"cs_00{i}" for i in range(10)],
        "ee": [f"ee_00{i}" for i in range(10)],
    }
    all_hist_cols = [
        c for grp in (hist_a_groups, hist_b_groups) for cols in grp.values() for c in cols
    ]
    used_set = set(base_cols) | set(all_hist_cols)
    rem_counters = [c for c in num_df.columns if c not in used_set]
    mid_c = len(rem_counters) // 2
    cnt_a_cols = rem_counters[:mid_c]
    cnt_b_cols = rem_counters[mid_c:]

    raw_aux = num_df[all_hist_cols].copy().reset_index(drop=True)
    compiler_factories: list[Callable[[], Any]] = [
        lambda: CriticalSlowingDownCompiler(hist_a_groups),
        lambda: CriticalSlowingDownCompiler(hist_b_groups),
    ]
    shape_a_cols = tuple(
        f"csd__{g}__{suf}"
        for g in hist_a_groups
        for suf in ("diff_base", "log_var_inflation", "ac1", "slope")
    )
    shape_b_cols = tuple(
        f"csd__{g}__{suf}"
        for g in hist_b_groups
        for suf in ("diff_base", "log_var_inflation", "ac1", "slope")
    )

    z_parts = [num_df[all_hist_cols + cnt_a_cols + cnt_b_cols].reset_index(drop=True)]
    hist_a_raw = tuple(c for cols in hist_a_groups.values() for c in cols)
    hist_b_raw = tuple(c for cols in hist_b_groups.values() for c in cols)

    candidates = [
        CandidateChannelSpec(
            "ch_hist_bins_A",
            hist_a_raw + shape_a_cols,
            acquisition_cost=1.0,
            mechanism_tag="histogram_bin_shape_A",
        ),
        CandidateChannelSpec(
            "ch_hist_bins_B",
            hist_b_raw + shape_b_cols,
            acquisition_cost=1.0,
            mechanism_tag="histogram_bin_shape_B",
        ),
        CandidateChannelSpec(
            "ch_anon_counters_A",
            tuple(cnt_a_cols),
            acquisition_cost=1.0,
            mechanism_tag="anonymized_counters_A",
        ),
        CandidateChannelSpec(
            "ch_anon_counters_B",
            tuple(cnt_b_cols),
            acquisition_cost=1.0,
            mechanism_tag="anonymized_counters_B",
        ),
    ]

    if include_synthetic_proxies:
        rng = np.random.default_rng(seed)
        aa_clean = np.nan_to_num(X_base["aa_000"].to_numpy(dtype=float), nan=0.0)
        z_parts.append(
            pd.DataFrame(
                {
                    "redundant_odometer_echo": (
                        aa_clean * 1.015 + rng.normal(0.0, 2200.0, len(df))
                    ).astype(np.float32),
                    "redundant_log_load_echo": (
                        np.log1p(np.maximum(0.0, aa_clean)) + rng.normal(0.0, 0.18, len(df))
                    ).astype(np.float32),
                    "decoy_ambient_temp": rng.normal(12.0, 6.0, len(df)).astype(np.float32),
                    "decoy_depot_code": rng.integers(1, 12, len(df)).astype(str),
                }
            )
        )
        candidates.extend(
            [
                CandidateChannelSpec(
                    "ch_redundant_odometer_proxy",
                    ("redundant_odometer_echo", "redundant_log_load_echo"),
                    acquisition_cost=1.0,
                    mechanism_tag="redundant_proxy",
                ),
                CandidateChannelSpec(
                    "ch_decoy_ambient_fleet",
                    ("decoy_ambient_temp", "decoy_depot_code"),
                    acquisition_cost=1.0,
                    mechanism_tag="decoy",
                ),
            ]
        )

    Z_pool = pd.concat(z_parts, axis=1)

    return MultiChannelCohort(
        name="Scania_APS_76k_Fleet",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_hist_bins_A",
            "ch_hist_bins_B",
            "ch_anon_counters_A",
            "ch_anon_counters_B",
        },
        y_true=y_true,
        y_tilde=y_true.copy(),
        contract=DecisionContract(
            task_name="Scania_APS_50to1_Cost",
            target_precision=0.50,
            max_fpr_budget=0.01,
            cost_fp=1.0,
            cost_fn=50.0,
            graded_action_cost_fp=0.15,
        ),
        acquisition_budget=2.0,
        base_semantic_types={
            **{c: SemanticType.CONTINUOUS for c in base_cols},
            "telemetry_missing_count": SemanticType.RECEIPT_FLAG,
        },
        receipt_cols=["telemetry_missing_count"],
        raw_aux_df=raw_aux,
        compiler_factories=compiler_factories,
        include_synthetic_proxies=include_synthetic_proxies,
        dataset_provenance="real_observational",
    )


def load_wave2_neurips_baf(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    window_idx: int | None = None,
    n_windows: int = 5,
    max_rows: int | None = None,
    seed: int = 42,
    include_synthetic_proxies: bool = False,
) -> MultiChannelCohort:
    """Load NeurIPS 2022 Bank Account Fraud Base Suite (1M rows, privacy-preserving synthetic)."""
    csv_path = cache_dir / "kaggle" / "BAF" / "Base.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run scripts/stage_wave2_datasets.py.")

    df = pd.read_csv(csv_path, low_memory=False)
    df["__seq_order__"] = df["month"].to_numpy(dtype=np.float64) * 1e7 + np.arange(
        len(df), dtype=np.float64
    )
    df = df.sort_values("__seq_order__", kind="mergesort").reset_index(drop=True)

    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)

    df, is_disjoint = _disjoint_chrono_slice(df, window_idx, n_windows)

    y_true = pd.to_numeric(df.pop("fraud_bool"), errors="coerce").fillna(0).to_numpy(dtype=int)
    time_arr = df.pop("__seq_order__").to_numpy(dtype=np.float64)
    df.pop("month")

    for neg_miss_col in (
        "prev_address_months_count",
        "current_address_months_count",
        "bank_months_count",
        "session_length_in_minutes",
        "intended_balcon_amount",
    ):
        if neg_miss_col in df.columns:
            v = pd.to_numeric(df[neg_miss_col], errors="coerce")
            df[neg_miss_col] = v.where(v >= 0, np.nan).astype(np.float32)

    base_num = [
        "income",
        "customer_age",
        "proposed_credit_limit",
        "intended_balcon_amount",
        "days_since_request",
        "foreign_request",
    ]
    base_cat = ["employment_status", "housing_status", "payment_type", "source"]
    X_base = pd.DataFrame(
        {
            **{c: pd.to_numeric(df[c], errors="coerce").astype(np.float32) for c in base_num},
            **{c: df[c].astype(str) for c in base_cat},
        }
    )

    bureau_cols = (
        "credit_risk_score",
        "has_other_cards",
        "bank_months_count",
        "prev_address_months_count",
        "current_address_months_count",
    )
    velocity_cols = (
        "velocity_6h",
        "velocity_24h",
        "velocity_4w",
        "zip_count_4w",
        "bank_branch_count_8w",
        "date_of_birth_distinct_emails_4w",
    )
    device_num_cols = (
        "name_email_similarity",
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "session_length_in_minutes",
        "keep_alive_session",
        "device_distinct_emails_8w",
    )

    z_dict: dict[str, Any] = {
        **{
            c: pd.to_numeric(df[c], errors="coerce").astype(np.float32)
            for c in bureau_cols + velocity_cols + device_num_cols
        },
        "device_os": df["device_os"].astype(str),
    }

    candidates = [
        CandidateChannelSpec(
            "ch_credit_bureau_api",
            bureau_cols,
            acquisition_cost=1.0,
            mechanism_tag="credit_bureau",
        ),
        CandidateChannelSpec(
            "ch_velocity_consortium_api",
            velocity_cols,
            acquisition_cost=1.0,
            mechanism_tag="velocity_network",
        ),
        CandidateChannelSpec(
            "ch_device_session_telemetry",
            device_num_cols + ("device_os",),
            acquisition_cost=1.0,
            mechanism_tag="device_fingerprint",
        ),
    ]

    if include_synthetic_proxies:
        rng = np.random.default_rng(seed)
        age_v = np.nan_to_num(X_base["customer_age"].to_numpy(dtype=float), nan=35.0)
        inc_v = np.nan_to_num(X_base["income"].to_numpy(dtype=float), nan=0.5)
        lim_v = np.nan_to_num(X_base["proposed_credit_limit"].to_numpy(dtype=float), nan=500.0)
        z_dict["redundant_age_income_score"] = (
            0.04 * age_v + 1.8 * inc_v + rng.normal(0.0, 0.12, len(df))
        ).astype(np.float32)
        z_dict["redundant_limit_echo"] = (
            np.log1p(np.maximum(0.0, lim_v)) + rng.normal(0.0, 0.20, len(df))
        ).astype(np.float32)
        z_dict["decoy_cdn_rtt_ms"] = rng.exponential(18.0, len(df)).astype(np.float32)
        z_dict["decoy_tls_bucket"] = rng.integers(0, 8, len(df)).astype(np.float32)
        candidates.extend(
            [
                CandidateChannelSpec(
                    "ch_redundant_affordability_proxy",
                    ("redundant_age_income_score", "redundant_limit_echo"),
                    acquisition_cost=1.0,
                    mechanism_tag="redundant_proxy",
                ),
                CandidateChannelSpec(
                    "ch_decoy_gateway_noise",
                    ("decoy_cdn_rtt_ms", "decoy_tls_bucket"),
                    acquisition_cost=1.0,
                    mechanism_tag="decoy",
                ),
            ]
        )

    Z_pool = pd.DataFrame(z_dict)

    return MultiChannelCohort(
        name="NeurIPS_BAF_1M_Chronological",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_credit_bureau_api",
            "ch_velocity_consortium_api",
            "ch_device_session_telemetry",
        },
        y_true=y_true,
        y_tilde=y_true.copy(),
        contract=DecisionContract(
            task_name="NeurIPS_BAF_Onboarding",
            target_precision=0.25,
            max_fpr_budget=0.01,
            cost_fp=1.0,
            cost_fn=35.0,
            graded_action_cost_fp=0.15,
        ),
        acquisition_budget=2.0,
        base_semantic_types={
            "income": SemanticType.CONTINUOUS,
            "customer_age": SemanticType.COUNT,
            "proposed_credit_limit": SemanticType.CONTINUOUS,
            "intended_balcon_amount": SemanticType.CONTINUOUS,
            "days_since_request": SemanticType.CONTINUOUS,
            "foreign_request": SemanticType.BINARY,
            "employment_status": SemanticType.NOMINAL,
            "housing_status": SemanticType.NOMINAL,
            "payment_type": SemanticType.NOMINAL,
            "source": SemanticType.NOMINAL,
        },
        time_order=time_arr,
        disjoint_test_blocks=is_disjoint,
        include_synthetic_proxies=include_synthetic_proxies,
        dataset_provenance="privacy_synthetic_from_real",
    )


def _prepare_ieee_cis_cache(cache_dir: Path) -> Path:
    """Build compact row-wise TransactionID left-join cache (zero cross-row statistics!)."""
    pkl_path = cache_dir / "tier2_ieee_cis_joined.pkl"
    if pkl_path.exists():
        return pkl_path

    tr_path = cache_dir / "kaggle" / "IEEE_FRAUD" / "train_transaction.csv"
    id_path = cache_dir / "kaggle" / "IEEE_FRAUD" / "train_identity.csv"
    if not (tr_path.exists() and id_path.exists()):
        raise FileNotFoundError("IEEE-CIS CSVs missing in data/cache/kaggle/IEEE_FRAUD/")

    base_cols = [
        "TransactionID",
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
        "addr2",
        "dist1",
        "dist2",
        "P_emaildomain",
        "R_emaildomain",
    ]
    c_cols = [f"C{i}" for i in range(1, 15)]
    d_cols = [f"D{i}" for i in range(1, 16)]
    m_cols = [f"M{i}" for i in range(1, 10)]
    # Representative Vesta card/address/amount blocks across all structural missingness groups
    v_indices = (
        list(range(1, 12))
        + list(range(12, 35))
        + list(range(35, 53))
        + list(range(53, 75))
        + list(range(75, 95))
        + list(range(95, 107))
        + list(range(126, 138))
        + list(range(279, 316))
    )
    v_cols = [f"V{i}" for i in v_indices]

    use_tr = base_cols + c_cols + d_cols + m_cols + v_cols
    df_tr = pd.read_csv(tr_path, usecols=use_tr, low_memory=False)
    df_id = pd.read_csv(id_path, low_memory=False)
    df_id["__has_id_receipt__"] = np.int8(1)

    merged = df_tr.merge(df_id, on="TransactionID", how="left")
    merged["__has_id_receipt__"] = merged["__has_id_receipt__"].fillna(0).astype(np.int8)
    merged = merged.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)

    # Downcast float64 -> float32 to keep memory footprint compact (~350 MB vs 1.8 GB)
    cat_set = {
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "DeviceInfo",
        *m_cols,
        *(f"id_{i}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)),
    }
    for col in merged.columns:
        if col in {"TransactionID", "isFraud", "TransactionDT", "__has_id_receipt__"}:
            continue
        if col in cat_set or merged[col].dtype == object:
            merged[col] = merged[col].fillna("__MISSING__").astype("category")
        else:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype(np.float32)

    merged.to_pickle(pkl_path)
    return pkl_path


def load_wave2_ieee_cis_fraud(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    window_idx: int | None = None,
    n_windows: int = 5,
    max_rows: int | None = None,
    seed: int = 42,
    include_synthetic_proxies: bool = False,
) -> MultiChannelCohort:
    """Load IEEE-CIS Fraud Detection (590,540 txns + selective identity table, 3.50% fraud)."""
    pkl_path = _prepare_ieee_cis_cache(cache_dir)
    df = pd.read_pickle(pkl_path)

    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].reset_index(drop=True)

    df, is_disjoint = _disjoint_chrono_slice(df, window_idx, n_windows)

    y_true = df["isFraud"].to_numpy(dtype=int)
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
            "dist1": df["dist1"].astype(np.float32),
            "P_emaildomain": df["P_emaildomain"].astype(str),
            "R_emaildomain": df["R_emaildomain"].astype(str),
            "has_identity_receipt": df["__has_id_receipt__"].astype(np.float32),
        }
    )

    id_cols = tuple(
        c for c in df.columns if c.startswith("id_") or c in {"DeviceType", "DeviceInfo"}
    )
    cm_cols = tuple(f"C{i}" for i in range(1, 15)) + tuple(f"M{i}" for i in range(1, 10))
    d_cols = ("dist2",) + tuple(f"D{i}" for i in range(1, 16))
    v_cols = tuple(c for c in df.columns if c.startswith("V"))

    z_dict: dict[str, Any] = {}
    for c in id_cols + cm_cols + d_cols + v_cols:
        if str(df[c].dtype) == "category" or df[c].dtype == object:
            z_dict[c] = df[c].astype(str)
        else:
            z_dict[c] = df[c].astype(np.float32)

    candidates = [
        CandidateChannelSpec(
            "ch_identity_device_table",
            id_cols,
            acquisition_cost=1.0,
            mechanism_tag="selective_identity_table",
        ),
        CandidateChannelSpec(
            "ch_entity_counting_graph_CM",
            cm_cols,
            acquisition_cost=1.0,
            mechanism_tag="entity_link_counts",
        ),
        CandidateChannelSpec(
            "ch_timedelta_dormancy_D",
            d_cols,
            acquisition_cost=1.0,
            mechanism_tag="inter_event_deltas",
        ),
        CandidateChannelSpec(
            "ch_vesta_card_velocity_V",
            v_cols,
            acquisition_cost=1.0,
            mechanism_tag="vesta_velocity_blocks",
        ),
    ]

    if include_synthetic_proxies:
        rng = np.random.default_rng(seed)
        amt_v = np.nan_to_num(X_base["TransactionAmt"].to_numpy(dtype=float), nan=50.0)
        c1_v = np.nan_to_num(X_base["card1"].to_numpy(dtype=float), nan=5000.0)
        rcpt_v = X_base["has_identity_receipt"].to_numpy(dtype=float)
        z_dict["redundant_amt_log_echo"] = (
            np.log1p(np.maximum(0.0, amt_v)) + 0.8 * rcpt_v + rng.normal(0.0, 0.18, len(df))
        ).astype(np.float32)
        z_dict["redundant_card_bin_echo"] = (c1_v + rng.normal(0.0, 150.0, len(df))).astype(
            np.float32
        )
        z_dict["decoy_switch_jitter"] = rng.normal(0.0, 1.0, len(df)).astype(np.float32)
        z_dict["decoy_iso_route"] = rng.integers(1, 10, len(df)).astype(np.float32)
        candidates.extend(
            [
                CandidateChannelSpec(
                    "ch_redundant_switch_echo",
                    ("redundant_amt_log_echo", "redundant_card_bin_echo"),
                    acquisition_cost=1.0,
                    mechanism_tag="redundant_proxy",
                ),
                CandidateChannelSpec(
                    "ch_decoy_switch_latency",
                    ("decoy_switch_jitter", "decoy_iso_route"),
                    acquisition_cost=1.0,
                    mechanism_tag="decoy",
                ),
            ]
        )

    Z_pool = pd.DataFrame(z_dict)

    return MultiChannelCohort(
        name="IEEE_CIS_590k_MultiTable",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_identity_device_table",
            "ch_entity_counting_graph_CM",
            "ch_timedelta_dormancy_D",
            "ch_vesta_card_velocity_V",
        },
        y_true=y_true,
        y_tilde=y_true.copy(),
        contract=DecisionContract(
            task_name="IEEE_CIS_MultiTable_Fraud",
            target_precision=0.65,
            max_fpr_budget=0.005,
            cost_fp=1.0,
            cost_fn=35.0,
            graded_action_cost_fp=0.15,
        ),
        acquisition_budget=2.0,
        base_semantic_types={
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
            "dist1": SemanticType.CONTINUOUS,
            "P_emaildomain": SemanticType.NOMINAL,
            "R_emaildomain": SemanticType.NOMINAL,
            "has_identity_receipt": SemanticType.RECEIPT_FLAG,
        },
        receipt_cols=["has_identity_receipt"],
        time_order=time_arr,
        disjoint_test_blocks=is_disjoint,
        include_synthetic_proxies=include_synthetic_proxies,
        dataset_provenance="real_observational",
    )


def _prepare_home_credit_7table_cache(cache_dir: Path) -> Path:
    """Aggregate all 6 child relational tables strictly per borrower SK_ID_CURR.

    100% leak-free: every summary & 6-step historical panel row depends ONLY on that single
    borrower's own historical pre-application records (DAYS_* <= 0, MONTHS_BALANCE <= 0).
    """
    pkl_path = cache_dir / "tier2_home_credit_7table.pkl"
    if pkl_path.exists():
        return pkl_path

    hc_dir = cache_dir / "kaggle" / "Home_Credit"
    app = pd.read_csv(hc_dir / "HC_application_train.csv", low_memory=False)
    # Clean famous Home Credit DAYS_EMPLOYED == 365243 pensioner sentinel -> NaN
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)

    # 1. Bureau + Bureau Balance (per SK_ID_CURR)
    bur = pd.read_csv(
        hc_dir / "HC_bureau.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_BUREAU",
            "CREDIT_ACTIVE",
            "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE",
            "AMT_CREDIT_MAX_OVERDUE",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
        ],
        low_memory=False,
    )
    bb = pd.read_csv(hc_dir / "HC_bureau_balance.csv", low_memory=False)
    bb_dpd = (
        bb.assign(is_dpd=(~bb["STATUS"].isin(["0", "C", "X"])).astype(np.int32))
        .groupby("SK_ID_BUREAU", as_index=False)["is_dpd"]
        .sum()
    )
    bur = bur.merge(bb_dpd, on="SK_ID_BUREAU", how="left")
    bur["is_active"] = (bur["CREDIT_ACTIVE"] == "Active").astype(np.float32)
    bur["is_dpd"] = bur["is_dpd"].fillna(0.0)
    bur_agg = bur.groupby("SK_ID_CURR", as_index=False).agg(
        bur_count=("SK_ID_BUREAU", "count"),
        bur_active_frac=("is_active", "mean"),
        bur_days_credit_max=("DAYS_CREDIT", "max"),
        bur_days_credit_mean=("DAYS_CREDIT", "mean"),
        bur_max_overdue=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bur_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bur_credit_sum=("AMT_CREDIT_SUM", "sum"),
        bur_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bur_bb_dpd_months=("is_dpd", "sum"),
    )
    bur_agg["bur_debt_to_credit_ratio"] = bur_agg["bur_debt_sum"] / np.maximum(
        1.0, bur_agg["bur_credit_sum"]
    )

    # 2. Previous Applications (including Refused underwriting receipts!)
    prev = pd.read_csv(
        hc_dir / "HC_previous_application.csv",
        usecols=[
            "SK_ID_CURR",
            "SK_ID_PREV",
            "NAME_CONTRACT_STATUS",
            "AMT_APPLICATION",
            "AMT_CREDIT",
            "DAYS_DECISION",
        ],
        low_memory=False,
    )
    prev["is_refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(np.float32)
    prev["is_approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(np.float32)
    prev["cred_to_app"] = prev["AMT_CREDIT"] / np.maximum(1.0, prev["AMT_APPLICATION"])
    prev_agg = prev.groupby("SK_ID_CURR", as_index=False).agg(
        prev_app_count=("SK_ID_PREV", "count"),
        prev_refused_frac=("is_refused", "mean"),
        prev_refused_count=("is_refused", "sum"),
        prev_approved_frac=("is_approved", "mean"),
        prev_cred_to_app_mean=("cred_to_app", "mean"),
        prev_days_decision_max=("DAYS_DECISION", "max"),
    )

    # 3. Installments Payments + 6-bin Pre-Application Trajectory for Post-Split Compilers!
    inst = pd.read_csv(
        hc_dir / "HC_installments_payments.csv",
        usecols=[
            "SK_ID_CURR",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ],
        low_memory=False,
    )
    inst["dpd"] = np.maximum(0.0, inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"])
    inst["pay_ratio"] = inst["AMT_PAYMENT"] / np.maximum(1.0, inst["AMT_INSTALMENT"])
    inst["underpaid"] = (inst["AMT_PAYMENT"] < 0.99 * inst["AMT_INSTALMENT"]).astype(np.float32)
    inst_agg = inst.groupby("SK_ID_CURR", as_index=False).agg(
        inst_count=("DAYS_INSTALMENT", "count"),
        inst_dpd_mean=("dpd", "mean"),
        inst_dpd_max=("dpd", "max"),
        inst_pay_ratio_mean=("pay_ratio", "mean"),
        inst_underpaid_frac=("underpaid", "mean"),
        inst_amt_due_sum=("AMT_INSTALMENT", "sum"),
        inst_amt_paid_sum=("AMT_PAYMENT", "sum"),
    )
    # 6 chronological 90-day windows prior to application: [-540..-450) (t0) up to [-90..0] (t5)
    recent_inst = inst[inst["DAYS_INSTALMENT"] >= -540].copy()
    recent_inst["t_bin"] = np.clip(((recent_inst["DAYS_INSTALMENT"] + 540) // 90).astype(int), 0, 5)
    inst_piv = (
        recent_inst.groupby(["SK_ID_CURR", "t_bin"])["AMT_PAYMENT"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=range(6), fill_value=0.0)
    )
    inst_piv.columns = [f"inst_pay_t{t}" for t in range(6)]
    inst_piv = inst_piv.reset_index()

    # 4. Credit Card Balance + POS Cash Balance + 6-Month Balance Trajectory!
    cc = pd.read_csv(
        hc_dir / "HC_credit_card_balance.csv",
        usecols=[
            "SK_ID_CURR",
            "MONTHS_BALANCE",
            "AMT_BALANCE",
            "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_DRAWINGS_CURRENT",
            "AMT_PAYMENT_TOTAL_CURRENT",
            "SK_DPD",
        ],
        low_memory=False,
    )
    cc["cc_util"] = cc["AMT_BALANCE"] / np.maximum(1000.0, cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc_agg = cc.groupby("SK_ID_CURR", as_index=False).agg(
        cc_months=("MONTHS_BALANCE", "count"),
        cc_util_mean=("cc_util", "mean"),
        cc_util_max=("cc_util", "max"),
        cc_drawings_sum=("AMT_DRAWINGS_CURRENT", "sum"),
        cc_payments_sum=("AMT_PAYMENT_TOTAL_CURRENT", "sum"),
        cc_dpd_max=("SK_DPD", "max"),
    )
    cc_rec = cc[cc["MONTHS_BALANCE"] >= -6].copy()
    cc_rec["m_bin"] = np.clip((cc_rec["MONTHS_BALANCE"] + 6).astype(int), 0, 5)
    cc_piv = (
        cc_rec.groupby(["SK_ID_CURR", "m_bin"])["cc_util"]
        .mean()
        .unstack(fill_value=np.nan)
        .reindex(columns=range(6))
    )
    cc_piv.columns = [f"cc_util_t{t}" for t in range(6)]
    cc_piv = cc_piv.reset_index()

    pos = pd.read_csv(
        hc_dir / "HC_POS_CASH_balance.csv",
        usecols=["SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"],
        low_memory=False,
    )
    pos_agg = pos.groupby("SK_ID_CURR", as_index=False).agg(
        pos_months=("MONTHS_BALANCE", "count"),
        pos_future_inst_min=("CNT_INSTALMENT_FUTURE", "min"),
        pos_dpd_max=("SK_DPD", "max"),
        pos_dpd_def_mean=("SK_DPD_DEF", "mean"),
    )

    merged = (
        app.merge(bur_agg, on="SK_ID_CURR", how="left")
        .merge(prev_agg, on="SK_ID_CURR", how="left")
        .merge(inst_agg, on="SK_ID_CURR", how="left")
        .merge(inst_piv, on="SK_ID_CURR", how="left")
        .merge(cc_agg, on="SK_ID_CURR", how="left")
        .merge(cc_piv, on="SK_ID_CURR", how="left")
        .merge(pos_agg, on="SK_ID_CURR", how="left")
    )
    merged.to_pickle(pkl_path)
    return pkl_path


def load_wave2_home_credit_relational(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_rows: int | None = None,
    seed: int = 42,
    include_synthetic_proxies: bool = False,
) -> MultiChannelCohort:
    """Load Home Credit Default Risk (307,511 borrowers across 7 tables, 8.07% default)."""
    pkl_path = _prepare_home_credit_7table_cache(cache_dir)
    df = pd.read_pickle(pkl_path)

    if max_rows is not None and len(df) > max_rows:
        rng_sub = np.random.default_rng(seed)
        keep_idx = np.sort(rng_sub.choice(len(df), size=max_rows, replace=False))
        df = df.iloc[keep_idx].reset_index(drop=True)

    y_true = df["TARGET"].to_numpy(dtype=int)

    req_cols = [
        c
        for c in (
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
        )
        if c in df.columns
    ]
    req_sum = df[req_cols].fillna(0.0).sum(axis=1).astype(np.float32)

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
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
    ]

    X_base = pd.DataFrame(
        {
            **{c: df[c].fillna("__MISSING__").astype(str) for c in base_cat},
            **{c: pd.to_numeric(df[c], errors="coerce").astype(np.float32) for c in base_num},
            "bureau_inquiry_receipt_total": req_sum,
        }
    )

    inst_traj_cols = [f"inst_pay_t{t}" for t in range(6)]
    cc_traj_cols = [f"cc_util_t{t}" for t in range(6)]
    raw_aux = pd.DataFrame(
        {
            **{
                c: pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
                for c in inst_traj_cols
            },
            **{c: pd.to_numeric(df[c], errors="coerce").astype(np.float32) for c in cc_traj_cols},
            "inst_amt_due_sum": pd.to_numeric(df["inst_amt_due_sum"], errors="coerce")
            .fillna(0.0)
            .astype(np.float32),
            "inst_amt_paid_sum": pd.to_numeric(df["inst_amt_paid_sum"], errors="coerce")
            .fillna(0.0)
            .astype(np.float32),
        }
    )

    compiler_factories: list[Callable[[], Any]] = [
        lambda: MissingLineSilenceCompiler({"inst_pay": inst_traj_cols}),
        lambda: CriticalSlowingDownCompiler({"inst_pay": inst_traj_cols, "cc_util": cc_traj_cols}),
        lambda: ConservationMismatchCompiler(
            [
                ConservationRule(
                    "inst_flow",
                    "flow_balance",
                    ("inst_amt_due_sum",),
                    ("inst_amt_paid_sum",),
                )
            ]
        ),
    ]

    eesh_inst_cols = (
        "eesh__inst_pay__silence_streak",
        "eesh__inst_pay__silence_hazard",
    )
    csd_inst_cols = tuple(
        f"csd__inst_pay__{s}" for s in ("diff_base", "log_var_inflation", "ac1", "slope")
    )
    inv_inst_cols = ("inv__inst_flow__signed_z", "inv__inst_flow__violation_abs")
    csd_cc_cols = tuple(
        f"csd__cc_util__{s}" for s in ("diff_base", "log_var_inflation", "ac1", "slope")
    )

    ext_cols = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
    bur_cols = (
        "bur_count",
        "bur_active_frac",
        "bur_days_credit_max",
        "bur_days_credit_mean",
        "bur_max_overdue",
        "bur_debt_sum",
        "bur_credit_sum",
        "bur_overdue_sum",
        "bur_bb_dpd_months",
        "bur_debt_to_credit_ratio",
    )
    inst_stat_cols = (
        "inst_count",
        "inst_dpd_mean",
        "inst_dpd_max",
        "inst_pay_ratio_mean",
        "inst_underpaid_frac",
    )
    rev_stat_cols = (
        "cc_months",
        "cc_util_mean",
        "cc_util_max",
        "cc_drawings_sum",
        "cc_payments_sum",
        "cc_dpd_max",
        "pos_months",
        "pos_future_inst_min",
        "pos_dpd_max",
        "pos_dpd_def_mean",
    )
    prev_cols = (
        "prev_app_count",
        "prev_refused_frac",
        "prev_refused_count",
        "prev_approved_frac",
        "prev_cred_to_app_mean",
        "prev_days_decision_max",
    )

    z_dict: dict[str, Any] = {
        c: pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        for c in ext_cols + bur_cols + inst_stat_cols + rev_stat_cols + prev_cols
    }

    candidates = [
        CandidateChannelSpec(
            "ch_external_bureau_scores",
            ext_cols,
            acquisition_cost=1.0,
            mechanism_tag="external_score_vendors",
        ),
        CandidateChannelSpec(
            "ch_bureau_tradelines_table",
            bur_cols,
            acquisition_cost=1.0,
            mechanism_tag="credit_bureau_tradelines",
        ),
        CandidateChannelSpec(
            "ch_installments_silence_flow_table",
            inst_stat_cols + eesh_inst_cols + csd_inst_cols + inv_inst_cols,
            acquisition_cost=1.0,
            mechanism_tag="installment_silence_conservation",
        ),
        CandidateChannelSpec(
            "ch_revolving_pos_card_table",
            rev_stat_cols + csd_cc_cols,
            acquisition_cost=1.0,
            mechanism_tag="revolving_csd_dynamics",
        ),
        CandidateChannelSpec(
            "ch_prior_applications_receipt_table",
            prev_cols,
            acquisition_cost=1.0,
            mechanism_tag="underwriting_refusal_receipts",
        ),
    ]

    if include_synthetic_proxies:
        rng = np.random.default_rng(seed)
        cred_v = np.nan_to_num(X_base["AMT_CREDIT"].to_numpy(dtype=float), nan=5e5)
        ann_v = np.nan_to_num(X_base["AMT_ANNUITY"].to_numpy(dtype=float), nan=2.5e4)
        inc_v = np.nan_to_num(X_base["AMT_INCOME_TOTAL"].to_numpy(dtype=float), nan=1.5e5)
        birth_v = np.nan_to_num(X_base["DAYS_BIRTH"].to_numpy(dtype=float), nan=-15000.0)
        z_dict["redundant_dti_echo"] = (
            (ann_v / np.maximum(1000.0, inc_v)) + rng.normal(0.0, 0.03, len(df))
        ).astype(np.float32)
        z_dict["redundant_lti_age_echo"] = (
            (cred_v / np.maximum(1000.0, inc_v)) - 0.0002 * birth_v + rng.normal(0.0, 0.35, len(df))
        ).astype(np.float32)
        z_dict["decoy_ad_campaign"] = rng.normal(0.0, 1.0, len(df)).astype(np.float32)
        z_dict["decoy_kiosk_id"] = rng.integers(1, 20, len(df)).astype(np.float32)
        candidates.extend(
            [
                CandidateChannelSpec(
                    "ch_redundant_intake_affordability",
                    ("redundant_dti_echo", "redundant_lti_age_echo"),
                    acquisition_cost=1.0,
                    mechanism_tag="redundant_proxy",
                ),
                CandidateChannelSpec(
                    "ch_decoy_marketing_channel",
                    ("decoy_ad_campaign", "decoy_kiosk_id"),
                    acquisition_cost=1.0,
                    mechanism_tag="decoy",
                ),
            ]
        )

    Z_pool = pd.DataFrame(z_dict)

    return MultiChannelCohort(
        name="Home_Credit_307k_7Table_Relational",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_external_bureau_scores",
            "ch_bureau_tradelines_table",
            "ch_installments_silence_flow_table",
            "ch_revolving_pos_card_table",
            "ch_prior_applications_receipt_table",
        },
        y_true=y_true,
        y_tilde=y_true.copy(),
        contract=DecisionContract(
            task_name="Home_Credit_Default_7Table",
            target_precision=0.32,
            max_fpr_budget=0.03,
            cost_fp=1.0,
            cost_fn=15.0,
            graded_action_cost_fp=0.20,
        ),
        acquisition_budget=3.0,
        base_semantic_types={
            **{c: SemanticType.NOMINAL for c in base_cat},
            **{c: SemanticType.CONTINUOUS for c in base_num},
            "bureau_inquiry_receipt_total": SemanticType.RECEIPT_FLAG,
        },
        receipt_cols=["bureau_inquiry_receipt_total"],
        raw_aux_df=raw_aux,
        compiler_factories=compiler_factories,
        include_synthetic_proxies=include_synthetic_proxies,
        dataset_provenance="real_observational",
    )
