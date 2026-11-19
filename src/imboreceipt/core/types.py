"""Core types, semantic operator safety, and leak-free split isolation."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


class SemanticType(StrEnum):
    """Dimensional / semantic types governing legal feature operations."""

    CONTINUOUS = "continuous"
    COUNT = "count"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    BINARY = "binary"
    TIMESTAMP = "timestamp"
    ENTITY_ID = "entity_id"
    RECEIPT_FLAG = "receipt_flag"


_QUANTITATIVE_TYPES: frozenset[SemanticType] = frozenset(
    {SemanticType.CONTINUOUS, SemanticType.COUNT}
)
_ORDERED_NUMERIC_TYPES: frozenset[SemanticType] = frozenset(
    {SemanticType.CONTINUOUS, SemanticType.COUNT, SemanticType.ORDINAL}
)
_DISCRETE_MATCHABLE_TYPES: frozenset[SemanticType] = frozenset(
    {SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY, SemanticType.RECEIPT_FLAG}
)


def infer_semantic_types(
    df: pd.DataFrame,
    overrides: dict[str, SemanticType | str] | None = None,
    exclude_cols: set[str] | None = None,
) -> dict[str, SemanticType]:
    """Infer column semantic types from a DataFrame with explicit override precedence.

    Crucially, explicit overrides (such as declaring an ``int64`` hospital code as
    ``SemanticType.NOMINAL``) take unconditional precedence at both fit and transform
    stages, preventing dtype-reinference drift.
    """
    excluded = exclude_cols or set()
    override_map: dict[str, SemanticType] = {}
    if overrides:
        for col, stype in overrides.items():
            override_map[col] = SemanticType(stype)

    result: dict[str, SemanticType] = {}
    for col in df.columns:
        if col in excluded:
            continue
        if col in override_map:
            result[col] = override_map[col]
            continue

        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            result[col] = SemanticType.TIMESTAMP
        elif pd.api.types.is_bool_dtype(s):
            result[col] = SemanticType.BINARY
        elif pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            unique_vals = set(non_null.unique().tolist()) if len(non_null) > 0 else set()
            if unique_vals and all(v in (0, 1) for v in unique_vals):
                result[col] = SemanticType.BINARY
            elif pd.api.types.is_integer_dtype(s) and (non_null >= 0).all():
                result[col] = SemanticType.COUNT
            else:
                result[col] = SemanticType.CONTINUOUS
        else:
            result[col] = SemanticType.NOMINAL
    return result


def is_operator_compatible(op_name: str, arg_types: tuple[SemanticType, ...]) -> bool:
    """Verify dimensional and semantic legality of an operator on input types.

    Forbids meaningless arithmetic (such as dividing nominal integer ID codes)
    while permitting appropriate indicator, frequency, and consistency operations.
    """
    if any(t == SemanticType.ENTITY_ID for t in arg_types):
        return False

    if op_name in {"ratio", "product", "difference", "sum", "abs_diff"}:
        return len(arg_types) == 2 and all(t in _QUANTITATIVE_TYPES for t in arg_types)

    if op_name in {"log1p", "zscore", "clip", "csd_variance", "csd_ac1", "diff_baseline"}:
        return len(arg_types) == 1 and arg_types[0] in _QUANTITATIVE_TYPES

    if op_name in {"quantile_bin", "threshold_gt", "threshold_lt"}:
        return len(arg_types) == 1 and arg_types[0] in _ORDERED_NUMERIC_TYPES

    if op_name == "missing_indicator":
        return len(arg_types) == 1

    if op_name == "frequency_encode":
        return len(arg_types) == 1 and arg_types[0] in _DISCRETE_MATCHABLE_TYPES

    if op_name in {"equality_mismatch", "conjunction_and", "disjunction_or"}:
        if op_name in {"conjunction_and", "disjunction_or"}:
            return len(arg_types) == 2 and all(
                t in {SemanticType.BINARY, SemanticType.RECEIPT_FLAG} for t in arg_types
            )
        return (
            len(arg_types) == 2
            and arg_types[0] in _DISCRETE_MATCHABLE_TYPES
            and arg_types[1] in _DISCRETE_MATCHABLE_TYPES
        )

    return False


class SplitProtocol(BaseModel):
    """Immutable 4-way partition (train / cal / val / test) with SHA-256 lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_idx: tuple[int, ...]
    cal_idx: tuple[int, ...]
    val_idx: tuple[int, ...]
    test_idx: tuple[int, ...]
    grouped_by: str | None = None
    ordered_by: str | None = None
    seed: int = 42
    split_hash: str

    @model_validator(mode="after")
    def verify_disjoint_nonempty(self) -> SplitProtocol:
        parts = [
            set(self.train_idx),
            set(self.cal_idx),
            set(self.val_idx),
            set(self.test_idx),
        ]
        for name, part in zip(("train", "cal", "val", "test"), parts, strict=True):
            if not part:
                raise ValueError(f"Split partition '{name}' cannot be empty.")
        total_unique = len(parts[0] | parts[1] | parts[2] | parts[3])
        if total_unique != sum(len(p) for p in parts):
            raise ValueError("Split partitions have overlapping row indices!")
        return self


def _compute_split_hash(
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    grouped_by: str | None,
    ordered_by: str | None,
    seed: int,
) -> str:
    payload = {
        "train": [int(i) for i in train_idx],
        "cal": [int(i) for i in cal_idx],
        "val": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
        "grouped_by": grouped_by,
        "ordered_by": ordered_by,
        "seed": seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_split(
    df: pd.DataFrame,
    target_col: str,
    *,
    group_col: str | None = None,
    time_col: str | None = None,
    train_frac: float = 0.55,
    cal_frac: float = 0.15,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitProtocol:
    """Create a leak-free 4-way split BEFORE any preprocessing or feature learning.

    - If ``time_col`` is supplied without ``group_col``, performs a strict chronological split.
    - If ``group_col`` is supplied, assigns entire entity/patient groups to partitions
      using stratified group allocation so zero group overlap occurs and every partition
      retains minority events.
    - Otherwise performs stratified random 4-way splitting.
    """
    if not np.isclose(train_frac + cal_frac + val_frac + test_frac, 1.0, atol=1e-5):
        raise ValueError("Partition fractions must sum to 1.0.")

    n = len(df)
    if n < 20:
        raise ValueError(f"Dataset too small ({n} rows) for 4-way splitting.")

    y = np.asarray(df[target_col], dtype=int)
    rng = np.random.default_rng(seed)

    if time_col is not None and group_col is None:
        t_arr = np.asarray(df[time_col], dtype=float)
        order = np.argsort(t_arr, kind="mergesort")
        c1 = int(round(n * train_frac))
        c2 = c1 + int(round(n * cal_frac))
        c3 = c2 + int(round(n * val_frac))

        # Advance cut indices past tied timestamps where possible so max(prev_t) < min(next_t)
        def _advance_past_ties(cut: int, limit_hi: int) -> int:
            if cut <= 0 or cut >= limit_hi:
                return cut
            t_boundary = t_arr[order[cut - 1]]
            cand = cut
            while cand < limit_hi - 4 and t_arr[order[cand]] <= t_boundary:
                cand += 1
            return cand if cand < limit_hi - 4 else cut

        c1 = _advance_past_ties(c1, c2)
        c2 = _advance_past_ties(c2, c3)
        c3 = _advance_past_ties(c3, n)

        train_idx = np.sort(order[:c1])
        cal_idx = np.sort(order[c1:c2])
        val_idx = np.sort(order[c2:c3])
        test_idx = np.sort(order[c3:])
    elif time_col is not None and group_col is not None:
        # Enforce BOTH forward chronological order AND zero entity overlap via forward purge
        t_arr = np.asarray(df[time_col], dtype=float)
        g_arr = df[group_col].to_numpy()
        order = np.argsort(t_arr, kind="mergesort")
        c1 = int(round(n * train_frac))
        c2 = c1 + int(round(n * cal_frac))
        c3 = c2 + int(round(n * val_frac))

        raw_tr = order[:c1]
        seen_tr = set(g_arr[raw_tr].tolist())

        raw_cal = order[c1:c2]
        keep_cal = np.array([g not in seen_tr for g in g_arr[raw_cal]], dtype=bool)
        cal_clean = raw_cal[keep_cal]
        seen_tr_cal = seen_tr | set(g_arr[cal_clean].tolist())

        raw_val = order[c2:c3]
        keep_val = np.array([g not in seen_tr_cal for g in g_arr[raw_val]], dtype=bool)
        val_clean = raw_val[keep_val]
        seen_all_dev = seen_tr_cal | set(g_arr[val_clean].tolist())

        raw_tst = order[c3:]
        keep_tst = np.array([g not in seen_all_dev for g in g_arr[raw_tst]], dtype=bool)
        tst_clean = raw_tst[keep_tst]

        if min(raw_tr.size, cal_clean.size, val_clean.size, tst_clean.size) == 0:
            raise ValueError(
                "Joint chronological + grouped split purged an entire partition; "
                "insufficient new groups in later eras."
            )
        train_idx = np.sort(raw_tr)
        cal_idx = np.sort(cal_clean)
        val_idx = np.sort(val_clean)
        test_idx = np.sort(tst_clean)
    elif group_col is not None:
        groups = df[group_col].to_numpy()
        # Group-level aggregation for stratified group allocation
        grp_df = pd.DataFrame({"row_pos": np.arange(n), "grp": groups, "y": y})
        grp_stats = (
            grp_df.groupby("grp", sort=False)
            .agg(
                has_pos=("y", "max"),
                n_rows=("row_pos", "count"),
            )
            .reset_index()
        )
        # Shuffle groups deterministically within positive/negative strata
        pos_grps = grp_stats[grp_stats["has_pos"] > 0]["grp"].to_numpy().copy()
        neg_grps = grp_stats[grp_stats["has_pos"] == 0]["grp"].to_numpy().copy()
        rng.shuffle(pos_grps)
        rng.shuffle(neg_grps)

        fracs = np.array([train_frac, cal_frac, val_frac, test_frac])
        buckets: list[list[object]] = [[], [], [], []]
        for grp_arr in (pos_grps, neg_grps):
            m = len(grp_arr)
            if m < 4:
                raise ValueError(
                    f"Too few distinct groups ({m}) in stratum to populate 4 partitions."
                )
            cuts = np.round(np.cumsum(fracs)[:-1] * m).astype(int)
            cuts = np.maximum.accumulate(np.clip(cuts, [1, 2, 3], [m - 3, m - 2, m - 1]))
            splits = np.split(grp_arr, cuts)
            for b_idx, chunk in enumerate(splits):
                buckets[b_idx].extend(chunk.tolist())

        grp_to_bucket: dict[object, int] = {}
        for b_idx, g_list in enumerate(buckets):
            for g in g_list:
                grp_to_bucket[g] = b_idx

        assigned = np.array([grp_to_bucket[g] for g in groups], dtype=int)
        train_idx = np.flatnonzero(assigned == 0)
        cal_idx = np.flatnonzero(assigned == 1)
        val_idx = np.flatnonzero(assigned == 2)
        test_idx = np.flatnonzero(assigned == 3)
    else:
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        if len(pos_idx) < 4 or len(neg_idx) < 4:
            raise ValueError("Need at least 4 positive and 4 negative instances for 4-way split.")
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        fracs = np.array([train_frac, cal_frac, val_frac, test_frac])

        part_lists: list[list[int]] = [[], [], [], []]
        for cls_idx in (pos_idx, neg_idx):
            m = len(cls_idx)
            cuts = np.round(np.cumsum(fracs)[:-1] * m).astype(int)
            cuts = np.maximum.accumulate(np.clip(cuts, [1, 2, 3], [m - 3, m - 2, m - 1]))
            for b_idx, chunk in enumerate(np.split(cls_idx, cuts)):
                part_lists[b_idx].extend(int(i) for i in chunk)

        train_idx = np.sort(np.array(part_lists[0], dtype=int))
        cal_idx = np.sort(np.array(part_lists[1], dtype=int))
        val_idx = np.sort(np.array(part_lists[2], dtype=int))
        test_idx = np.sort(np.array(part_lists[3], dtype=int))

    shash = _compute_split_hash(train_idx, cal_idx, val_idx, test_idx, group_col, time_col, seed)
    return SplitProtocol(
        train_idx=tuple(int(i) for i in train_idx),
        cal_idx=tuple(int(i) for i in cal_idx),
        val_idx=tuple(int(i) for i in val_idx),
        test_idx=tuple(int(i) for i in test_idx),
        grouped_by=group_col,
        ordered_by=time_col,
        seed=seed,
        split_hash=shash,
    )
