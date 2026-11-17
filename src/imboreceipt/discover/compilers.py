"""Stage D (Part 2): Mechanistic sequence/invariant compilers and typed feature DAGs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from imboreceipt.core.canonical import (
    CanonicalNode,
    canonical_plan_hash,
    prediction_signature_hash,
)
from imboreceipt.core.types import SemanticType, is_operator_compatible


class CriticalSlowingDownCompiler:
    """Extracts H8 dynamical early-warning & astronomy difference-imaging features from panels.

    Given ordered longitudinal measurements ``(col_t0, col_t1, ..., col_tK)`` (oldest to newest),
    computes strictly row-causal (pre-index) trajectory summaries:
    - ``diff_base``: recent state minus unit's early baseline mean (removes hard negatives).
    - ``var_ratio``: log ratio of recent to early baseline variance (critical slowing down).
    - ``ac1``: within-unit lag-1 autocorrelation (rising memory near regime bifurcation).
    - ``recovery_slope``: linear trend over the trajectory window.
    """

    def __init__(self, trajectory_groups: dict[str, list[str]]) -> None:
        self.trajectory_groups = trajectory_groups
        self._fitted = False

    def fit(self, X_train: pd.DataFrame) -> CriticalSlowingDownCompiler:
        for name, cols in self.trajectory_groups.items():
            if len(cols) < 3:
                raise ValueError(
                    f"Trajectory '{name}' needs >=3 ordered time steps, got {len(cols)}."
                )
            missing = [c for c in cols if c not in X_train.columns]
            if missing:
                raise ValueError(f"Trajectory '{name}' missing columns: {missing}")
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("CriticalSlowingDownCompiler must be fit before transform.")

        out: dict[str, np.ndarray] = {}
        for name, cols in self.trajectory_groups.items():
            mat = np.column_stack(
                [pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float) for c in cols]
            )
            # Forward-fill NaNs along row trajectory, fallback to 0
            for t in range(1, mat.shape[1]):
                nan_m = ~np.isfinite(mat[:, t])
                mat[nan_m, t] = mat[nan_m, t - 1]
            mat = np.nan_to_num(mat, nan=0.0)

            t_len = mat.shape[1]
            mid = max(1, t_len // 2)
            early = mat[:, :mid]
            recent = mat[:, mid:]

            # 1. Difference-imaging self-baseline
            early_mean = np.mean(early, axis=1)
            out[f"csd__{name}__diff_base"] = mat[:, -1] - early_mean

            # 2. Variance inflation under load (Critical Slowing Down)
            var_early = np.var(early, axis=1) + 1e-4
            var_recent = np.var(recent, axis=1) + 1e-4
            out[f"csd__{name}__log_var_inflation"] = np.log(var_recent / var_early)

            # 3. Within-unit Lag-1 Autocorrelation (AC1)
            x_prev = mat[:, :-1]
            x_next = mat[:, 1:]
            m_prev = np.mean(x_prev, axis=1, keepdims=True)
            m_next = np.mean(x_next, axis=1, keepdims=True)
            num = np.sum((x_prev - m_prev) * (x_next - m_next), axis=1)
            den = np.sqrt(
                np.sum((x_prev - m_prev) ** 2, axis=1) * np.sum((x_next - m_next) ** 2, axis=1)
            )
            ac1 = np.where(den > 1e-9, num / np.maximum(1e-9, den), 0.0)
            out[f"csd__{name}__ac1"] = np.clip(ac1, -1.0, 1.0)

            # 4. Trajectory slope
            t_grid = np.arange(t_len, dtype=float)
            t_centered = t_grid - np.mean(t_grid)
            t_ss = max(1e-9, float(np.sum(t_centered**2)))
            slope = (mat @ t_centered) / t_ss
            out[f"csd__{name}__slope"] = slope

        return pd.DataFrame(out, index=X.index)


class MissingLineSilenceCompiler:
    """Extracts 'The Missing Line' Expected-Event Silence Hazard (EESH) from activity cadences.

    In longitudinal event/payment/refill logs, an expected periodic action that suddenly stops
    occurring (e.g., regular monthly payment dropping to 0 while balance > 0, or regular
    heartbeat/check-in halting) carries strong rare-event hazard signal invisible to snapshots.
    """

    def __init__(self, activity_sequences: dict[str, list[str]]) -> None:
        self.activity_sequences = activity_sequences
        self._cadence_eps: dict[str, float] = {}
        self._fitted = False

    def fit(self, X_train: pd.DataFrame) -> MissingLineSilenceCompiler:
        self._cadence_eps = {}
        for name, cols in self.activity_sequences.items():
            mat = np.column_stack(
                [
                    pd.to_numeric(X_train[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    for c in cols
                ]
            )
            pos_vals = np.abs(mat[np.abs(mat) > 0])
            thresh = float(np.quantile(pos_vals, 0.05)) if pos_vals.size > 0 else 1e-6
            self._cadence_eps[name] = max(1e-6, thresh * 0.5)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("MissingLineSilenceCompiler must be fit before transform.")

        out: dict[str, np.ndarray] = {}
        for name, cols in self.activity_sequences.items():
            eps = self._cadence_eps[name]
            mat = np.column_stack(
                [
                    pd.to_numeric(X[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    for c in cols
                ]
            )
            active = (mat > eps).astype(float)
            t_len = active.shape[1]

            # Historical expected cadence rate prior to final step(s)
            hist_rate = np.mean(active[:, :-1], axis=1)
            # Trailing silence streak length (consecutive inactive steps at end of window)
            streak = np.zeros(len(X), dtype=float)
            still_silent = np.ones(len(X), dtype=bool)
            for step in range(t_len - 1, -1, -1):
                is_zero = active[:, step] < 0.5
                still_silent &= is_zero
                streak += still_silent.astype(float)

            # Expected-Event Silence Hazard: -streak * log(1 - hist_rate)
            # High ONLY when an entity with established regular activity suddenly goes silent!
            hazard = -streak * np.log(np.clip(1.0 - hist_rate, 0.02, 1.0))
            out[f"eesh__{name}__silence_streak"] = streak
            out[f"eesh__{name}__silence_hazard"] = hazard

        return pd.DataFrame(out, index=X.index)


@dataclass(frozen=True, slots=True)
class ConservationRule:
    """Declarative conservation or consistency invariant between columns."""

    name: str
    kind: str  # "product_equals", "flow_balance", "ratio_consistency"
    lhs_cols: tuple[str, ...]
    rhs_cols: tuple[str, ...]


class ConservationMismatchCompiler:
    """Computes violated conservation-law and cross-channel consistency residuals.

     learns normal healthy residual mean/scale strictly on training rows and emits
    standardized invariant-violation magnitudes on validation/test rows.
    """

    def __init__(self, rules: list[ConservationRule]) -> None:
        self.rules = rules
        self._stats: dict[str, tuple[float, float]] = {}
        self._fitted = False

    def _raw_residual(self, X: pd.DataFrame, rule: ConservationRule) -> np.ndarray:
        if rule.kind == "product_equals":
            # e.g. Mechanical Power = Torque * Omega vs measured power/thermal proxy
            prod = np.ones(len(X), dtype=float)
            for c in rule.lhs_cols:
                prod *= pd.to_numeric(X[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if not rule.rhs_cols:
                return prod
            rhs = (
                pd.to_numeric(X[rule.rhs_cols[0]], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            return prod - rhs
        if rule.kind == "flow_balance":
            # e.g. sum(lhs) - sum(rhs) conservation balance
            lhs = np.zeros(len(X), dtype=float)
            for c in rule.lhs_cols:
                lhs += pd.to_numeric(X[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            rhs = np.zeros(len(X), dtype=float)
            for c in rule.rhs_cols:
                rhs += pd.to_numeric(X[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            return lhs - rhs
        if rule.kind == "ratio_consistency":
            num = (
                pd.to_numeric(X[rule.lhs_cols[0]], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            den = (
                pd.to_numeric(X[rule.rhs_cols[0]], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            return num / (np.abs(den) + 1.0)
        raise ValueError(f"Unknown ConservationRule kind: {rule.kind}")

    def fit(self, X_train: pd.DataFrame) -> ConservationMismatchCompiler:
        self._stats = {}
        for rule in self.rules:
            res = self._raw_residual(X_train, rule)
            med = float(np.nanmedian(res))
            mad = float(np.nanmedian(np.abs(res - med))) * 1.4826
            self._stats[rule.name] = (med, max(1e-6, mad))
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("ConservationMismatchCompiler must be fit before transform.")

        out: dict[str, np.ndarray] = {}
        for rule in self.rules:
            med, scale = self._stats[rule.name]
            res = self._raw_residual(X, rule)
            z = (res - med) / scale
            out[f"inv__{rule.name}__signed_z"] = np.clip(z, -25.0, 25.0)
            out[f"inv__{rule.name}__violation_abs"] = np.clip(np.abs(z), 0.0, 25.0)
        return pd.DataFrame(out, index=X.index)


class TypedFeatureDAGCompiler:
    """Executes semantically validated, 3-identity canonicalized feature DAGs without raw exec()."""

    def __init__(
        self,
        plans: dict[str, list[CanonicalNode]],
        semantic_types: dict[str, SemanticType],
    ) -> None:
        self.plans = plans
        self.semantic_types = semantic_types
        self.accepted_plans_: dict[str, list[CanonicalNode]] = {}
        self.plan_hashes_: dict[str, str] = {}
        self.prediction_hashes_: dict[str, str] = {}
        self._learned_thresholds: dict[tuple[str, str], float] = {}
        self._fitted = False

    def _eval_plan(
        self,
        plan_name: str,
        nodes: list[CanonicalNode],
        X: pd.DataFrame,
        fit_mode: bool,
    ) -> np.ndarray:
        vals: dict[str, np.ndarray] = {}
        types: dict[str, SemanticType] = dict(self.semantic_types)

        for node in nodes:
            in_types = tuple(types[inp] for inp in node.inputs)
            if not is_operator_compatible(node.op, in_types):
                raise ValueError(
                    f"Illegal semantic operation '{node.op}' on types {in_types} in '{plan_name}'."
                )

            args: list[np.ndarray] = []
            for inp in node.inputs:
                if inp in vals:
                    args.append(vals[inp])
                else:
                    if types[inp] == SemanticType.NOMINAL:
                        args.append(X[inp].fillna("__MISSING__").astype(str).to_numpy())
                    else:
                        args.append(
                            pd.to_numeric(X[inp], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                        )

            params_dict = dict(node.params)
            op = node.op
            if op == "sum":
                out_v = args[0] + args[1]
                out_t = SemanticType.CONTINUOUS
            elif op == "difference":
                out_v = args[0] - args[1]
                out_t = SemanticType.CONTINUOUS
            elif op == "abs_diff":
                out_v = np.abs(args[0] - args[1])
                out_t = SemanticType.CONTINUOUS
            elif op == "product":
                out_v = args[0] * args[1]
                out_t = SemanticType.CONTINUOUS
            elif op == "ratio":
                denom = np.where(np.abs(args[1]) < 1e-3, np.sign(args[1] + 1e-9) * 1e-3, args[1])
                out_v = args[0] / denom
                out_t = SemanticType.CONTINUOUS
            elif op == "log1p":
                out_v = np.sign(args[0]) * np.log1p(np.abs(args[0]))
                out_t = SemanticType.CONTINUOUS
            elif op == "threshold_gt":
                key = (plan_name, node.node_id)
                if fit_mode:
                    q = float(params_dict.get("quantile", 0.85))
                    thr = (
                        float(np.nanquantile(args[0], q))
                        if "value" not in params_dict
                        else float(params_dict["value"])
                    )
                    self._learned_thresholds[key] = thr
                thr = self._learned_thresholds[key]
                out_v = (args[0] > thr).astype(float)
                out_t = SemanticType.BINARY
            elif op == "threshold_lt":
                key = (plan_name, node.node_id)
                if fit_mode:
                    q = float(params_dict.get("quantile", 0.15))
                    thr = (
                        float(np.nanquantile(args[0], q))
                        if "value" not in params_dict
                        else float(params_dict["value"])
                    )
                    self._learned_thresholds[key] = thr
                thr = self._learned_thresholds[key]
                out_v = (args[0] < thr).astype(float)
                out_t = SemanticType.BINARY
            elif op == "equality_mismatch":
                out_v = (args[0] != args[1]).astype(float)
                out_t = SemanticType.BINARY
            elif op == "conjunction_and":
                out_v = ((args[0] > 0.5) & (args[1] > 0.5)).astype(float)
                out_t = SemanticType.BINARY
            elif op == "disjunction_or":
                out_v = ((args[0] > 0.5) | (args[1] > 0.5)).astype(float)
                out_t = SemanticType.BINARY
            elif op == "missing_indicator":
                src_col = node.inputs[0]
                out_v = X[src_col].isna().to_numpy(dtype=float)
                out_t = SemanticType.BINARY
            else:
                raise ValueError(f"Unsupported DAG operator: {op}")

            vals[node.node_id] = out_v
            types[node.node_id] = out_t

        return np.asarray(vals[nodes[-1].node_id], dtype=float)

    def fit(self, X_train: pd.DataFrame) -> TypedFeatureDAGCompiler:
        """Validate semantics and deduplicate by BOTH canonical plan hash and prediction hash."""
        self.accepted_plans_ = {}
        self.plan_hashes_ = {}
        self.prediction_hashes_ = {}
        seen_plan_hashes: set[str] = set()
        seen_pred_hashes: set[str] = set()

        for name, nodes in self.plans.items():
            p_hash = canonical_plan_hash(nodes, self.semantic_types)
            if p_hash in seen_plan_hashes:
                continue

            vec = self._eval_plan(name, nodes, X_train, fit_mode=True)
            pred_hash = prediction_signature_hash(vec)
            if pred_hash in seen_pred_hashes:
                continue

            seen_plan_hashes.add(p_hash)
            seen_pred_hashes.add(pred_hash)
            self.accepted_plans_[name] = nodes
            self.plan_hashes_[name] = p_hash
            self.prediction_hashes_[name] = pred_hash

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TypedFeatureDAGCompiler must be fit on X_train before transform.")
        out: dict[str, np.ndarray] = {}
        for name, nodes in self.accepted_plans_.items():
            out[name] = self._eval_plan(name, nodes, X, fit_mode=False)
        return pd.DataFrame(out, index=X.index)
