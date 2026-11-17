"""Three-identity canonicalization preventing duplicate evaluation spend and FDR dilution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from imboreceipt.core.types import SemanticType

_COMMUTATIVE_OPS: frozenset[str] = frozenset(
    {"sum", "product", "abs_diff", "equality_mismatch", "conjunction_and", "disjunction_or"}
)

_OP_ALIASES: dict[str, str] = {
    "add": "sum",
    "plus": "sum",
    "mul": "product",
    "multiply": "product",
    "sub": "difference",
    "minus": "difference",
    "div": "ratio",
    "divide": "ratio",
    "mismatch": "equality_mismatch",
    "and": "conjunction_and",
    "or": "disjunction_or",
}


@dataclass(frozen=True, slots=True)
class CanonicalNode:
    """Single canonicalized operation node inside a feature DAG."""

    node_id: str
    op: str
    inputs: tuple[str, ...]
    params: tuple[tuple[str, float | int | str | bool], ...] = ()


def build_construct_id(mechanism_family: str, target_stratum: str, concept_slug: str) -> str:
    """Compute Construct Identity (conceptual mechanism + targeted residual stratum)."""
    norm = {
        "mechanism": mechanism_family.strip().lower(),
        "stratum": target_stratum.strip().lower(),
        "concept": concept_slug.strip().lower(),
    }
    raw = json.dumps(norm, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def canonical_plan_hash(
    nodes: list[CanonicalNode] | tuple[CanonicalNode, ...],
    semantic_types: dict[str, SemanticType],
    output_node_id: str | None = None,
) -> str:
    """Compute Executable-Plan Identity by normalizing aliases, commutativity, and DAG order.

    Algebraically equivalent feature graphs (e.g. ``add(b, a)`` vs ``sum(a, b)``, or
    differently named intermediate nodes) hash to the exact same 64-char SHA-256 digest.
    """
    if not nodes:
        raise ValueError("Feature plan must contain at least one node.")

    expr_by_id: dict[str, str] = {}
    # Source columns resolve to their column name + semantic type
    for col, stype in semantic_types.items():
        expr_by_id[col] = f"src:{col}:{stype.value}"

    for node in nodes:
        norm_op = _OP_ALIASES.get(node.op.strip().lower(), node.op.strip().lower())
        resolved_inputs: list[str] = []
        for inp in node.inputs:
            if inp not in expr_by_id:
                raise ValueError(f"Node '{node.node_id}' references unknown input '{inp}'.")
            resolved_inputs.append(expr_by_id[inp])

        if norm_op in _COMMUTATIVE_OPS:
            resolved_inputs.sort()

        sorted_params = sorted((k, v) for k, v in node.params)
        node_repr = json.dumps(
            {"op": norm_op, "in": resolved_inputs, "params": sorted_params},
            sort_keys=True,
            separators=(",", ":"),
        )
        expr_by_id[node.node_id] = node_repr

    target_id = output_node_id or nodes[-1].node_id
    if target_id not in expr_by_id:
        raise ValueError(f"Output node '{target_id}' not found in plan.")

    return hashlib.sha256(expr_by_id[target_id].encode("utf-8")).hexdigest()


def prediction_signature_hash(values: np.ndarray, decimals: int = 4) -> str:
    """Compute Fitted-Prediction Identity invariant to positive/negative affine scaling.

    Catches linear-span or rank-identical duplicates (e.g. ``2 * x + 5`` or ``-x``)
    before they consume hypothesis-testing multiplicity budget.
    """
    arr = np.asarray(values, dtype=float).ravel()
    nan_mask = ~np.isfinite(arr)
    finite_vals = arr[~nan_mask]

    if finite_vals.size == 0 or np.ptp(finite_vals) < 1e-12:
        payload = {"constant": True, "nan_positions": np.flatnonzero(nan_mask).tolist()}
        return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()

    # Rank-normalize finite entries to [0, 1] and orient sign canonically so f and -f match
    ranks = (rankdata(finite_vals, method="average") - 1.0) / max(1.0, finite_vals.size - 1.0)
    centered = ranks - 0.5
    # Canonical orientation: first non-zero centered element must be positive
    nz = np.flatnonzero(np.abs(centered) > 1e-9)
    if nz.size > 0 and centered[nz[0]] < 0:
        centered = -centered

    quantized = np.round(centered, decimals=decimals).tolist()
    payload = {
        "q": quantized,
        "nan_idx": np.flatnonzero(nan_mask).tolist(),
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
