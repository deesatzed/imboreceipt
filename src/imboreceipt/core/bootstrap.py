"""Paired entity/cluster bootstrap inference for rare-event frontier comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Uncertainty summary for ``metric(score_b) - metric(score_a)``."""

    score_a_point: float
    score_b_point: float
    delta_point: float
    delta_mean: float
    delta_se: float
    ci_low: float
    ci_high: float
    p_value_two_sided: float
    n_units: int
    n_positives: int
    statistically_significant: bool


def paired_cluster_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    groups: np.ndarray | None = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Compute paired bootstrap CIs over independent units/clusters.

    Unlike independent single-model CIs, paired resampling eliminates shared
    sample-to-sample baseline variance while respecting patient/entity clustering.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    sa = np.asarray(score_a, dtype=float).ravel()
    sb = np.asarray(score_b, dtype=float).ravel()
    if not (y.size == sa.size == sb.size):
        raise ValueError("y_true, score_a, and score_b must have identical length.")

    pt_a = float(metric_fn(y, sa))
    pt_b = float(metric_fn(y, sb))
    pt_delta = pt_b - pt_a

    rng = np.random.default_rng(seed)

    if groups is None:
        n_units = int(y.size)
        deltas = np.empty(n_boot, dtype=float)
        valid = 0
        attempts = 0
        while valid < n_boot and attempts < n_boot * 4:
            attempts += 1
            idx = rng.integers(0, n_units, size=n_units)
            yb = y[idx]
            if yb.sum() == 0 or yb.sum() == yb.size:
                continue
            deltas[valid] = float(metric_fn(yb, sb[idx]) - metric_fn(yb, sa[idx]))
            valid += 1
        deltas = deltas[:valid]
    else:
        g_arr = np.asarray(groups)
        unique_groups, inv = np.unique(g_arr, return_inverse=True)
        n_units = int(unique_groups.size)
        group_rows: list[np.ndarray] = [[] for _ in range(n_units)]  # type: ignore[list-item]
        order = np.argsort(inv, kind="mergesort")
        counts = np.bincount(inv, minlength=n_units)
        splits = np.split(order, np.cumsum(counts)[:-1])
        group_rows = list(splits)

        deltas_list: list[float] = []
        attempts = 0
        while len(deltas_list) < n_boot and attempts < n_boot * 4:
            attempts += 1
            sampled_g = rng.integers(0, n_units, size=n_units)
            idx = np.concatenate([group_rows[g] for g in sampled_g])
            yb = y[idx]
            if yb.sum() == 0 or yb.sum() == yb.size:
                continue
            deltas_list.append(float(metric_fn(yb, sb[idx]) - metric_fn(yb, sa[idx])))
        deltas = np.asarray(deltas_list, dtype=float)

    if deltas.size < 10:
        return PairedBootstrapResult(
            score_a_point=pt_a,
            score_b_point=pt_b,
            delta_point=pt_delta,
            delta_mean=pt_delta,
            delta_se=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value_two_sided=1.0,
            n_units=n_units,
            n_positives=int(y.sum()),
            statistically_significant=False,
        )

    ci_low = float(np.quantile(deltas, alpha / 2.0))
    ci_high = float(np.quantile(deltas, 1.0 - alpha / 2.0))
    # Two-sided bootstrap p-value with +1 finite-sample correction
    p_le_zero = (np.sum(deltas <= 0.0) + 1.0) / (deltas.size + 1.0)
    p_ge_zero = (np.sum(deltas >= 0.0) + 1.0) / (deltas.size + 1.0)
    p_two = float(min(1.0, 2.0 * min(p_le_zero, p_ge_zero)))
    sig = bool(ci_low > 0.0 or ci_high < 0.0)

    return PairedBootstrapResult(
        score_a_point=pt_a,
        score_b_point=pt_b,
        delta_point=pt_delta,
        delta_mean=float(np.mean(deltas)),
        delta_se=float(np.std(deltas, ddof=1)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value_two_sided=p_two,
        n_units=n_units,
        n_positives=int(y.sum()),
        statistically_significant=sig,
    )
