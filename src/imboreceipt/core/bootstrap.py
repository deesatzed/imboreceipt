"""Paired entity/cluster bootstrap inference for rare-event frontier comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True, slots=True)
class NadeauBengioResult:
    """Variance-corrected resampling t-test (Nadeau & Bengio, 2003, Machine Learning)."""

    mean_delta: float
    sample_sd: float
    corrected_se: float
    ci95_low: float
    ci95_high: float
    t_stat: float
    df: int
    p_value_one_sided: float
    p_value_two_sided: float


def nadeau_bengio_ttest(
    deltas: list[float] | np.ndarray,
    n_test: int,
    n_train: int,
    *,
    disjoint_test_blocks: bool = False,
) -> NadeauBengioResult:
    """Compute Nadeau & Bengio (2003) overlap-corrected t-statistic across K splits.

    When K splits are drawn from a finite dataset, naive ``sd / sqrt(K)`` underestimates SE
    due to shared observations. Nadeau & Bengio replace ``1 / K`` with ``1 / K + n_test / n_train``
    (or ``1 / K`` when test blocks are strictly disjoint chronological eras).
    """
    arr = np.asarray(deltas, dtype=float).ravel()
    k = int(arr.size)
    mean_d = float(np.mean(arr)) if k > 0 else 0.0
    if k < 2:
        return NadeauBengioResult(
            mean_delta=mean_d,
            sample_sd=0.0,
            corrected_se=0.0,
            ci95_low=mean_d,
            ci95_high=mean_d,
            t_stat=0.0,
            df=0,
            p_value_one_sided=1.0,
            p_value_two_sided=1.0,
        )

    s2 = float(np.var(arr, ddof=1))
    sd = float(np.sqrt(s2))
    overlap_ratio = 0.0 if disjoint_test_blocks else float(n_test) / max(1.0, float(n_train))
    se_nb = float(np.sqrt((1.0 / float(k) + overlap_ratio) * s2))
    df = k - 1
    if se_nb <= 1e-15:
        p_one = 0.0 if mean_d > 0 else 1.0
        p_two = 0.0 if abs(mean_d) > 0 else 1.0
        return NadeauBengioResult(
            mean_delta=mean_d,
            sample_sd=sd,
            corrected_se=0.0,
            ci95_low=mean_d,
            ci95_high=mean_d,
            t_stat=0.0,
            df=df,
            p_value_one_sided=p_one,
            p_value_two_sided=p_two,
        )

    t_val = float(mean_d / se_nb)
    t_crit = float(student_t.ppf(0.975, df=df))
    p_one = float(student_t.sf(t_val, df=df))
    p_two = float(2.0 * student_t.sf(abs(t_val), df=df))
    return NadeauBengioResult(
        mean_delta=mean_d,
        sample_sd=sd,
        corrected_se=se_nb,
        ci95_low=mean_d - t_crit * se_nb,
        ci95_high=mean_d + t_crit * se_nb,
        t_stat=t_val,
        df=df,
        p_value_one_sided=p_one,
        p_value_two_sided=p_two,
    )


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    """Step-down Holm-Bonferroni family-wise error rate (FWER) adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    pv = np.clip(np.asarray(p_values, dtype=float), 0.0, 1.0)
    order = np.argsort(pv)
    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank_0, orig_i in enumerate(order):
        scaled = float((m - rank_0) * pv[orig_i])
        running_max = max(running_max, min(1.0, scaled))
        adj_sorted[rank_0] = running_max
    out = np.empty(m, dtype=float)
    for rank_0, orig_i in enumerate(order):
        out[orig_i] = adj_sorted[rank_0]
    return out.tolist()


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
