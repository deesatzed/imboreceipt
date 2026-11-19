"""Resampling, equivalence (TOST), meta-analytic, and cluster bootstrap inference utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t


@dataclass(frozen=True, slots=True)
class NadeauBengioResult:
    """Variance-corrected resampling t-test (Nadeau & Bengio, 2003, Machine Learning)."""

    mean_delta: float
    sample_sd: float
    corrected_se: float
    ci90_low: float
    ci90_high: float
    ci95_low: float
    ci95_high: float
    t_stat: float
    df: int
    p_value_one_sided: float
    p_value_two_sided: float


@dataclass(frozen=True, slots=True)
class TOSTResult:
    """Schuirmann (1987) Two One-Sided Tests (TOST) for equivalence or non-inferiority."""

    mean_delta: float
    corrected_se: float
    margin: float
    ci90_low: float
    ci90_high: float
    df: int
    p_noninferiority: float
    p_tost_equivalence: float
    noninferior_at_05: bool
    equivalent_at_05: bool


@dataclass(frozen=True, slots=True)
class MetaAnalysisResult:
    """DerSimonian & Laird (1986) random-effects meta-analysis across studies/cohorts."""

    k_studies: int
    pooled_effect: float
    pooled_se: float
    ci95_low: float
    ci95_high: float
    z_stat: float
    p_value_two_sided: float
    tau2_between: float
    i2_heterogeneity: float


def nadeau_bengio_ttest(
    deltas: list[float] | np.ndarray,
    n_test: int,
    n_train: int,
    *,
    disjoint_test_blocks: bool = False,
) -> NadeauBengioResult:
    """Compute Nadeau & Bengio (2003) overlap-corrected t-statistic across R splits.

    Uses standard ``(1 / R + n_test / n_train) * s^2`` on finite observational datasets
    and ``(1 / R) * s^2`` only for independent Monte Carlo simulation replicates.
    """
    arr = np.asarray(deltas, dtype=float).ravel()
    k = int(arr.size)
    mean_d = float(np.mean(arr)) if k > 0 else 0.0
    if k < 2:
        return NadeauBengioResult(
            mean_delta=mean_d,
            sample_sd=0.0,
            corrected_se=0.0,
            ci90_low=mean_d,
            ci90_high=mean_d,
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
            ci90_low=mean_d,
            ci90_high=mean_d,
            ci95_low=mean_d,
            ci95_high=mean_d,
            t_stat=0.0,
            df=df,
            p_value_one_sided=p_one,
            p_value_two_sided=p_two,
        )

    t_val = float(mean_d / se_nb)
    t_crit90 = float(student_t.ppf(0.95, df=df))
    t_crit95 = float(student_t.ppf(0.975, df=df))
    p_one = float(student_t.sf(t_val, df=df))
    p_two = float(2.0 * student_t.sf(abs(t_val), df=df))
    return NadeauBengioResult(
        mean_delta=mean_d,
        sample_sd=sd,
        corrected_se=se_nb,
        ci90_low=mean_d - t_crit90 * se_nb,
        ci90_high=mean_d + t_crit90 * se_nb,
        ci95_low=mean_d - t_crit95 * se_nb,
        ci95_high=mean_d + t_crit95 * se_nb,
        t_stat=t_val,
        df=df,
        p_value_one_sided=p_one,
        p_value_two_sided=p_two,
    )


def tost_equivalence_and_noninferiority(
    deltas: list[float] | np.ndarray,
    margin: float,
    n_test: int,
    n_train: int,
    *,
    independent_replicates: bool = False,
) -> TOSTResult:
    """Compute Schuirmann (1987) TOST equivalence and non-inferiority p-values with NB SE.

    - Non-inferiority tests ``H_0: mu <= -margin`` vs ``H_1: mu > -margin``.
    - Equivalence tests ``H_0: |mu| >= margin`` vs ``H_1: |mu| < margin`` at alpha = 0.05
      (checking whether the 90% NB confidence interval lies inside ``(-margin, +margin)``).
    """
    if margin <= 0.0:
        raise ValueError("Equivalence/non-inferiority margin must be strictly positive.")
    nb = nadeau_bengio_ttest(
        deltas,
        n_test=n_test,
        n_train=n_train,
        disjoint_test_blocks=independent_replicates,
    )
    if nb.df < 1:
        return TOSTResult(
            mean_delta=nb.mean_delta,
            corrected_se=0.0,
            margin=margin,
            ci90_low=nb.mean_delta,
            ci90_high=nb.mean_delta,
            df=0,
            p_noninferiority=1.0,
            p_tost_equivalence=1.0,
            noninferior_at_05=False,
            equivalent_at_05=False,
        )
    if nb.corrected_se <= 1e-15:
        ni_ok = bool(nb.mean_delta > -margin)
        eq_ok = bool(abs(nb.mean_delta) < margin)
        return TOSTResult(
            mean_delta=nb.mean_delta,
            corrected_se=0.0,
            margin=margin,
            ci90_low=nb.mean_delta,
            ci90_high=nb.mean_delta,
            df=nb.df,
            p_noninferiority=0.0 if ni_ok else 1.0,
            p_tost_equivalence=0.0 if eq_ok else 1.0,
            noninferior_at_05=ni_ok,
            equivalent_at_05=eq_ok,
        )

    t_lower = float((nb.mean_delta + margin) / nb.corrected_se)
    t_upper = float((margin - nb.mean_delta) / nb.corrected_se)
    p_ni = float(student_t.sf(t_lower, df=nb.df))
    p_up = float(student_t.sf(t_upper, df=nb.df))
    p_tost = float(max(p_ni, p_up))
    return TOSTResult(
        mean_delta=nb.mean_delta,
        corrected_se=nb.corrected_se,
        margin=margin,
        ci90_low=nb.ci90_low,
        ci90_high=nb.ci90_high,
        df=nb.df,
        p_noninferiority=p_ni,
        p_tost_equivalence=p_tost,
        noninferior_at_05=bool(p_ni < 0.05),
        equivalent_at_05=bool(p_tost < 0.05),
    )


def random_effects_meta_analysis(
    estimates: list[float] | np.ndarray,
    standard_errors: list[float] | np.ndarray,
) -> MetaAnalysisResult:
    """DerSimonian & Laird (1986) random-effects meta-analysis across cohorts/datasets."""
    y = np.asarray(estimates, dtype=float).ravel()
    se = np.maximum(1e-9, np.asarray(standard_errors, dtype=float).ravel())
    k = int(y.size)
    if k == 0:
        return MetaAnalysisResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    w_fe = 1.0 / (se**2)
    mu_fe = float(np.sum(w_fe * y) / np.sum(w_fe))
    if k == 1:
        se0 = float(se[0])
        z0 = mu_fe / se0
        return MetaAnalysisResult(
            k_studies=1,
            pooled_effect=mu_fe,
            pooled_se=se0,
            ci95_low=mu_fe - 1.96 * se0,
            ci95_high=mu_fe + 1.96 * se0,
            z_stat=z0,
            p_value_two_sided=float(2.0 * norm.sf(abs(z0))),
            tau2_between=0.0,
            i2_heterogeneity=0.0,
        )

    q_stat = float(np.sum(w_fe * ((y - mu_fe) ** 2)))
    c_denom = float(np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe))
    tau2 = max(0.0, (q_stat - float(k - 1)) / max(1e-12, c_denom))
    i2 = max(0.0, (q_stat - float(k - 1)) / max(1e-12, q_stat)) if q_stat > 0 else 0.0

    w_re = 1.0 / (se**2 + tau2)
    mu_re = float(np.sum(w_re * y) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    z_re = mu_re / max(1e-12, se_re)
    p_two = float(2.0 * norm.sf(abs(z_re)))
    return MetaAnalysisResult(
        k_studies=k,
        pooled_effect=mu_re,
        pooled_se=se_re,
        ci95_low=mu_re - 1.96 * se_re,
        ci95_high=mu_re + 1.96 * se_re,
        z_stat=z_re,
        p_value_two_sided=p_two,
        tau2_between=tau2,
        i2_heterogeneity=i2,
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
    """Compute paired bootstrap CIs over independent units/clusters."""
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
        order = np.argsort(inv, kind="mergesort")
        counts = np.bincount(inv, minlength=n_units)
        group_rows = list(np.split(order, np.cumsum(counts)[:-1]))

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
