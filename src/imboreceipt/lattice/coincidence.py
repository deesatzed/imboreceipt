"""Stage O (Part 1): LIGO-style time-shift & permutation surrogates for joint tail FPR."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.stats import multivariate_normal, norm, rankdata


class TailCoincidenceReport(BaseModel):
    """Empirical and copula-projected negative-tail false-coincidence diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha_a: float
    alpha_b: float
    independent_product_fpr: float
    shifted_background_fpr: float
    shifted_background_se: float
    empirical_joint_fpr: float
    projected_joint_fpr: float
    negative_tail_coincidence_kappa0: float
    upper_tail_gaussian_corr: float
    look_elsewhere_factor: float


def estimate_tail_coincidence_surrogate(
    scores_a_neg: np.ndarray,
    scores_b_neg: np.ndarray,
    *,
    alpha_a: float = 0.01,
    alpha_b: float = 0.01,
    n_shifts: int = 200,
    n_candidate_pairs_tested: int = 1,
    seed: int = 42,
) -> TailCoincidenceReport:
    """Estimate extreme-tail joint false-positive rate ``P(A>=tau_A, B>=tau_B | Y=0)``.

    Borrows the gravitational-wave (LIGO) time-slide / entity-permutation technique:
    1. Circularly shifts ``scores_b_neg`` relative to ``scores_a_neg`` across ``n_shifts``
       non-zero lags to measure the exact finite-sample independent background coincidence
       rate ``shifted_background_fpr`` without parametric assumptions on marginals.
    2. Measures zero-lag vs. shifted coincidence excess ``kappa_0(A, B)`` at diagnostic tail
       level ``max(alpha, 0.05)`` where sufficient negative counts exist, plus normal-score
       tail correlation ``rho_tail``, to project true joint FPR down to ``10^-4`` regimes
       where raw empirical counts have high variance.
    """
    sa = np.asarray(scores_a_neg, dtype=float).ravel()
    sb = np.asarray(scores_b_neg, dtype=float).ravel()
    if sa.size != sb.size or sa.size < 20:
        ind = float(alpha_a * alpha_b)
        return TailCoincidenceReport(
            alpha_a=alpha_a,
            alpha_b=alpha_b,
            independent_product_fpr=ind,
            shifted_background_fpr=ind,
            shifted_background_se=0.0,
            empirical_joint_fpr=ind,
            projected_joint_fpr=ind,
            negative_tail_coincidence_kappa0=1.0,
            upper_tail_gaussian_corr=0.0,
            look_elsewhere_factor=1.0,
        )

    m = sa.size
    tau_a = float(np.quantile(sa, 1.0 - alpha_a))
    tau_b = float(np.quantile(sb, 1.0 - alpha_b))

    hit_a = sa >= tau_a
    hit_b = sb >= tau_b
    emp_joint = float(np.mean(hit_a & hit_b))

    # LIGO circular time-shifts (lags >= max(5, m//25) to break local autocorrelation)
    min_lag = max(5, m // 25)
    max_lag = max(min_lag + 1, m - min_lag)
    rng = np.random.default_rng(seed)
    lags = rng.integers(min_lag, max_lag, size=n_shifts)

    shifted_rates = np.empty(n_shifts, dtype=float)
    for idx, lag in enumerate(lags):
        sb_shift = np.roll(hit_b, int(lag))
        shifted_rates[idx] = float(np.mean(hit_a & sb_shift))

    bg_fpr = float(np.mean(shifted_rates))
    bg_se = float(np.std(shifted_rates, ddof=1) / np.sqrt(n_shifts))

    # Diagnostic tail coincidence ratio kappa_0 at resolvable tail (at least ~5% per marginal)
    diag_alpha = max(alpha_a, alpha_b, min(0.10, max(0.04, 15.0 / m)))
    da_hit = sa >= float(np.quantile(sa, 1.0 - diag_alpha))
    db_hit = sb >= float(np.quantile(sb, 1.0 - diag_alpha))
    zero_lag_diag = float(np.mean(da_hit & db_hit))

    diag_shifts = np.empty(min(100, n_shifts), dtype=float)
    for idx in range(diag_shifts.size):
        diag_shifts[idx] = float(np.mean(da_hit & np.roll(db_hit, int(lags[idx]))))
    bg_diag = max(1e-6, float(np.mean(diag_shifts)))
    kappa_0 = float((zero_lag_diag + 0.5 / m) / (bg_diag + 0.5 / m))

    # Normal-score rank transformation to project bivariate extreme-tail FPR
    ua = rankdata(sa, method="average") / (m + 1.0)
    ub = rankdata(sb, method="average") / (m + 1.0)
    za = norm.ppf(ua)
    zb = norm.ppf(ub)
    # Focus correlation on upper half of negative distribution to capture upper-tail dependence
    upper_mask = (za > 0.0) | (zb > 0.0)
    if int(np.sum(upper_mask)) >= 15:
        rho_mat = np.corrcoef(za[upper_mask], zb[upper_mask])
        rho_tail = float(np.clip(rho_mat[0, 1], -0.98, 0.98)) if np.isfinite(rho_mat[0, 1]) else 0.0
    else:
        rho_tail = float(np.clip(np.corrcoef(za, zb)[0, 1], -0.98, 0.98))

    # Exact bivariate normal survival probability P(Z_a >= z_a, Z_b >= z_b; rho_tail)
    # Symmetry of centered bivariate normal: P(Z_a >= c_a, Z_b >= c_b) = P(Z_a <= -c_a, Z_b <= -c_b)
    za_cut = float(norm.ppf(1.0 - alpha_a))
    zb_cut = float(norm.ppf(1.0 - alpha_b))
    cov = np.array([[1.0, rho_tail], [rho_tail, 1.0]])
    copula_joint = float(
        multivariate_normal.cdf(np.array([-za_cut, -zb_cut]), mean=np.zeros(2), cov=cov)
    )
    copula_joint = float(np.clip(copula_joint, 0.0, min(alpha_a, alpha_b)))

    # Shrink empirical toward copula/shifted estimate when expected counts m * copula_joint < 10
    expected_counts = m * max(copula_joint, alpha_a * alpha_b)
    weight_emp = float(expected_counts / (expected_counts + 8.0))
    projected = float(weight_emp * emp_joint + (1.0 - weight_emp) * copula_joint)

    # Particle-physics "look-elsewhere" factor for testing K candidate pairs
    k_pairs = max(1, int(n_candidate_pairs_tested))
    look_elsewhere = float(1.0 + np.log(k_pairs)) if k_pairs > 1 else 1.0

    return TailCoincidenceReport(
        alpha_a=alpha_a,
        alpha_b=alpha_b,
        independent_product_fpr=float(alpha_a * alpha_b),
        shifted_background_fpr=bg_fpr,
        shifted_background_se=bg_se,
        empirical_joint_fpr=emp_joint,
        projected_joint_fpr=projected,
        negative_tail_coincidence_kappa0=kappa_0,
        upper_tail_gaussian_corr=rho_tail,
        look_elsewhere_factor=look_elsewhere,
    )
