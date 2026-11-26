"""Two-phase Neyman-stratified pilot designs and sequential conditional channel discovery.

Grounds budgeted feature-channel discovery in two-phase epidemiologic sampling theory
(Horvitz & Thompson, 1952; Breslow & Wellner, 2007; Breslow et al., 2009; Lumley, 2010;
McIsaac & Cook, 2014; Tao, Zeng & Lin, 2020) and cross-fitted nonparametric variable
importance (Williamson et al., 2021; Covert et al., 2023; Gadgil et al., 2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, Ridge

from imboreceipt.core.bootstrap import holm_bonferroni_adjust
from imboreceipt.core.types import SemanticType
from imboreceipt.discover.selector import CandidateChannelSpec, benjamini_hochberg_mask
from imboreceipt.discover.strata import make_leak_free_cv_splits
from imboreceipt.frontier.baselines import TabularPreprocessor

_PROB_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class TwoPhasePilot:
    """Known-probability two-phase pilot sample over causal-valid Phase-1 training records."""

    design: str
    sampled_indices: np.ndarray
    stratum_ids_sampled: np.ndarray
    inclusion_probs: np.ndarray
    ht_weights: np.ndarray
    stratum_pop_counts: dict[int, int]
    stratum_sample_counts: dict[int, int]
    n_phase1_valid: int
    n_pilot: int
    pilot_fraction_of_valid: float


@dataclass(frozen=True, slots=True)
class SequentialChannelStep:
    """Audit record for one forward selection step of sequential conditional discovery."""

    step: int
    selected_channel_id: str | None
    gain_nats: float
    se_total: float
    se_phase1: float
    se_phase2: float
    z_stat: float
    p_raw: float
    p_adjusted: float
    lcb_95: float
    cost: float
    stopped_by_error_control: bool


@dataclass(frozen=True, slots=True)
class DiscoverySelectionResult:
    """Complete output of a channel discovery algorithm on a candidate pool."""

    method: str
    pilot_design: str
    pilot_fraction: float
    pilot_channel_accesses: int
    selected_channels: list[CandidateChannelSpec]
    selected_ids: list[str]
    total_cost_spent: float
    step_trace: list[SequentialChannelStep] = field(default_factory=list)
    initial_scores: dict[str, float] = field(default_factory=dict)
    initial_ses: dict[str, float] = field(default_factory=dict)
    initial_pvals: dict[str, float] = field(default_factory=dict)
    initial_pospart_scores: dict[str, float] = field(default_factory=dict)


def _waterfill_constrained_neyman(
    pop_counts: np.ndarray,
    dispersions: np.ndarray,
    target_total: int,
    min_per_stratum: int = 2,
) -> np.ndarray:
    """Solve exact constrained Neyman allocation ``min sum_h N_h^2 (1 - n_h/N_h) s_h^2 / n_h``.

    Subject to ``sum_h n_h = target_total`` and ``min(min_per_stratum, N_h) <= n_h <= N_h``
    via iterative boundary clamping (Wright, 2017; Tao, Zeng & Lin, 2020).
    """
    h_count = int(pop_counts.size)
    total_pop = int(np.sum(pop_counts))
    if target_total >= total_pop:
        return pop_counts.astype(int).copy()

    lo = np.minimum(pop_counts, min_per_stratum).astype(int)
    if int(np.sum(lo)) >= target_total:
        # Budget smaller than minimum floors: allocate at least 1 per non-empty stratum first
        lo1 = np.minimum(pop_counts, 1).astype(int)
        if int(np.sum(lo1)) >= target_total:
            alloc = np.zeros(h_count, dtype=int)
            order = np.argsort(-(pop_counts * dispersions))
            for idx in order[:target_total]:
                alloc[idx] = 1
            return alloc
        lo = lo1

    weights = pop_counts.astype(float) * np.maximum(1e-6, dispersions.astype(float))
    fixed = np.zeros(h_count, dtype=bool)
    alloc_f = np.zeros(h_count, dtype=float)

    for _ in range(h_count * 2 + 4):
        free = ~fixed
        if not np.any(free):
            break
        rem_budget = float(target_total) - float(np.sum(alloc_f[fixed]))
        w_free_sum = float(np.sum(weights[free]))
        if w_free_sum <= 0.0 or rem_budget <= 0.0:
            alloc_f[free] = lo[free].astype(float)
            break
        prop = rem_budget * (weights[free] / w_free_sum)
        over = prop > pop_counts[free]
        under = prop < lo[free]
        if not np.any(over) and not np.any(under):
            alloc_f[free] = prop
            break
        free_idx = np.flatnonzero(free)
        if np.any(over):
            ov_idx = free_idx[over]
            alloc_f[ov_idx] = pop_counts[ov_idx].astype(float)
            fixed[ov_idx] = True
        if np.any(under):
            un_idx = free_idx[under]
            alloc_f[un_idx] = lo[un_idx].astype(float)
            fixed[un_idx] = True

    alloc = np.clip(np.floor(alloc_f).astype(int), lo, pop_counts)
    diff = int(target_total - int(np.sum(alloc)))
    if diff > 0:
        frac = alloc_f - np.floor(alloc_f)
        # Prioritize strata with slack < N_h by fractional remainder + Neyman weight
        order = np.argsort(-(frac + 1e-6 * weights / max(1.0, float(np.max(weights)))))
        for idx in order:
            if diff <= 0:
                break
            if alloc[idx] < pop_counts[idx]:
                alloc[idx] += 1
                diff -= 1
        while diff > 0:
            slack = np.flatnonzero(alloc < pop_counts)
            if slack.size == 0:
                break
            best = slack[int(np.argmax(weights[slack] / np.maximum(1, alloc[slack])))]
            alloc[best] += 1
            diff -= 1
    elif diff < 0:
        while diff < 0:
            reducible = np.flatnonzero(alloc > lo)
            if reducible.size == 0:
                break
            worst = reducible[int(np.argmin(weights[reducible] / np.maximum(1, alloc[reducible])))]
            alloc[worst] -= 1
            diff += 1

    return alloc


def sample_two_phase_pilot(
    y_train: np.ndarray,
    oof_base_prob: np.ndarray,
    oof_valid_mask: np.ndarray | None = None,
    *,
    pilot_fraction: float = 0.15,
    design: str = "neyman",
    n_score_bins: int = 8,
    seed: int = 42,
) -> TwoPhasePilot:
    """Draw a known-probability Phase-2 pilot from causal-valid Phase-1 training records.

    Supported ``design`` values:
    - ``"neyman"``: Constrained Neyman-optimal allocation across joint ``(Y, score_bin)`` strata
      using first-order log-loss dispersion ``|y - eta_b| + 0.25 * eta_b * (1 - eta_b)``.
    - ``"srs"``: Simple random sampling without replacement of size ``n_pilot``.
    - ``"bcc"``: Balanced outcome-only case-control sampling of size ``n_pilot``.
    - ``"full"``: Census of 100% of causal-valid Phase-1 records (``w_i = 1.0``).
    """
    y = np.asarray(y_train, dtype=int).ravel()
    p0 = np.clip(np.asarray(oof_base_prob, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
    valid = (
        np.ones(y.size, dtype=bool)
        if oof_valid_mask is None
        else np.asarray(oof_valid_mask, dtype=bool).ravel()
    )
    valid_idx = np.flatnonzero(valid)
    n_valid = int(valid_idx.size)
    if n_valid == 0:
        raise ValueError("No causal-valid Phase-1 training observations available.")

    if design == "full" or pilot_fraction >= 0.999:
        ones = np.ones(n_valid, dtype=float)
        zeros_s = np.zeros(n_valid, dtype=int)
        return TwoPhasePilot(
            design="full",
            sampled_indices=valid_idx,
            stratum_ids_sampled=zeros_s,
            inclusion_probs=ones.copy(),
            ht_weights=ones,
            stratum_pop_counts={0: n_valid},
            stratum_sample_counts={0: n_valid},
            n_phase1_valid=n_valid,
            n_pilot=n_valid,
            pilot_fraction_of_valid=1.0,
        )

    rng = np.random.default_rng(seed)
    n_pilot = int(np.clip(round(float(pilot_fraction) * n_valid), min(40, n_valid), n_valid))

    if design == "srs":
        chosen_local = np.sort(rng.choice(n_valid, size=n_pilot, replace=False))
        pi_val = float(n_pilot) / float(n_valid)
        w_val = float(n_valid) / float(n_pilot)
        return TwoPhasePilot(
            design="srs",
            sampled_indices=valid_idx[chosen_local],
            stratum_ids_sampled=np.zeros(n_pilot, dtype=int),
            inclusion_probs=np.full(n_pilot, pi_val, dtype=float),
            ht_weights=np.full(n_pilot, w_val, dtype=float),
            stratum_pop_counts={0: n_valid},
            stratum_sample_counts={0: n_pilot},
            n_phase1_valid=n_valid,
            n_pilot=n_pilot,
            pilot_fraction_of_valid=pi_val,
        )

    y_v = y[valid_idx]
    p_v = p0[valid_idx]

    if design == "bcc":
        pos_loc = np.flatnonzero(y_v == 1)
        neg_loc = np.flatnonzero(y_v == 0)
        n1 = min(int(pos_loc.size), max(1, n_pilot // 2))
        n0 = min(int(neg_loc.size), n_pilot - n1)
        if n1 + n0 < n_pilot and int(pos_loc.size) > n1:
            n1 = min(int(pos_loc.size), n_pilot - n0)
        s1 = np.sort(rng.choice(pos_loc, size=n1, replace=False)) if n1 > 0 else np.empty(0, int)
        s0 = np.sort(rng.choice(neg_loc, size=n0, replace=False)) if n0 > 0 else np.empty(0, int)
        comb = np.concatenate([s0, s1])
        order = np.argsort(comb)
        comb = comb[order]
        h_ids = y_v[comb].astype(int)
        pop_map = {0: int(neg_loc.size), 1: int(pos_loc.size)}
        smp_map = {0: int(n0), 1: int(n1)}
        pi_arr = np.array([smp_map[int(h)] / max(1, pop_map[int(h)]) for h in h_ids], dtype=float)
        w_arr = 1.0 / pi_arr
        return TwoPhasePilot(
            design="bcc",
            sampled_indices=valid_idx[comb],
            stratum_ids_sampled=h_ids,
            inclusion_probs=pi_arr,
            ht_weights=w_arr,
            stratum_pop_counts=pop_map,
            stratum_sample_counts=smp_map,
            n_phase1_valid=n_valid,
            n_pilot=int(comb.size),
            pilot_fraction_of_valid=float(comb.size) / float(n_valid),
        )

    if design != "neyman":
        raise ValueError(f"Unsupported two-phase design: {design}")

    # Partition Phase-1 OOF baseline probabilities into quantile score bins
    quantiles = np.linspace(0.0, 1.0, n_score_bins + 1)
    edges = np.unique(np.quantile(logit(p_v), quantiles))
    if edges.size <= 2:
        bins = np.zeros(n_valid, dtype=int)
        n_bins_eff = 1
    else:
        bins = np.clip(np.digitize(logit(p_v), edges[1:-1], right=False), 0, edges.size - 2)
        n_bins_eff = int(edges.size - 1)

    # Joint strata h = y * n_bins_eff + b, keeping only non-empty strata
    raw_h = y_v * n_bins_eff + bins
    uniq_h = np.unique(raw_h)
    pop_counts = np.empty(uniq_h.size, dtype=int)
    dispersions = np.empty(uniq_h.size, dtype=float)
    stratum_members: list[np.ndarray] = []

    for idx_u, h_code in enumerate(uniq_h):
        loc_m = np.flatnonzero(raw_h == h_code)
        stratum_members.append(loc_m)
        pop_counts[idx_u] = int(loc_m.size)
        yv_h = float(int(h_code) // n_bins_eff)
        eta_bar = float(np.mean(p_v[loc_m]))
        # First-order conditional log-loss Taylor dispersion scale in joint stratum (y, b)
        dispersions[idx_u] = abs(yv_h - eta_bar) + 0.25 * eta_bar * (1.0 - eta_bar) + 1e-3

    alloc = _waterfill_constrained_neyman(pop_counts, dispersions, target_total=n_pilot)

    sampled_loc_parts: list[np.ndarray] = []
    sampled_h_parts: list[np.ndarray] = []
    pop_dict: dict[int, int] = {}
    smp_dict: dict[int, int] = {}

    for idx_u, n_take in enumerate(alloc):
        if n_take <= 0:
            continue
        members = stratum_members[idx_u]
        n_pop_h = int(pop_counts[idx_u])
        chosen_h = (
            members.copy()
            if n_take >= n_pop_h
            else np.sort(rng.choice(members, size=int(n_take), replace=False))
        )
        sampled_loc_parts.append(chosen_h)
        sampled_h_parts.append(np.full(chosen_h.size, idx_u, dtype=int))
        pop_dict[idx_u] = n_pop_h
        smp_dict[idx_u] = int(chosen_h.size)

    all_loc = np.concatenate(sampled_loc_parts)
    all_h = np.concatenate(sampled_h_parts)
    ord_i = np.argsort(all_loc)
    all_loc = all_loc[ord_i]
    all_h = all_h[ord_i]

    pi_arr = np.array([smp_dict[int(h)] / float(pop_dict[int(h)]) for h in all_h], dtype=float)
    w_arr = 1.0 / pi_arr

    return TwoPhasePilot(
        design="neyman",
        sampled_indices=valid_idx[all_loc],
        stratum_ids_sampled=all_h,
        inclusion_probs=pi_arr,
        ht_weights=w_arr,
        stratum_pop_counts=pop_dict,
        stratum_sample_counts=smp_dict,
        n_phase1_valid=n_valid,
        n_pilot=int(all_loc.size),
        pilot_fraction_of_valid=float(all_loc.size) / float(n_valid),
    )


def two_phase_variance_decomposition(
    psi_diff: np.ndarray,
    ht_weights: np.ndarray,
    stratum_ids: np.ndarray,
    stratum_pop_counts: dict[int, int],
    stratum_sample_counts: dict[int, int],
    groups: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    """Compute untruncated HT point estimate and two-phase SE (Phase-1 + Phase-2 components).

    Implements the exact two-phase law-of-total-variance decomposition (Lumley, 2010, Ch. 9;
    Sarndal, Swensson & Wretman, 1992, Result 12.2.1; Williamson et al., 2021):
      Var(G_hat) = V_I (Phase-1 superpopulation variance)
                 + V_II (Phase-2 stratified without-replacement sampling variance).

    Returns ``(g_hat, se_total, se_phase1, se_phase2)``.
    """
    psi = np.asarray(psi_diff, dtype=float).ravel()
    w = np.asarray(ht_weights, dtype=float).ravel()
    h_arr = np.asarray(stratum_ids, dtype=int).ravel()
    n_obs = int(psi.size)
    if n_obs < 3:
        return 0.0, 1.0, 1.0, 0.0

    w_tot = max(1.0, float(np.sum(w)))
    g_hat = float(np.sum(w * psi) / w_tot)
    resid = psi - g_hat

    # 1. Phase-1 superpopulation variance estimator V_I
    dof_obs = float(n_obs) / max(1.0, float(n_obs - 1))
    v1_iid = dof_obs * float(np.sum(w * (resid**2)))
    if groups is not None and groups.size == n_obs:
        _, inv = np.unique(groups, return_inverse=True)
        cl_ht_sum = np.bincount(inv, weights=w * resid)
        n_cl = int(cl_ht_sum.size)
        dof_cl = float(n_cl) / max(1.0, float(n_cl - 1))
        # Horvitz-Thompson cluster sum square minus diagonal self-weight excess w_i*(w_i - 1)
        cl_num = dof_cl * float(np.sum(cl_ht_sum**2) - np.sum(w * (w - 1.0) * (resid**2)))
        v_phase1 = max(v1_iid, cl_num) / (w_tot**2)
    else:
        v_phase1 = v1_iid / (w_tot**2)

    # 2. Phase-2 stratified finite-population sampling variance V_II
    v_phase2_num = 0.0
    for h_id in np.unique(h_arr):
        mask_h = h_arr == h_id
        m_eval = int(np.sum(mask_h))
        if m_eval < 2:
            continue
        n_pop_h = int(stratum_pop_counts.get(int(h_id), m_eval))
        n_smp_h = int(stratum_sample_counts.get(int(h_id), m_eval))
        # Adjust effective stratum population to the evaluated causal subset
        eval_frac = float(m_eval) / max(1.0, float(n_smp_h))
        n_pop_eff = max(float(m_eval), float(n_pop_h) * eval_frac)
        f_h = min(1.0, float(m_eval) / n_pop_eff)
        if f_h >= 1.0 - 1e-9:
            continue
        r_h = resid[mask_h]
        s2_h = float(np.var(r_h, ddof=1))
        v_phase2_num += (n_pop_eff**2) * (1.0 - f_h) * (s2_h / float(m_eval))

    v_phase2 = v_phase2_num / (w_tot**2)

    # Boundary-degeneracy stabilization margin (Williamson et al., 2021, JASA)
    eps_var = (0.015 / np.sqrt(max(10.0, float(n_obs)))) ** 2
    v_tot = max(eps_var, v_phase1 + v_phase2)
    return (
        g_hat,
        float(np.sqrt(v_tot)),
        float(np.sqrt(max(0.0, v_phase1))),
        float(np.sqrt(max(0.0, v_phase2))),
    )


def _bernoulli_loglik(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, _PROB_EPS, 1.0 - _PROB_EPS)
    return y * np.log(p) + (1.0 - y) * np.log(1.0 - p)


def _source_to_transformed_indices(prep: TabularPreprocessor) -> dict[str, list[int]]:
    """Map each original DataFrame column name to all transformed column indices in ``prep``."""
    mapping: dict[str, list[int]] = {}
    offset = 0
    for col in prep.num_cols_:
        mapping.setdefault(col, []).append(offset)
        offset += 1
    for col in prep.bin_cols_:
        mapping.setdefault(col, []).append(offset)
        offset += 1
    for col in prep.ord_cols_:
        mapping.setdefault(col, []).append(offset)
        offset += 1
    if prep.nom_cols_ and prep._nom_enc is not None:
        infreq_list = getattr(prep._nom_enc, "infrequent_categories_", None)
        for i, src_col in enumerate(prep.nom_cols_):
            cats = prep._nom_enc.categories_[i]
            infreq = infreq_list[i] if infreq_list is not None else None
            n_out = (len(cats) - len(infreq) + 1) if infreq is not None else len(cats)
            for _ in range(n_out):
                mapping.setdefault(src_col, []).append(offset)
                offset += 1
    for col in prep.miss_cols_:
        mapping.setdefault(col, []).append(offset)
        offset += 1
    return mapping


def _prepare_fold_matrices(
    df: pd.DataFrame,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    semantic_types: dict[str, SemanticType] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pre-fit leak-free fold matrices once per block to avoid redundant DataFrame encoding."""
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for tr_i, va_i in cv_splits:
        if np.unique(y[tr_i]).size < 2:
            continue
        prep = TabularPreprocessor(semantic_types=semantic_types)
        X_tr = prep.fit_transform(df.iloc[tr_i])
        X_va = prep.transform(df.iloc[va_i])
        out.append((tr_i, va_i, X_tr, X_va))
    return out


def _compute_breslow_cain_pilot_offset(
    y_pilot: np.ndarray,
    init_logit_pilot: np.ndarray,
    ht_weights_pilot: np.ndarray,
    n_bins: int = 8,
) -> np.ndarray:
    """Compute Breslow & Cain (1988) case-control log-odds offset delta_P(b) per baseline bin.

    Within score bin b, pi_P(y, b) = n_{y,b} / N_{y,b} where N_{y,b} = sum_{i in P: Y=y, B=b} w_i.
    Then delta_P(b) = log(pi_P(1, b) / pi_P(0, b)), so unweighted pilot Bernoulli fitting anchored
    at logit(p_1) + delta_P(b) targets exact population log-odds after subtracting delta_P(b).
    """
    n_p = int(y_pilot.size)
    if n_p < 4:
        return np.zeros(n_p, dtype=float)
    edges = np.unique(np.quantile(init_logit_pilot, np.linspace(0.0, 1.0, n_bins + 1)))
    bins = (
        np.zeros(n_p, dtype=int)
        if edges.size <= 2
        else np.clip(np.digitize(init_logit_pilot, edges[1:-1], right=False), 0, edges.size - 2)
    )
    delta = np.zeros(n_p, dtype=float)
    # Global fallback offset
    pos_m_all = y_pilot == 1
    neg_m_all = y_pilot == 0
    pi1_all = float(np.sum(pos_m_all)) / max(1.0, float(np.sum(ht_weights_pilot[pos_m_all])))
    pi0_all = float(np.sum(neg_m_all)) / max(1.0, float(np.sum(ht_weights_pilot[neg_m_all])))
    global_delta = float(np.log(max(1e-6, pi1_all) / max(1e-6, pi0_all)))

    for b_val in np.unique(bins):
        bm = bins == b_val
        pos_b = bm & (y_pilot == 1)
        neg_b = bm & (y_pilot == 0)
        n1_b = int(np.sum(pos_b))
        n0_b = int(np.sum(neg_b))
        if n1_b >= 1 and n0_b >= 1:
            pop1_b = max(float(n1_b), float(np.sum(ht_weights_pilot[pos_b])))
            pop0_b = max(float(n0_b), float(np.sum(ht_weights_pilot[neg_b])))
            pi1_b = float(n1_b) / pop1_b
            pi0_b = float(n0_b) / pop0_b
            delta[bm] = float(np.clip(np.log(max(1e-6, pi1_b) / max(1e-6, pi0_b)), -5.0, 5.0))
        else:
            delta[bm] = global_delta
    return delta


def _fit_oof_offset_gbdt_mats(
    fold_mats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    y: np.ndarray,
    sample_weight: np.ndarray,
    init_logit: np.ndarray,
    seed: int,
    breslow_cain_delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit shallow regularized offset-anchored GBDT + fold-internal HT Platt calibration."""
    n = int(y.size)
    p_base = expit(init_logit)
    oof_p = p_base.copy()
    valid_mask = np.zeros(n, dtype=bool)
    use_bc = breslow_cain_delta is not None

    for f_idx, (tr_i, va_i, X_tr, X_va) in enumerate(fold_mats):
        w_tr = sample_weight[tr_i]
        w_tr_norm = w_tr * (float(tr_i.size) / max(1e-9, float(np.sum(w_tr))))
        w_tree = np.ones(tr_i.size, dtype=float) if use_bc else w_tr_norm
        init_tr = init_logit[tr_i] + breslow_cain_delta[tr_i] if use_bc else init_logit[tr_i]

        min_child = max(8, min(25, int(tr_i.size // 25)))
        clf = lgb.LGBMClassifier(
            n_estimators=36,
            learning_rate=0.08,
            num_leaves=15,
            max_depth=4,
            min_child_samples=min_child,
            subsample_for_bin=min(50000, max(2000, int(tr_i.size))),
            max_bin=127,
            reg_alpha=1.0,
            reg_lambda=5.0,
            max_delta_step=1.0,
            random_state=seed + f_idx,
            n_jobs=2,
            verbose=-1,
        )
        clf.fit(X_tr, y[tr_i], sample_weight=w_tree, init_score=init_tr)
        # Subtract Breslow-Cain pilot sampling offset to recover population-scale margins
        tr_margin = init_logit[tr_i] + np.asarray(clf.predict(X_tr, raw_score=True), dtype=float)
        va_margin = init_logit[va_i] + np.asarray(clf.predict(X_va, raw_score=True), dtype=float)

        cal = LogisticRegression(C=2.0, solver="lbfgs", max_iter=200)
        cal.fit(tr_margin.reshape(-1, 1), y[tr_i], sample_weight=w_tr_norm)
        oof_p[va_i] = np.clip(
            cal.predict_proba(va_margin.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS
        )
        valid_mask[va_i] = True

    return oof_p, valid_mask


def discover_sequential_conditional_gain(
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    oof_base_prob: np.ndarray,
    pilot: TwoPhasePilot,
    max_budget: float,
    *,
    base_semantic_types: dict[str, SemanticType] | None = None,
    groups_train: np.ndarray | None = None,
    time_order_train: np.ndarray | None = None,
    sequential: bool = True,
    use_ht_weights: bool = True,
    use_breslow_cain_offset: bool = False,
    allow_lazy_submodular_screen: bool = True,
    multiplicity: str = "holm",
    alpha_select: float = 0.05,
    method_name: str = "neyman_seq_cmi",
    seed: int = 42,
) -> DiscoverySelectionResult:
    """Forward stepwise two-phase conditional log-likelihood discovery with LCB stopping.

    - Uses UNTRUNCATED per-record log-likelihood differences (fixing positive-part bias).
    - Estimates exact two-phase SE (Phase-1 superpopulation + Phase-2 stratified sampling).
    - Updates the conditioning set ``W_t = [X_base, Z_{S_t}]`` sequentially on the SAME pilot
      to eliminate Stage-2-to-Stage-2 redundant twins at zero extra row-access cost.
    - Stops immediately when no remaining channel passes multiplicity-adjusted ``alpha_select``
      (never filling remaining budget with noise!).
    """
    y_all = np.asarray(y_train, dtype=int).ravel()
    idx = pilot.sampled_indices
    y_p = y_all[idx]
    w_p = pilot.ht_weights if use_ht_weights else np.ones(idx.size, dtype=float)
    h_p = pilot.stratum_ids_sampled
    grp_p = groups_train[idx] if groups_train is not None else None
    time_p = time_order_train[idx] if time_order_train is not None else None
    sem_p = dict(base_semantic_types) if base_semantic_types is not None else {}

    cv_splits = make_leak_free_cv_splits(y_p, n_folds=4, groups=grp_p, time_order=time_p, seed=seed)

    curr_df = X_base_train.iloc[idx].reset_index(drop=True)
    Z_pilot_df = Z_pool_train.iloc[idx].reset_index(drop=True)
    base_p_clip = np.clip(oof_base_prob[idx], _PROB_EPS, 1.0 - _PROB_EPS)
    init_logit_0 = logit(base_p_clip)
    bc_delta = (
        _compute_breslow_cain_pilot_offset(y_p, init_logit_0, pilot.ht_weights)
        if use_breslow_cain_offset
        else None
    )

    curr_mats = _prepare_fold_matrices(curr_df, y_p, cv_splits, semantic_types=sem_p)
    cand_mats = {
        c.channel_id: _prepare_fold_matrices(Z_pilot_df[list(c.columns)], y_p, cv_splits)
        for c in candidates
    }

    p_curr, valid_curr = _fit_oof_offset_gbdt_mats(
        curr_mats, y_p, w_p, init_logit_0, seed=seed, breslow_cain_delta=bc_delta
    )
    ll_curr = _bernoulli_loglik(y_p, p_curr)

    remaining = list(candidates)
    selected: list[CandidateChannelSpec] = []
    spent = 0.0
    step_trace: list[SequentialChannelStep] = []
    init_scores: dict[str, float] = {}
    init_ses: dict[str, float] = {}
    init_pvals: dict[str, float] = {}
    init_pospart_scores: dict[str, float] = {}

    # Cache step-1 results for non-sequential mode and optional lazy submodular pruning
    step1_stats: dict[str, tuple[float, float, float, float, float, np.ndarray, np.ndarray]] = {}

    step_num = 0
    while remaining and spent < max_budget - 1e-9:
        step_num += 1
        affordable = [c for c in remaining if spent + c.acquisition_cost <= max_budget + 1e-9]
        if not affordable:
            break

        cand_eval: list[
            tuple[CandidateChannelSpec, float, float, float, float, float, np.ndarray, np.ndarray]
        ] = []

        for cand in affordable:
            if step_num > 1 and not sequential and cand.channel_id in step1_stats:
                g_m, se_t, se_1, se_2, p_raw, p_m, val_m = step1_stats[cand.channel_id]
                cand_eval.append((cand, g_m, se_t, se_1, se_2, p_raw, p_m, val_m))
                continue

            # Lazy submodular screen: reuse Step-1 negative bound when pool is large
            if (
                allow_lazy_submodular_screen
                and step_num > 1
                and sequential
                and len(affordable) > 12
                and cand.channel_id in step1_stats
                and step1_stats[cand.channel_id][0] < -0.002
            ):
                g_m, se_t, se_1, se_2, p_raw, p_m, val_m = step1_stats[cand.channel_id]
                cand_eval.append((cand, g_m, se_t, se_1, se_2, p_raw, p_m, val_m))
                continue

            c_m_folds = cand_mats[cand.channel_id]
            aug_mats = [
                (tr_i, va_i, np.hstack([X_tr, Zc_tr]), np.hstack([X_va, Zc_va]))
                for (tr_i, va_i, X_tr, X_va), (_, _, Zc_tr, Zc_va) in zip(
                    curr_mats, c_m_folds, strict=True
                )
            ]
            # Fit from identical Phase-1 anchor init_logit_0 and seed for matched nested capacity
            p_m, val_m = _fit_oof_offset_gbdt_mats(
                aug_mats, y_p, w_p, init_logit_0, seed=seed, breslow_cain_delta=bc_delta
            )
            ev = valid_curr & val_m
            if int(np.sum(ev)) < 10:
                g_m, se_t, se_1, se_2, p_raw = 0.0, 1.0, 1.0, 0.0, 1.0
                g_pospart = 0.0
            else:
                psi_m = (_bernoulli_loglik(y_p, p_m) - ll_curr)[ev]
                w_ev = w_p[ev] if use_ht_weights else np.ones(int(np.sum(ev)), dtype=float)
                g_pospart = float(
                    np.sum(w_ev * np.maximum(0.0, psi_m)) / max(1.0, float(np.sum(w_ev)))
                )
                if use_ht_weights:
                    g_m, se_t, se_1, se_2 = two_phase_variance_decomposition(
                        psi_m,
                        w_p[ev],
                        h_p[ev],
                        pilot.stratum_pop_counts,
                        pilot.stratum_sample_counts,
                        groups=grp_p[ev] if grp_p is not None else None,
                    )
                else:
                    g_m = float(np.mean(psi_m))
                    se_t = float(
                        np.sqrt(max(1e-8, float(np.var(psi_m, ddof=1)) / float(psi_m.size)))
                    )
                    se_1, se_2 = se_t, 0.0
                z_m = g_m / max(1e-12, se_t)
                p_raw = float(norm.sf(z_m))

            if step_num == 1:
                init_scores[cand.channel_id] = g_m
                init_ses[cand.channel_id] = se_t
                init_pvals[cand.channel_id] = p_raw
                init_pospart_scores[cand.channel_id] = g_pospart
                step1_stats[cand.channel_id] = (g_m, se_t, se_1, se_2, p_raw, p_m, val_m)

            cand_eval.append((cand, g_m, se_t, se_1, se_2, p_raw, p_m, val_m))

        raw_pvals = [row[5] for row in cand_eval]
        if multiplicity == "holm":
            adj_pvals = holm_bonferroni_adjust(raw_pvals)
            passes = [
                (ap < alpha_select) and (row[1] > 0.0)
                for ap, row in zip(adj_pvals, cand_eval, strict=True)
            ]
        elif multiplicity == "bh":
            bh_flags = benjamini_hochberg_mask(raw_pvals, fdr_q=alpha_select)
            # Compute monotone BH q-values for reporting
            m_rem = len(raw_pvals)
            order = np.argsort(raw_pvals)
            q_sorted = np.empty(m_rem, dtype=float)
            run_min = 1.0
            for rk in range(m_rem - 1, -1, -1):
                q_val = min(1.0, raw_pvals[order[rk]] * m_rem / float(rk + 1))
                run_min = min(run_min, q_val)
                q_sorted[rk] = run_min
            adj_pvals = [0.0] * m_rem
            for rk, orig_i in enumerate(order):
                adj_pvals[orig_i] = float(q_sorted[rk])
            passes = [
                bool(fl) and (row[1] > 0.0) for fl, row in zip(bh_flags, cand_eval, strict=True)
            ]
        else:
            adj_pvals = raw_pvals
            passes = [
                (pv < alpha_select) and (row[1] > 0.0)
                for pv, row in zip(raw_pvals, cand_eval, strict=True)
            ]

        passing_indices = [i for i, ok in enumerate(passes) if ok]
        if not passing_indices:
            # Record clean error-controlled stopping step (zero noise channels bought!)
            best_i = int(
                np.argmax([row[1] / max(1e-6, row[0].acquisition_cost) for row in cand_eval])
            )
            b_cand, b_g, b_se, b_s1, b_s2, b_pr, _, _ = cand_eval[best_i]
            step_trace.append(
                SequentialChannelStep(
                    step=step_num,
                    selected_channel_id=None,
                    gain_nats=b_g,
                    se_total=b_se,
                    se_phase1=b_s1,
                    se_phase2=b_s2,
                    z_stat=b_g / max(1e-12, b_se),
                    p_raw=b_pr,
                    p_adjusted=adj_pvals[best_i],
                    lcb_95=b_g - 1.6448536 * b_se,
                    cost=0.0,
                    stopped_by_error_control=True,
                )
            )
            break

        # Pick passing channel maximizing cost-normalized conditional information gain
        best_i = max(
            passing_indices,
            key=lambda i: cand_eval[i][1] / max(1e-6, cand_eval[i][0].acquisition_cost),
        )
        win_cand, win_g, win_se, win_s1, win_s2, win_pr, win_pm, win_valm = cand_eval[best_i]
        selected.append(win_cand)
        spent += win_cand.acquisition_cost
        remaining = [c for c in remaining if c.channel_id != win_cand.channel_id]

        step_trace.append(
            SequentialChannelStep(
                step=step_num,
                selected_channel_id=win_cand.channel_id,
                gain_nats=win_g,
                se_total=win_se,
                se_phase1=win_s1,
                se_phase2=win_s2,
                z_stat=win_g / max(1e-12, win_se),
                p_raw=win_pr,
                p_adjusted=adj_pvals[best_i],
                lcb_95=win_g - 1.6448536 * win_se,
                cost=win_cand.acquisition_cost,
                stopped_by_error_control=False,
            )
        )

        if sequential and remaining and spent < max_budget - 1e-9:
            w_folds = cand_mats[win_cand.channel_id]
            curr_mats = [
                (tr_i, va_i, np.hstack([X_tr, Zw_tr]), np.hstack([X_va, Zw_va]))
                for (tr_i, va_i, X_tr, X_va), (_, _, Zw_tr, Zw_va) in zip(
                    curr_mats, w_folds, strict=True
                )
            ]
            valid_curr = valid_curr & win_valm
            p_curr = win_pm
            ll_curr = _bernoulli_loglik(y_p, p_curr)

    return DiscoverySelectionResult(
        method=method_name,
        pilot_design=pilot.design,
        pilot_fraction=pilot.pilot_fraction_of_valid,
        pilot_channel_accesses=int(pilot.n_pilot * len(candidates)),
        selected_channels=selected,
        selected_ids=[c.channel_id for c in selected],
        total_cost_spent=spent,
        step_trace=step_trace,
        initial_scores=init_scores,
        initial_ses=init_ses,
        initial_pvals=init_pvals,
        initial_pospart_scores=init_pospart_scores,
    )


def discover_competitor_method(
    method: str,
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    oof_base_prob: np.ndarray,
    pilot: TwoPhasePilot,
    max_budget: float,
    *,
    base_semantic_types: dict[str, SemanticType] | None = None,
    groups_train: np.ndarray | None = None,
    time_order_train: np.ndarray | None = None,
    alpha_select: float = 0.05,
    seed: int = 42,
) -> DiscoverySelectionResult:
    """Execute competitive conditional and marginal feature-channel discovery baselines.

    Supported ``method`` values:
    - ``"cpi_pilot"``: Baseline-Stratified Permutation Importance (permute Z_m in X_base strata).
    - ``"chamma_residual_cpi_pilot"``: Full-Conditioning Residualized CPI (Chamma et al., 2023;
      predict Z_m from [X_base, Z_{-m}] on training folds and permute OOF residuals).
    - ``"loco_pilot"``: Leave-One-Channel-Out log-loss importance (Williamson et al., 2021).
    - ``"crt_knockoff_pilot"``: Gaussian-copula Model-X Knockoff / CRT (Candes et al., 2018).
    - ``"unweighted_pilot_no_lcb"`` (alias ``"dime_cmi_pilot"``): Unweighted sequential pilot CMI
      ablation omitting Horvitz-Thompson weights and Holm LCB stopping.
    - ``"global_mi"``: Full-cohort marginal mutual information ``I(Y; Z_m) / c_m``.
    - ``"random"``: Uniform random channel selection within ``max_budget``.
    """
    y_all = np.asarray(y_train, dtype=int).ravel()
    rng = np.random.default_rng(seed)
    sem_p = dict(base_semantic_types) if base_semantic_types is not None else {}

    if method in {"unweighted_pilot_no_lcb", "dime_cmi_pilot"}:
        return discover_sequential_conditional_gain(
            X_base_train,
            Z_pool_train,
            y_all,
            candidates,
            oof_base_prob,
            pilot,
            max_budget,
            base_semantic_types=base_semantic_types,
            groups_train=groups_train,
            time_order_train=time_order_train,
            sequential=True,
            use_ht_weights=False,
            multiplicity="none",
            alpha_select=0.50,
            method_name=method,
            seed=seed,
        )

    if method == "random":
        perm = list(candidates)
        rng.shuffle(perm)
        chosen: list[CandidateChannelSpec] = []
        spent = 0.0
        for c in perm:
            if spent + c.acquisition_cost <= max_budget + 1e-9:
                chosen.append(c)
                spent += c.acquisition_cost
        return DiscoverySelectionResult(
            method="random",
            pilot_design="none",
            pilot_fraction=0.0,
            pilot_channel_accesses=0,
            selected_channels=chosen,
            selected_ids=[c.channel_id for c in chosen],
            total_cost_spent=spent,
        )

    if method == "global_mi":
        # Subsample up to 40k rows for fast kNN MI estimation while counting full-table access
        n_tot = len(X_base_train)
        sub_i = (
            np.arange(n_tot)
            if n_tot <= 40000
            else np.sort(rng.choice(n_tot, size=40000, replace=False))
        )
        all_z_cols = [col for c in candidates for col in c.columns]
        prep = TabularPreprocessor()
        Z_mat = np.nan_to_num(prep.fit_transform(Z_pool_train.iloc[sub_i][all_z_cols]), nan=0.0)
        mi_vec = mutual_info_classif(Z_mat, y_all[sub_i], random_state=seed)
        src_map = _source_to_transformed_indices(prep)
        scores = {
            c.channel_id: float(sum(mi_vec[j] for col in c.columns for j in src_map.get(col, [])))
            for c in candidates
        }
        ranked = sorted(
            candidates,
            key=lambda c: scores[c.channel_id] / max(1e-6, c.acquisition_cost),
            reverse=True,
        )
        chosen = []
        spent = 0.0
        for c in ranked:
            if scores[c.channel_id] > 0.0 and spent + c.acquisition_cost <= max_budget + 1e-9:
                chosen.append(c)
                spent += c.acquisition_cost
        return DiscoverySelectionResult(
            method="global_mi",
            pilot_design="full",
            pilot_fraction=1.0,
            pilot_channel_accesses=int(n_tot * len(candidates)),
            selected_channels=chosen,
            selected_ids=[c.channel_id for c in chosen],
            total_cost_spent=spent,
            initial_scores=scores,
        )

    idx = pilot.sampled_indices
    y_p = y_all[idx]
    w_p = pilot.ht_weights
    h_p = pilot.stratum_ids_sampled
    grp_p = groups_train[idx] if groups_train is not None else None
    time_p = time_order_train[idx] if time_order_train is not None else None
    cv_splits = make_leak_free_cv_splits(y_p, n_folds=4, groups=grp_p, time_order=time_p, seed=seed)

    base_df_p = X_base_train.iloc[idx].reset_index(drop=True).copy()
    init_logit_0 = logit(np.clip(oof_base_prob[idx], _PROB_EPS, 1.0 - _PROB_EPS))
    all_z_cols = [col for c in candidates for col in c.columns]
    Z_df_p = Z_pool_train.iloc[idx][all_z_cols].reset_index(drop=True).copy()
    joint_df_p = pd.concat([base_df_p, Z_df_p], axis=1)

    if method == "cpi_pilot":
        # Conditional Permutation Importance: permute Z_m within baseline score strata
        n_p = int(y_p.size)
        oof_full = np.clip(oof_base_prob[idx], _PROB_EPS, 1.0 - _PROB_EPS)
        valid_f = np.zeros(n_p, dtype=bool)
        # Store per-channel permuted OOF log-likelihood drops
        perm_ll_drops = {c.channel_id: np.zeros(n_p, dtype=float) for c in candidates}

        # Baseline quantile bins on pilot for conditional permutation
        b_edges = np.unique(np.quantile(init_logit_0, np.linspace(0.0, 1.0, 9)))
        b_bins = (
            np.zeros(n_p, dtype=int)
            if b_edges.size <= 2
            else np.clip(np.digitize(init_logit_0, b_edges[1:-1]), 0, b_edges.size - 2)
        )

        for f_idx, (tr_i, va_i) in enumerate(cv_splits):
            if np.unique(y_p[tr_i]).size < 2:
                continue
            prep = TabularPreprocessor(semantic_types=sem_p)
            X_tr = prep.fit_transform(joint_df_p.iloc[tr_i])
            X_va = prep.transform(joint_df_p.iloc[va_i])
            w_tr = w_p[tr_i] * (float(tr_i.size) / max(1e-9, float(np.sum(w_p[tr_i]))))
            clf = lgb.LGBMClassifier(
                n_estimators=55,
                learning_rate=0.07,
                num_leaves=23,
                max_depth=5,
                min_child_samples=max(8, min(25, int(tr_i.size // 25))),
                max_bin=127,
                reg_alpha=1.0,
                reg_lambda=5.0,
                max_delta_step=1.0,
                random_state=seed + f_idx,
                n_jobs=4,
                verbose=-1,
            )
            clf.fit(X_tr, y_p[tr_i], sample_weight=w_tr, init_score=init_logit_0[tr_i])
            tr_m = init_logit_0[tr_i] + np.asarray(clf.predict(X_tr, raw_score=True), dtype=float)
            va_m = init_logit_0[va_i] + np.asarray(clf.predict(X_va, raw_score=True), dtype=float)
            cal = LogisticRegression(C=2.0, solver="lbfgs", max_iter=200)
            cal.fit(tr_m.reshape(-1, 1), y_p[tr_i], sample_weight=w_tr)
            p_va = np.clip(cal.predict_proba(va_m.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS)
            oof_full[va_i] = p_va
            valid_f[va_i] = True
            ll_va_full = _bernoulli_loglik(y_p[va_i], p_va)

            src_map = _source_to_transformed_indices(prep)
            va_bins = b_bins[va_i]
            for cand in candidates:
                c_idxs = [j for col in cand.columns for j in src_map.get(col, [])]
                if not c_idxs:
                    continue
                acc_drop = np.zeros(va_i.size, dtype=float)
                n_perm = 3
                for p_rep in range(n_perm):
                    X_va_perm = X_va.copy()
                    perm_rng = np.random.default_rng(seed + 1009 * f_idx + 37 * p_rep)
                    for b_val in np.unique(va_bins):
                        bm = np.flatnonzero(va_bins == b_val)
                        if bm.size > 1:
                            shuf = perm_rng.permutation(bm)
                            for cj in c_idxs:
                                X_va_perm[bm, cj] = X_va[shuf, cj]
                    m_perm = init_logit_0[va_i] + np.asarray(
                        clf.predict(X_va_perm, raw_score=True), dtype=float
                    )
                    p_perm = np.clip(
                        cal.predict_proba(m_perm.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS
                    )
                    acc_drop += ll_va_full - _bernoulli_loglik(y_p[va_i], p_perm)
                perm_ll_drops[cand.channel_id][va_i] = acc_drop / float(n_perm)

        scores_d: dict[str, float] = {}
        ses_d: dict[str, float] = {}
        pvals_d: dict[str, float] = {}
        for cand in candidates:
            psi_c = perm_ll_drops[cand.channel_id][valid_f]
            g_c, se_c, _, _ = two_phase_variance_decomposition(
                psi_c,
                w_p[valid_f],
                h_p[valid_f],
                pilot.stratum_pop_counts,
                pilot.stratum_sample_counts,
                groups=grp_p[valid_f] if grp_p is not None else None,
            )
            scores_d[cand.channel_id] = g_c
            ses_d[cand.channel_id] = se_c
            pvals_d[cand.channel_id] = float(norm.sf(g_c / max(1e-12, se_c)))

        adj = holm_bonferroni_adjust([pvals_d[c.channel_id] for c in candidates])
        passing = [
            c
            for c, ap in zip(candidates, adj, strict=True)
            if ap < alpha_select and scores_d[c.channel_id] > 0.0
        ]
        passing.sort(
            key=lambda c: scores_d[c.channel_id] / max(1e-6, c.acquisition_cost), reverse=True
        )
        chosen = []
        spent = 0.0
        for c in passing:
            if spent + c.acquisition_cost <= max_budget + 1e-9:
                chosen.append(c)
                spent += c.acquisition_cost
        return DiscoverySelectionResult(
            method="cpi_pilot",
            pilot_design=pilot.design,
            pilot_fraction=pilot.pilot_fraction_of_valid,
            pilot_channel_accesses=int(pilot.n_pilot * len(candidates)),
            selected_channels=chosen,
            selected_ids=[c.channel_id for c in chosen],
            total_cost_spent=spent,
            initial_scores=scores_d,
            initial_ses=ses_d,
            initial_pvals=pvals_d,
        )

    if method == "loco_pilot":
        # Williamson et al. (2021) Leave-One-Channel-Out log-likelihood importance on pilot
        base_mats_l = _prepare_fold_matrices(base_df_p, y_p, cv_splits, semantic_types=sem_p)
        cand_mats_l = [
            (c.channel_id, _prepare_fold_matrices(Z_df_p[list(c.columns)], y_p, cv_splits))
            for c in candidates
        ]
        full_mats = [
            (
                tr_i,
                va_i,
                np.hstack([Xb_tr] + [cm[f_i][2] for _, cm in cand_mats_l]),
                np.hstack([Xb_va] + [cm[f_i][3] for _, cm in cand_mats_l]),
            )
            for f_i, (tr_i, va_i, Xb_tr, Xb_va) in enumerate(base_mats_l)
        ]
        p_full, val_full = _fit_oof_offset_gbdt_mats(full_mats, y_p, w_p, init_logit_0, seed=seed)
        ll_full = _bernoulli_loglik(y_p, p_full)
        scores_d = {}
        ses_d = {}
        pvals_d = {}
        for cand in candidates:
            minus_mats = [
                (
                    tr_i,
                    va_i,
                    np.hstack(
                        [Xb_tr] + [cm[f_i][2] for cid, cm in cand_mats_l if cid != cand.channel_id]
                    ),
                    np.hstack(
                        [Xb_va] + [cm[f_i][3] for cid, cm in cand_mats_l if cid != cand.channel_id]
                    ),
                )
                for f_i, (tr_i, va_i, Xb_tr, Xb_va) in enumerate(base_mats_l)
            ]
            p_minus, val_m = _fit_oof_offset_gbdt_mats(
                minus_mats, y_p, w_p, init_logit_0, seed=seed
            )
            ev = val_full & val_m
            psi_l = (ll_full - _bernoulli_loglik(y_p, p_minus))[ev]
            g_l, se_l, _, _ = two_phase_variance_decomposition(
                psi_l,
                w_p[ev],
                h_p[ev],
                pilot.stratum_pop_counts,
                pilot.stratum_sample_counts,
                groups=grp_p[ev] if grp_p is not None else None,
            )
            scores_d[cand.channel_id] = g_l
            ses_d[cand.channel_id] = se_l
            pvals_d[cand.channel_id] = float(norm.sf(g_l / max(1e-12, se_l)))

        adj = holm_bonferroni_adjust([pvals_d[c.channel_id] for c in candidates])
        passing = [
            c
            for c, ap in zip(candidates, adj, strict=True)
            if ap < alpha_select and scores_d[c.channel_id] > 0.0
        ]
        passing.sort(
            key=lambda c: scores_d[c.channel_id] / max(1e-6, c.acquisition_cost), reverse=True
        )
        chosen = []
        spent = 0.0
        for c in passing:
            if spent + c.acquisition_cost <= max_budget + 1e-9:
                chosen.append(c)
                spent += c.acquisition_cost
        return DiscoverySelectionResult(
            method="loco_pilot",
            pilot_design=pilot.design,
            pilot_fraction=pilot.pilot_fraction_of_valid,
            pilot_channel_accesses=int(pilot.n_pilot * len(candidates)),
            selected_channels=chosen,
            selected_ids=[c.channel_id for c in chosen],
            total_cost_spent=spent,
            initial_scores=scores_d,
            initial_ses=ses_d,
            initial_pvals=pvals_d,
        )

    if method == "crt_knockoff_pilot":
        # Gaussian-copula Model-X Knockoff / CRT importance (Candes et al., 2018)
        prep_b = TabularPreprocessor(semantic_types=sem_p)
        Xb_mat = np.nan_to_num(prep_b.fit_transform(base_df_p), nan=0.0)
        prep_z = TabularPreprocessor()
        Z_mat = np.nan_to_num(prep_z.fit_transform(Z_df_p), nan=0.0)

        # Standardize Xb and generate conditional residual knockoffs Z_tilde | X_base
        Xb_std = (Xb_mat - np.mean(Xb_mat, axis=0)) / np.maximum(1e-6, np.std(Xb_mat, axis=0))
        reg = Ridge(alpha=10.0)
        reg.fit(Xb_std, Z_mat, sample_weight=w_p)
        Z_hat = reg.predict(Xb_std)
        resid_z = Z_mat - Z_hat
        # Permute residuals independently per channel block within baseline score strata
        Z_tilde = Z_mat.copy()
        src_map_z = _source_to_transformed_indices(prep_z)
        b_edges = np.unique(np.quantile(init_logit_0, np.linspace(0.0, 1.0, 9)))
        b_bins = (
            np.zeros(Xb_mat.shape[0], dtype=int)
            if b_edges.size <= 2
            else np.clip(np.digitize(init_logit_0, b_edges[1:-1]), 0, b_edges.size - 2)
        )
        for c_idx, cand in enumerate(candidates):
            c_js = [j for col in cand.columns for j in src_map_z.get(col, [])]
            if not c_js:
                continue
            k_rng = np.random.default_rng(seed + 71 * c_idx)
            perm_idx = np.arange(Xb_mat.shape[0])
            for bv in np.unique(b_bins):
                bm = np.flatnonzero(b_bins == bv)
                if bm.size > 1:
                    perm_idx[bm] = k_rng.permutation(bm)
            for cj in c_js:
                Z_tilde[:, cj] = Z_hat[:, cj] + resid_z[perm_idx, cj]

        X_aug_ko = np.hstack([Xb_mat, Z_mat, Z_tilde])
        w_norm = w_p * (float(y_p.size) / max(1e-9, float(np.sum(w_p))))
        clf_ko = lgb.LGBMClassifier(
            n_estimators=70,
            learning_rate=0.06,
            num_leaves=23,
            max_depth=5,
            min_child_samples=max(8, min(25, int(y_p.size // 30))),
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            max_delta_step=1.0,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
        clf_ko.fit(X_aug_ko, y_p, sample_weight=w_norm, init_score=init_logit_0)
        gains_imp = clf_ko.booster_.feature_importance(importance_type="gain")
        n_b = Xb_mat.shape[1]
        n_z = Z_mat.shape[1]

        w_stats: dict[str, float] = {}
        for cand in candidates:
            c_js = [j for col in cand.columns for j in src_map_z.get(col, [])]
            g_orig = float(sum(gains_imp[n_b + cj] for cj in c_js))
            g_ko = float(sum(gains_imp[n_b + n_z + cj] for cj in c_js))
            w_stats[cand.channel_id] = g_orig - g_ko

        # Knockoff+ data-dependent threshold (Candes et al., 2018) with fallback if M < 1/q
        abs_vals = sorted({abs(v) for v in w_stats.values() if abs(v) > 0.0})
        ko_fdr_q = max(0.15, alpha_select * 2.0)
        tau_ko = float("inf")
        for t_cand in abs_vals:
            n_neg = sum(1 for v in w_stats.values() if v <= -t_cand)
            n_pos = max(1, sum(1 for v in w_stats.values() if v >= t_cand))
            if (1.0 + float(n_neg)) / float(n_pos) <= ko_fdr_q:
                tau_ko = t_cand
                break
        if tau_ko == float("inf") and abs_vals:
            # Small-pool conservative positive margin above largest negative mirror statistic
            max_neg = max([0.0] + [-v for v in w_stats.values() if v < 0.0])
            tau_ko = max_neg * 1.10 + 1e-6

        passing = [
            c for c in candidates if w_stats[c.channel_id] >= tau_ko and w_stats[c.channel_id] > 0.0
        ]
        passing.sort(
            key=lambda c: w_stats[c.channel_id] / max(1e-6, c.acquisition_cost), reverse=True
        )
        chosen = []
        spent = 0.0
        for c in passing:
            if spent + c.acquisition_cost <= max_budget + 1e-9:
                chosen.append(c)
                spent += c.acquisition_cost
        return DiscoverySelectionResult(
            method="crt_knockoff_pilot",
            pilot_design=pilot.design,
            pilot_fraction=pilot.pilot_fraction_of_valid,
            pilot_channel_accesses=int(pilot.n_pilot * len(candidates)),
            selected_channels=chosen,
            selected_ids=[c.channel_id for c in chosen],
            total_cost_spent=spent,
            initial_scores=w_stats,
        )

    if method == "chamma_residual_cpi_pilot":
        # Chamma et al. (2023) Residualized Conditional Permutation Importance conditional on
        # BOTH X_base and all other candidate channels Z_{-m}.
        n_p = int(y_p.size)
        valid_f = np.zeros(n_p, dtype=bool)
        perm_ll_drops = {c.channel_id: np.zeros(n_p, dtype=float) for c in candidates}

        for f_idx, (tr_i, va_i) in enumerate(cv_splits):
            if np.unique(y_p[tr_i]).size < 2:
                continue
            prep = TabularPreprocessor(semantic_types=sem_p)
            X_tr = np.nan_to_num(prep.fit_transform(joint_df_p.iloc[tr_i]), nan=0.0)
            X_va = np.nan_to_num(prep.transform(joint_df_p.iloc[va_i]), nan=0.0)
            w_tr = w_p[tr_i] * (float(tr_i.size) / max(1e-9, float(np.sum(w_p[tr_i]))))
            clf = lgb.LGBMClassifier(
                n_estimators=55,
                learning_rate=0.07,
                num_leaves=23,
                max_depth=5,
                min_child_samples=max(8, min(25, int(tr_i.size // 25))),
                max_bin=127,
                reg_alpha=1.0,
                reg_lambda=5.0,
                max_delta_step=1.0,
                random_state=seed + f_idx,
                n_jobs=2,
                verbose=-1,
            )
            clf.fit(X_tr, y_p[tr_i], sample_weight=w_tr, init_score=init_logit_0[tr_i])
            tr_m = init_logit_0[tr_i] + np.asarray(clf.predict(X_tr, raw_score=True), dtype=float)
            va_m = init_logit_0[va_i] + np.asarray(clf.predict(X_va, raw_score=True), dtype=float)
            cal = LogisticRegression(C=2.0, solver="lbfgs", max_iter=200)
            cal.fit(tr_m.reshape(-1, 1), y_p[tr_i], sample_weight=w_tr)
            p_va = np.clip(cal.predict_proba(va_m.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS)
            valid_f[va_i] = True
            ll_va_full = _bernoulli_loglik(y_p[va_i], p_va)

            src_map = _source_to_transformed_indices(prep)
            n_cols_tot = X_tr.shape[1]
            mu_tr = np.mean(X_tr, axis=0)
            sd_tr = np.maximum(1e-6, np.std(X_tr, axis=0))
            X_tr_std = (X_tr - mu_tr) / sd_tr
            X_va_std = (X_va - mu_tr) / sd_tr

            for c_idx, cand in enumerate(candidates):
                c_idxs = [j for col in cand.columns for j in src_map.get(col, [])]
                if not c_idxs:
                    continue
                minus_idxs = [j for j in range(n_cols_tot) if j not in set(c_idxs)]
                reg_c = Ridge(alpha=5.0)
                reg_c.fit(X_tr_std[:, minus_idxs], X_tr[:, c_idxs], sample_weight=w_tr)
                Z_hat_va = reg_c.predict(X_va_std[:, minus_idxs])
                if Z_hat_va.ndim == 1:
                    Z_hat_va = Z_hat_va[:, None]
                res_va = X_va[:, c_idxs] - Z_hat_va

                acc_drop = np.zeros(va_i.size, dtype=float)
                n_perm = 3
                for p_rep in range(n_perm):
                    perm_rng = np.random.default_rng(seed + 1301 * f_idx + 53 * c_idx + p_rep)
                    shuf = perm_rng.permutation(va_i.size)
                    X_va_cf = X_va.copy()
                    X_va_cf[:, c_idxs] = Z_hat_va + res_va[shuf]
                    m_cf = init_logit_0[va_i] + np.asarray(
                        clf.predict(X_va_cf, raw_score=True), dtype=float
                    )
                    p_cf = np.clip(
                        cal.predict_proba(m_cf.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS
                    )
                    acc_drop += ll_va_full - _bernoulli_loglik(y_p[va_i], p_cf)
                perm_ll_drops[cand.channel_id][va_i] = acc_drop / float(n_perm)

        scores_d = {}
        ses_d = {}
        pvals_d = {}
        for cand in candidates:
            psi_c = perm_ll_drops[cand.channel_id][valid_f]
            g_c, se_c, _, _ = two_phase_variance_decomposition(
                psi_c,
                w_p[valid_f],
                h_p[valid_f],
                pilot.stratum_pop_counts,
                pilot.stratum_sample_counts,
                groups=grp_p[valid_f] if grp_p is not None else None,
            )
            scores_d[cand.channel_id] = g_c
            ses_d[cand.channel_id] = se_c
            pvals_d[cand.channel_id] = float(norm.sf(g_c / max(1e-12, se_c)))

        adj = holm_bonferroni_adjust([pvals_d[c.channel_id] for c in candidates])
        passing = [
            c
            for c, ap in zip(candidates, adj, strict=True)
            if ap < alpha_select and scores_d[c.channel_id] > 0.0
        ]
        passing.sort(
            key=lambda c: scores_d[c.channel_id] / max(1e-6, c.acquisition_cost), reverse=True
        )
        chosen = []
        spent = 0.0
        for c in passing:
            if spent + c.acquisition_cost <= max_budget + 1e-9:
                chosen.append(c)
                spent += c.acquisition_cost
        return DiscoverySelectionResult(
            method="chamma_residual_cpi_pilot",
            pilot_design=pilot.design,
            pilot_fraction=pilot.pilot_fraction_of_valid,
            pilot_channel_accesses=int(pilot.n_pilot * len(candidates)),
            selected_channels=chosen,
            selected_ids=[c.channel_id for c in chosen],
            total_cost_spent=spent,
            initial_scores=scores_d,
            initial_ses=ses_d,
            initial_pvals=pvals_d,
        )

    raise ValueError(f"Unknown discovery competitor method: {method}")


def evaluate_split_pilot_step1_inference(
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    oof_base_prob: np.ndarray,
    pilot: TwoPhasePilot,
    *,
    base_semantic_types: dict[str, SemanticType] | None = None,
    alpha_select: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    """Disjoint split-pilot inference eliminating K-fold training-overlap covariance.

    Partitions each Phase-2 pilot stratum h without replacement into a fitting half P_{h,fit}
    and a disjoint evaluation half P_{h,eval}. Fits capacity-matched offset GBDTs strictly on
    P_{fit} and evaluates two-phase SEs (V_I + V_II) and Holm LCBs on P_{eval}, plus the exact
    full-cohort target G_N(eta_fit) across all causal-valid Phase-1 records.
    """
    rng = np.random.default_rng(seed + 31337)
    idx = pilot.sampled_indices
    h_p = pilot.stratum_ids_sampled
    y_all = np.asarray(y_train, dtype=int).ravel()
    sem_p = dict(base_semantic_types) if base_semantic_types is not None else {}

    fit_pos_list: list[np.ndarray] = []
    eval_pos_list: list[np.ndarray] = []
    fit_smp_counts: dict[int, int] = {}
    eval_smp_counts: dict[int, int] = {}

    for h_val in sorted(set(h_p.tolist())):
        loc_h = np.flatnonzero(h_p == h_val)
        n_h = int(loc_h.size)
        if n_h < 2:
            fit_pos_list.append(loc_h)
            fit_smp_counts[int(h_val)] = n_h
            continue
        perm = rng.permutation(loc_h)
        n_fit = max(1, n_h // 2)
        n_ev = n_h - n_fit
        fit_pos_list.append(np.sort(perm[:n_fit]))
        eval_pos_list.append(np.sort(perm[n_fit:]))
        fit_smp_counts[int(h_val)] = n_fit
        eval_smp_counts[int(h_val)] = n_ev

    fit_loc = np.sort(np.concatenate(fit_pos_list))
    eval_loc = np.sort(np.concatenate(eval_pos_list))
    fit_idx = idx[fit_loc]
    eval_idx = idx[eval_loc]

    h_fit = h_p[fit_loc]
    h_eval = h_p[eval_loc]
    w_fit = np.array(
        [pilot.stratum_pop_counts[int(h)] / max(1, fit_smp_counts[int(h)]) for h in h_fit],
        dtype=float,
    )
    w_eval = np.array(
        [pilot.stratum_pop_counts[int(h)] / max(1, eval_smp_counts[int(h)]) for h in h_eval],
        dtype=float,
    )
    w_fit_norm = w_fit * (float(w_fit.size) / max(1e-12, float(np.sum(w_fit))))

    Xb_fit = X_base_train.iloc[fit_idx].reset_index(drop=True)
    Xb_eval = X_base_train.iloc[eval_idx].reset_index(drop=True)
    Xb_all = X_base_train.reset_index(drop=True)
    y_fit = y_all[fit_idx]
    y_eval = y_all[eval_idx]

    init_fit = logit(np.clip(oof_base_prob[fit_idx], _PROB_EPS, 1.0 - _PROB_EPS))
    init_eval = logit(np.clip(oof_base_prob[eval_idx], _PROB_EPS, 1.0 - _PROB_EPS))
    init_all = logit(np.clip(oof_base_prob, _PROB_EPS, 1.0 - _PROB_EPS))

    def _fit_and_predict(
        df_tr: pd.DataFrame,
        df_ev: pd.DataFrame,
        df_full: pd.DataFrame,
        s_types: dict[str, SemanticType] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        prep = TabularPreprocessor(semantic_types=s_types)
        X_tr_m = prep.fit_transform(df_tr)
        X_ev_m = prep.transform(df_ev)
        X_fl_m = prep.transform(df_full)
        min_child = max(8, min(25, int(X_tr_m.shape[0] // 22)))
        clf = lgb.LGBMClassifier(
            n_estimators=55,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=min_child,
            subsample_for_bin=min(50000, max(2000, int(X_tr_m.shape[0]))),
            max_bin=127,
            reg_alpha=1.0,
            reg_lambda=5.0,
            max_delta_step=1.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        clf.fit(X_tr_m, y_fit, sample_weight=w_fit_norm, init_score=init_fit)
        tr_m = init_fit + np.asarray(clf.predict(X_tr_m, raw_score=True), dtype=float)
        ev_m = init_eval + np.asarray(clf.predict(X_ev_m, raw_score=True), dtype=float)
        fl_m = init_all + np.asarray(clf.predict(X_fl_m, raw_score=True), dtype=float)

        cal = LogisticRegression(C=2.0, solver="lbfgs", max_iter=200)
        cal.fit(tr_m.reshape(-1, 1), y_fit, sample_weight=w_fit_norm)
        p_ev = np.clip(cal.predict_proba(ev_m.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS)
        p_fl = np.clip(cal.predict_proba(fl_m.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS)
        return p_ev, p_fl

    p0_ev, p0_fl = _fit_and_predict(Xb_fit, Xb_eval, Xb_all, sem_p)
    ll0_ev = _bernoulli_loglik(y_eval, p0_ev)
    ll0_fl = _bernoulli_loglik(y_all, p0_fl)

    scores: dict[str, float] = {}
    ses_total: dict[str, float] = {}
    ses_p1: dict[str, float] = {}
    ses_p2: dict[str, float] = {}
    raw_pvals: list[float] = []
    census_targets: dict[str, float] = {}

    Z_fit = Z_pool_train.iloc[fit_idx].reset_index(drop=True)
    Z_eval = Z_pool_train.iloc[eval_idx].reset_index(drop=True)
    Z_all = Z_pool_train.reset_index(drop=True)

    for cand in candidates:
        cols = list(cand.columns)
        pm_ev, pm_fl = _fit_and_predict(
            pd.concat([Xb_fit, Z_fit[cols]], axis=1),
            pd.concat([Xb_eval, Z_eval[cols]], axis=1),
            pd.concat([Xb_all, Z_all[cols]], axis=1),
            sem_p,
        )
        psi_ev = _bernoulli_loglik(y_eval, pm_ev) - ll0_ev
        psi_fl = _bernoulli_loglik(y_all, pm_fl) - ll0_fl
        g_m, se_t, se_1, se_2 = two_phase_variance_decomposition(
            psi_ev,
            w_eval,
            h_eval,
            pilot.stratum_pop_counts,
            eval_smp_counts,
        )
        scores[cand.channel_id] = g_m
        ses_total[cand.channel_id] = se_t
        ses_p1[cand.channel_id] = se_1
        ses_p2[cand.channel_id] = se_2
        census_targets[cand.channel_id] = float(np.mean(psi_fl))
        raw_pvals.append(float(norm.sf(g_m / max(1e-12, se_t))))

    adj_p = holm_bonferroni_adjust(raw_pvals)
    selected_ids = [
        cand.channel_id
        for cand, ap in zip(candidates, adj_p, strict=True)
        if ap < alpha_select and scores[cand.channel_id] > 0.0
    ]

    return {
        "scores": scores,
        "ses_total": ses_total,
        "ses_phase1": ses_p1,
        "ses_phase2": ses_p2,
        "census_targets_conditional": census_targets,
        "holm_pvals": {c.channel_id: float(ap) for c, ap in zip(candidates, adj_p, strict=True)},
        "selected_ids": selected_ids,
        "n_fit": int(fit_idx.size),
        "n_eval": int(eval_idx.size),
    }


def discover_sequential_split_pilot(
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    oof_base_prob: np.ndarray,
    pilot: TwoPhasePilot,
    max_budget: float,
    *,
    base_semantic_types: dict[str, SemanticType] | None = None,
    alpha_select: float = 0.05,
    n_jobs: int = 1,
    seed: int = 42,
) -> DiscoverySelectionResult:
    """Multi-step sequential discovery on a disjoint 50/50 split-pilot (Theorem 1).

    Partitions each Phase-2 pilot stratum ``h`` without replacement into a fitting half
    ``P_{h,fit}`` and a disjoint evaluation half ``P_{h,eval}``. At each forward step ``t``,
    conditional on ``P_{fit}``, fits capacity-matched offset GBDTs strictly on ``P_{fit}`` and
    computes design-based two-phase variance (``V_I + V_II``) and Holm-adjusted one-sided stopping
    strictly on ``P_{eval}``, eliminating cross-fitting fold-overlap covariance.
    """
    rng = np.random.default_rng(seed + 31337)
    idx = pilot.sampled_indices
    h_p = pilot.stratum_ids_sampled
    y_all = np.asarray(y_train, dtype=int).ravel()
    sem_p = dict(base_semantic_types) if base_semantic_types is not None else {}

    fit_pos_list: list[np.ndarray] = []
    eval_pos_list: list[np.ndarray] = []
    fit_smp_counts: dict[int, int] = {}
    eval_smp_counts: dict[int, int] = {}

    for h_val in sorted(set(h_p.tolist())):
        loc_h = np.flatnonzero(h_p == h_val)
        n_h = int(loc_h.size)
        if n_h < 2:
            fit_pos_list.append(loc_h)
            fit_smp_counts[int(h_val)] = n_h
            continue
        perm = rng.permutation(loc_h)
        n_fit = max(1, n_h // 2)
        n_ev = n_h - n_fit
        fit_pos_list.append(np.sort(perm[:n_fit]))
        eval_pos_list.append(np.sort(perm[n_fit:]))
        fit_smp_counts[int(h_val)] = n_fit
        eval_smp_counts[int(h_val)] = n_ev

    fit_loc = np.sort(np.concatenate(fit_pos_list))
    eval_loc = np.sort(np.concatenate(eval_pos_list))
    fit_idx = idx[fit_loc]
    eval_idx = idx[eval_loc]

    h_fit = h_p[fit_loc]
    h_eval = h_p[eval_loc]
    w_fit = np.array(
        [pilot.stratum_pop_counts[int(h)] / max(1, fit_smp_counts[int(h)]) for h in h_fit],
        dtype=float,
    )
    w_eval = np.array(
        [pilot.stratum_pop_counts[int(h)] / max(1, eval_smp_counts[int(h)]) for h in h_eval],
        dtype=float,
    )
    w_fit_norm = w_fit * (float(w_fit.size) / max(1e-12, float(np.sum(w_fit))))

    Xb_fit = X_base_train.iloc[fit_idx].reset_index(drop=True)
    Xb_eval = X_base_train.iloc[eval_idx].reset_index(drop=True)
    Z_fit = Z_pool_train.iloc[fit_idx].reset_index(drop=True)
    Z_eval = Z_pool_train.iloc[eval_idx].reset_index(drop=True)
    y_fit = y_all[fit_idx]
    y_eval = y_all[eval_idx]

    init_fit = logit(np.clip(oof_base_prob[fit_idx], _PROB_EPS, 1.0 - _PROB_EPS))
    init_eval = logit(np.clip(oof_base_prob[eval_idx], _PROB_EPS, 1.0 - _PROB_EPS))

    def _fit_eval_probs(df_tr: pd.DataFrame, df_ev: pd.DataFrame, step_seed: int) -> np.ndarray:
        prep = TabularPreprocessor(semantic_types=sem_p)
        X_tr_m = prep.fit_transform(df_tr)
        X_ev_m = prep.transform(df_ev)
        min_child = max(8, min(25, int(X_tr_m.shape[0] // 22)))
        clf = lgb.LGBMClassifier(
            n_estimators=55,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=min_child,
            subsample_for_bin=min(50000, max(2000, int(X_tr_m.shape[0]))),
            max_bin=127,
            reg_alpha=1.0,
            reg_lambda=5.0,
            max_delta_step=1.0,
            random_state=step_seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        clf.fit(X_tr_m, y_fit, sample_weight=w_fit_norm, init_score=init_fit)
        tr_m = init_fit + np.asarray(clf.predict(X_tr_m, raw_score=True), dtype=float)
        ev_m = init_eval + np.asarray(clf.predict(X_ev_m, raw_score=True), dtype=float)
        cal = LogisticRegression(C=2.0, solver="lbfgs", max_iter=200)
        cal.fit(tr_m.reshape(-1, 1), y_fit, sample_weight=w_fit_norm)
        return np.clip(cal.predict_proba(ev_m.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS)

    chosen: list[CandidateChannelSpec] = []
    chosen_ids: set[str] = set()
    spent = 0.0
    step_trace: list[SequentialChannelStep] = []
    init_scores: dict[str, float] = {}
    init_ses: dict[str, float] = {}
    init_pvals: dict[str, float] = {}

    step_num = 0
    while True:
        step_num += 1
        rem_budget = max_budget - spent
        affordable = [
            c
            for c in candidates
            if c.channel_id not in chosen_ids and c.acquisition_cost <= rem_budget + 1e-9
        ]
        if not affordable:
            break

        sel_cols = [col for c in chosen for col in c.columns]
        cur_fit = pd.concat([Xb_fit, Z_fit[sel_cols]], axis=1) if sel_cols else Xb_fit
        cur_eval = pd.concat([Xb_eval, Z_eval[sel_cols]], axis=1) if sel_cols else Xb_eval
        p_base_ev = _fit_eval_probs(cur_fit, cur_eval, seed + 100 * step_num)
        ll_base_ev = _bernoulli_loglik(y_eval, p_base_ev)

        step_g: dict[str, float] = {}
        step_se: dict[str, tuple[float, float, float]] = {}
        raw_ps: list[float] = []
        for m_i, cand in enumerate(affordable):
            c_cols = sel_cols + list(cand.columns)
            pm_ev = _fit_eval_probs(
                pd.concat([Xb_fit, Z_fit[c_cols]], axis=1),
                pd.concat([Xb_eval, Z_eval[c_cols]], axis=1),
                seed + 100 * step_num + m_i + 1,
            )
            psi_ev = _bernoulli_loglik(y_eval, pm_ev) - ll_base_ev
            g_m, se_t, se_1, se_2 = two_phase_variance_decomposition(
                psi_ev, w_eval, h_eval, pilot.stratum_pop_counts, eval_smp_counts
            )
            step_g[cand.channel_id] = g_m
            step_se[cand.channel_id] = (se_t, se_1, se_2)
            p_r = float(norm.sf(g_m / max(1e-12, se_t)))
            raw_ps.append(p_r)
            if step_num == 1:
                init_scores[cand.channel_id] = g_m
                init_ses[cand.channel_id] = se_t
                init_pvals[cand.channel_id] = p_r

        adj_ps = holm_bonferroni_adjust(raw_ps)
        passing: list[tuple[float, CandidateChannelSpec, float, float]] = []
        best_any: tuple[float, CandidateChannelSpec, float, float] | None = None
        for cand, pr, pa in zip(affordable, raw_ps, adj_ps, strict=True):
            gm = step_g[cand.channel_id]
            ratio = gm / max(1e-6, cand.acquisition_cost)
            if best_any is None or ratio > best_any[0]:
                best_any = (ratio, cand, pr, float(pa))
            if pa < alpha_select and gm > 0.0:
                passing.append((ratio, cand, pr, float(pa)))

        if not passing:
            if best_any is not None:
                _, bc, bpr, bpa = best_any
                b_se, b_s1, b_s2 = step_se[bc.channel_id]
                bg = step_g[bc.channel_id]
                step_trace.append(
                    SequentialChannelStep(
                        step=step_num,
                        selected_channel_id=None,
                        gain_nats=bg,
                        se_total=b_se,
                        se_phase1=b_s1,
                        se_phase2=b_s2,
                        z_stat=bg / max(1e-12, b_se),
                        p_raw=bpr,
                        p_adjusted=bpa,
                        lcb_95=bg - float(norm.ppf(1.0 - alpha_select)) * b_se,
                        cost=0.0,
                        stopped_by_error_control=True,
                    )
                )
            break

        passing.sort(key=lambda t: t[0], reverse=True)
        _, win_c, win_pr, win_pa = passing[0]
        w_se, w_s1, w_s2 = step_se[win_c.channel_id]
        wg = step_g[win_c.channel_id]
        chosen.append(win_c)
        chosen_ids.add(win_c.channel_id)
        spent += win_c.acquisition_cost
        step_trace.append(
            SequentialChannelStep(
                step=step_num,
                selected_channel_id=win_c.channel_id,
                gain_nats=wg,
                se_total=w_se,
                se_phase1=w_s1,
                se_phase2=w_s2,
                z_stat=wg / max(1e-12, w_se),
                p_raw=win_pr,
                p_adjusted=win_pa,
                lcb_95=wg - float(norm.ppf(1.0 - alpha_select)) * w_se,
                cost=win_c.acquisition_cost,
                stopped_by_error_control=False,
            )
        )

    return DiscoverySelectionResult(
        method="neyman_split_pilot_seq_cmi",
        pilot_design=pilot.design,
        pilot_fraction=pilot.pilot_fraction_of_valid,
        pilot_channel_accesses=int(pilot.n_pilot * len(candidates)),
        selected_channels=chosen,
        selected_ids=[c.channel_id for c in chosen],
        total_cost_spent=spent,
        step_trace=step_trace,
        initial_scores=init_scores,
        initial_ses=init_ses,
        initial_pvals=init_pvals,
    )


__all__ = [
    "DiscoverySelectionResult",
    "SequentialChannelStep",
    "TwoPhasePilot",
    "discover_competitor_method",
    "discover_sequential_conditional_gain",
    "discover_sequential_split_pilot",
    "evaluate_split_pilot_step1_inference",
    "sample_two_phase_pilot",
    "two_phase_variance_decomposition",
]
