"""Stage D (Part 1): Cross-fitted consensus strata and Horvitz-Thompson case-cohort sampling."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from imboreceipt.core.types import SemanticType
from imboreceipt.frontier.baselines import TabularPreprocessor


@dataclass(frozen=True, slots=True)
class ConsensusResidualStrata:
    """Out-of-fold consensus predictions and matched error strata on training data."""

    oof_glm_prob: np.ndarray
    oof_gbdt_prob: np.ndarray
    oof_consensus_prob: np.ndarray
    oof_valid_mask: np.ndarray
    temporal_burnin_fraction: float
    temporal_cv_feasible: bool
    confident_fp_mask: np.ndarray
    confident_fn_mask: np.ndarray
    matched_tp_mask: np.ndarray
    matched_tn_mask: np.ndarray
    alert_region_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class CaseCohortPilot:
    """Stratified case-cohort design with exact Horvitz-Thompson inclusion weights."""

    sampled_indices: np.ndarray
    sampling_weights: np.ndarray
    inclusion_probabilities: np.ndarray
    stratum_ids: np.ndarray
    is_matched_alert_stratum: np.ndarray
    fraction_of_cohort_measured: float
    fraction_of_eligible_measured: float
    matching_mode: str


def make_leak_free_cv_splits(
    y: np.ndarray,
    *,
    n_folds: int = 5,
    groups: np.ndarray | None = None,
    time_order: np.ndarray | None = None,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct group-aware or strictly forward chronological CV splits without leakage.

    When ``time_order`` is provided:
    - Enforces strictly forward expanding-block chronological splits: every returned pair
      ``(train_idx, val_idx)`` satisfies ``max(time_order[train_idx]) <= min(time_order[val_idx])``.
    - Uses earliest block(s) strictly as a causal burn-in period (never predicted from future
      blocks and never falling back silently to random stratified CV).
    """
    yv = np.asarray(y, dtype=int).ravel()
    n = yv.size
    n_pos = int(yv.sum())
    k = min(n_folds, max(2, n_pos), max(2, n - n_pos))

    if time_order is not None:
        t_arr = np.asarray(time_order, dtype=float).ravel()
        order = np.argsort(t_arr, kind="mergesort")
        # Partition into k + 1 chronological blocks: block 0 is initial burn-in
        n_blocks = max(2, k + 1)
        raw_blocks = [b for b in np.array_split(order, n_blocks) if b.size > 0]
        if len(raw_blocks) < 2:
            return []

        # Widen initial burn-in if earliest block lacks both classes
        burn_end = 1
        while burn_end < len(raw_blocks) - 1:
            burn_idx = np.concatenate(raw_blocks[:burn_end])
            if int(yv[burn_idx].sum()) >= 1 and int((1 - yv[burn_idx]).sum()) >= 1:
                break
            burn_end += 1

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        g_arr_t = np.asarray(groups) if groups is not None else None
        for j in range(burn_end, len(raw_blocks)):
            past_tr = np.concatenate(raw_blocks[:j])
            va_j = raw_blocks[j]
            # Enforce strict non-overlapping timestamp boundary when ties straddle the cut
            max_tr_t = float(np.max(t_arr[past_tr]))
            min_va_t = float(np.min(t_arr[va_j]))
            if max_tr_t >= min_va_t:
                tied = va_j[t_arr[va_j] <= max_tr_t]
                va_clean = va_j[t_arr[va_j] > max_tr_t]
                if tied.size > 0 and va_clean.size > 0:
                    past_tr = np.concatenate([past_tr, tied])
                    va_j = va_clean
            # When both entity groups and chronology are supplied, purge seen entities from va_j
            if g_arr_t is not None and va_j.size > 0:
                seen_grps = np.unique(g_arr_t[past_tr])
                va_j = va_j[~np.isin(g_arr_t[va_j], seen_grps)]
            if va_j.size > 0 and int(yv[past_tr].sum()) >= 1 and int((1 - yv[past_tr]).sum()) >= 1:
                splits.append((past_tr, va_j))
        return splits

    if groups is not None:
        g_arr = np.asarray(groups)
        n_grp = np.unique(g_arr).size
        if n_grp >= k * 2:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            cand_splits = list(sgkf.split(np.zeros(n), yv, g_arr))
            if all(
                int(yv[tr].sum()) >= 1 and int((1 - yv[tr]).sum()) >= 1 for tr, _ in cand_splits
            ):
                return cand_splits

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(n), yv))


def build_consensus_residual_strata(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    semantic_types: dict[str, SemanticType] | None = None,
    groups: np.ndarray | None = None,
    time_order: np.ndarray | None = None,
    alert_tail_quantile: float = 0.92,
    n_folds: int = 5,
    seed: int = 42,
) -> ConsensusResidualStrata:
    """Compute leak-free out-of-fold predictions across linear + GBDT families.

    Under ``time_order``, burn-in records receive ``oof_valid_mask = False`` and are
    excluded from causal out-of-fold pilot selection and residual scoring.
    """
    y = np.asarray(y_train, dtype=int).ravel()
    n = y.size
    prior_p = float(np.clip(y.mean(), 1e-5, 1.0 - 1e-5))

    oof_glm = np.full(n, prior_p, dtype=float)
    oof_gbdt = np.full(n, prior_p, dtype=float)
    oof_valid = np.zeros(n, dtype=bool)

    cv_splits = make_leak_free_cv_splits(
        y, n_folds=n_folds, groups=groups, time_order=time_order, seed=seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_i, va_i in cv_splits:
            prep = TabularPreprocessor(semantic_types=semantic_types).fit(X_train.iloc[tr_i])
            X_tr = prep.transform(X_train.iloc[tr_i])
            X_va = prep.transform(X_train.iloc[va_i])

            glm = LogisticRegression(C=1.0, max_iter=600, random_state=seed)
            glm.fit(X_tr, y[tr_i])
            oof_glm[va_i] = glm.predict_proba(X_va)[:, 1]

            gbdt = LGBMClassifier(
                n_estimators=120,
                learning_rate=0.05,
                num_leaves=23,
                min_child_samples=15,
                max_delta_step=1.0,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            )
            gbdt.fit(X_tr, y[tr_i])
            oof_gbdt[va_i] = gbdt.predict_proba(X_va)[:, 1]
            oof_valid[va_i] = True

    temporal_feasible = True
    if not np.any(oof_valid):
        temporal_feasible = time_order is None
        oof_valid[:] = True

    burnin_frac = float(1.0 - np.mean(oof_valid))
    oof_cons = 0.5 * (oof_glm + oof_gbdt)
    val_cons = oof_cons[oof_valid]
    glm_hi = float(np.quantile(oof_glm[oof_valid], alert_tail_quantile))
    gbdt_hi = float(np.quantile(oof_gbdt[oof_valid], alert_tail_quantile))
    cons_hi = float(np.quantile(val_cons, alert_tail_quantile))

    pos_scores = oof_cons[oof_valid & (y == 1)]
    fn_cut = float(np.quantile(pos_scores, 0.40)) if pos_scores.size > 0 else prior_p

    alert_region = oof_valid & (oof_cons >= cons_hi)
    conf_fp = oof_valid & (y == 0) & (((oof_glm >= glm_hi) & (oof_gbdt >= gbdt_hi)) | alert_region)
    conf_fn = oof_valid & (y == 1) & (oof_glm <= fn_cut) & (oof_gbdt <= fn_cut)
    matched_tp = oof_valid & (y == 1) & alert_region
    matched_tn = oof_valid & (y == 0) & (oof_cons <= fn_cut)

    return ConsensusResidualStrata(
        oof_glm_prob=oof_glm,
        oof_gbdt_prob=oof_gbdt,
        oof_consensus_prob=oof_cons,
        oof_valid_mask=oof_valid,
        temporal_burnin_fraction=burnin_frac,
        temporal_cv_feasible=temporal_feasible,
        confident_fp_mask=conf_fp,
        confident_fn_mask=conf_fn,
        matched_tp_mask=matched_tp,
        matched_tn_mask=matched_tn,
        alert_region_mask=alert_region,
    )


def adaptive_controls_per_positive(prevalence: float) -> int:
    """Prespecified prevalence heuristic for case-cohort control multiplier m(pi).

    - Moderate imbalance (pi >= 10%): m = 2 (limits pilot size to ~3*pi and avoids
      exhausting upper-stratum controls).
    - Classic rare events (0.5% <= pi < 10%): m = 4 (5*pi pilot size).
    - Ultra-rare events (pi < 0.5%): m = 8 (9*pi pilot size, stabilizing sparse strata
      while keeping total pilot < 4.5% of eligible records).
    """
    pi = float(prevalence)
    if pi >= 0.10:
        return 2
    if pi < 0.005:
        return 8
    return 4


def sample_case_cohort_pilot(
    y_train: np.ndarray,
    oof_scores: np.ndarray,
    *,
    oof_valid_mask: np.ndarray | None = None,
    controls_per_positive: int | None = None,
    matching_mode: Literal["score_stratified", "unmatched_random"] = "score_stratified",
    n_strata: int = 10,
    observation_propensity: np.ndarray | None = None,
    seed: int = 42,
) -> CaseCohortPilot:
    """Sample 100% of eligible positives + stratified controls with exact HT weights.

    Under ``matching_mode="score_stratified"``:
    - Partitions eligible (post-burn-in) records into ``n_strata`` quantile bins of the
      out-of-fold baseline score (or joint baseline/observation-propensity index).
    - Samples 100% of eligible cases (inclusion probability 1.0, weight 1.0).
    - Allocates ``m * n_+`` controls across strata proportional to stratum case counts
      (with a small positive floor in every stratum so ``pi_{0,b} = n_{0,b} / N_{0,b} > 0``).
    - Samples ``n_{0,b}`` controls uniformly without replacement inside stratum ``b`` and
      assigns exact Horvitz-Thompson inverse-probability weights ``w_i = N_{0,b} / n_{0,b}``,
      satisfying ``sum_{i in P, y_i=0} w_i == N_0`` identically.
    """
    y = np.asarray(y_train, dtype=int).ravel()
    scores = np.asarray(oof_scores, dtype=float).ravel()
    n_total = max(1, y.size)
    valid = (
        np.asarray(oof_valid_mask, dtype=bool).ravel()
        if oof_valid_mask is not None
        else np.ones(y.size, dtype=bool)
    )
    elig_idx = np.flatnonzero(valid)
    if elig_idx.size == 0:
        elig_idx = np.arange(y.size, dtype=int)

    y_el = y[elig_idx]
    s_el = scores[elig_idx]
    pos_rel = np.flatnonzero(y_el == 1)
    neg_rel = np.flatnonzero(y_el == 0)

    if pos_rel.size == 0 or neg_rel.size == 0:
        return CaseCohortPilot(
            sampled_indices=elig_idx,
            sampling_weights=np.ones(elig_idx.size, dtype=float),
            inclusion_probabilities=np.ones(elig_idx.size, dtype=float),
            stratum_ids=np.zeros(elig_idx.size, dtype=int),
            is_matched_alert_stratum=np.ones(elig_idx.size, dtype=bool),
            fraction_of_cohort_measured=float(elig_idx.size / n_total),
            fraction_of_eligible_measured=1.0,
            matching_mode=matching_mode,
        )

    m_eff = (
        adaptive_controls_per_positive(float(y_el.mean()))
        if controls_per_positive is None
        else int(controls_per_positive)
    )
    target_n_ctrl = min(neg_rel.size, pos_rel.size * m_eff)
    if target_n_ctrl >= neg_rel.size:
        alert_cut = float(np.quantile(s_el[pos_rel], 0.35))
        return CaseCohortPilot(
            sampled_indices=elig_idx,
            sampling_weights=np.ones(elig_idx.size, dtype=float),
            inclusion_probabilities=np.ones(elig_idx.size, dtype=float),
            stratum_ids=np.zeros(elig_idx.size, dtype=int),
            is_matched_alert_stratum=s_el >= alert_cut,
            fraction_of_cohort_measured=float(elig_idx.size / n_total),
            fraction_of_eligible_measured=1.0,
            matching_mode=matching_mode,
        )

    rng = np.random.default_rng(seed)

    if matching_mode == "unmatched_random":
        picked_neg_rel = np.sort(rng.choice(neg_rel, size=target_n_ctrl, replace=False))
        sampled_rel = np.sort(np.concatenate([pos_rel, picked_neg_rel]))
        sampled_abs = elig_idx[sampled_rel]
        pi_0 = float(target_n_ctrl) / float(neg_rel.size)
        w_0 = float(neg_rel.size) / float(target_n_ctrl)
        is_pos = y[sampled_abs] == 1
        weights = np.where(is_pos, 1.0, w_0)
        incl = np.where(is_pos, 1.0, pi_0)
        alert_cut = float(np.quantile(s_el[pos_rel], 0.35))
        return CaseCohortPilot(
            sampled_indices=sampled_abs,
            sampling_weights=weights,
            inclusion_probabilities=incl,
            stratum_ids=np.zeros(sampled_abs.size, dtype=int),
            is_matched_alert_stratum=scores[sampled_abs] >= alert_cut,
            fraction_of_cohort_measured=float(sampled_abs.size / n_total),
            fraction_of_eligible_measured=float(sampled_abs.size / max(1, elig_idx.size)),
            matching_mode=matching_mode,
        )

    # Prespecified quantile strata on baseline score (+ observation propensity if present)
    eps = 1e-6
    s_logit = np.log(np.clip(s_el, eps, 1.0 - eps) / np.clip(1.0 - s_el, eps, 1.0))
    s_std = (s_logit - float(np.mean(s_logit))) / max(1e-6, float(np.std(s_logit)))
    if observation_propensity is not None:
        op = np.asarray(observation_propensity, dtype=float).ravel()[elig_idx]
        if float(np.std(op)) > 1e-6:
            op_std = (op - float(np.mean(op))) / float(np.std(op))
            index_coord = 0.80 * s_std + 0.20 * op_std
        else:
            index_coord = s_std
    else:
        index_coord = s_std

    eff_b = min(max(2, n_strata), max(2, neg_rel.size // 4))
    edges = np.unique(np.quantile(index_coord, np.linspace(0.0, 1.0, eff_b + 1)))
    if edges.size <= 2:
        bins = np.zeros(elig_idx.size, dtype=int)
        n_bins = 1
    else:
        bins = np.searchsorted(edges[1:-1], index_coord, side="right")
        n_bins = int(edges.size - 1)

    neg_counts = np.array(
        [int(np.sum((y_el == 0) & (bins == b))) for b in range(n_bins)], dtype=int
    )
    pos_counts = np.array(
        [int(np.sum((y_el == 1) & (bins == b))) for b in range(n_bins)], dtype=int
    )

    # Allocate controls across strata: 35% size-proportional base floor (bounding max HT weight
    # ratio <= ~2.85x so Kish effective sample size stays high) + 65% case-density matching!
    alloc = np.zeros(n_bins, dtype=int)
    nonempty_neg = np.flatnonzero(neg_counts > 0)
    total_neg_el = max(1, int(neg_counts.sum()))
    base_budget = max(nonempty_neg.size, int(round(0.35 * target_n_ctrl)))
    for b in nonempty_neg:
        prop_b = max(1, int(np.floor(base_budget * (float(neg_counts[b]) / float(total_neg_el)))))
        alloc[b] = min(int(neg_counts[b]), prop_b)
    # Cap base allocation at target_n_ctrl when target_n_ctrl < nonempty_neg.size
    while int(alloc.sum()) > target_n_ctrl:
        hi_b = int(np.argmax(alloc))
        alloc[hi_b] -= 1

    rem_budget = target_n_ctrl - int(alloc.sum())
    if rem_budget > 0:
        spare_cap = neg_counts - alloc
        weights_b = pos_counts.astype(float) + 0.10
        weights_b = np.where(spare_cap > 0, weights_b, 0.0)
        if float(weights_b.sum()) > 0:
            raw_add = rem_budget * (weights_b / float(weights_b.sum()))
            add_int = np.minimum(spare_cap, np.floor(raw_add).astype(int))
            alloc += add_int
            rem_budget = target_n_ctrl - int(alloc.sum())

        # Distribute any rounding remainder by highest fractional case density with spare capacity
        while rem_budget > 0:
            avail_bins = np.flatnonzero(neg_counts > alloc)
            if avail_bins.size == 0:
                break
            order_b = avail_bins[np.argsort(-pos_counts[avail_bins], kind="mergesort")]
            for b in order_b:
                if rem_budget <= 0:
                    break
                if alloc[b] < neg_counts[b]:
                    alloc[b] += 1
                    rem_budget -= 1

    picked_neg_list: list[np.ndarray] = []
    neg_ht_weights: dict[int, float] = {}
    neg_incl_probs: dict[int, float] = {}

    for b in nonempty_neg:
        neg_in_b = np.flatnonzero((y_el == 0) & (bins == b))
        k_b = min(int(alloc[b]), neg_in_b.size)
        if k_b <= 0:
            continue
        chosen_b = np.sort(rng.choice(neg_in_b, size=k_b, replace=False))
        w_b = float(neg_in_b.size) / float(k_b)
        p_b = float(k_b) / float(neg_in_b.size)
        picked_neg_list.append(chosen_b)
        for r_idx in chosen_b:
            neg_ht_weights[int(r_idx)] = w_b
            neg_incl_probs[int(r_idx)] = p_b

    picked_neg_rel = (
        np.sort(np.concatenate(picked_neg_list)) if picked_neg_list else np.empty(0, dtype=int)
    )
    sampled_rel = np.sort(np.concatenate([pos_rel, picked_neg_rel]))
    sampled_abs = elig_idx[sampled_rel]

    weights_out = np.ones(sampled_rel.size, dtype=float)
    incl_out = np.ones(sampled_rel.size, dtype=float)
    for out_i, rel_i in enumerate(sampled_rel):
        if y_el[rel_i] == 0:
            weights_out[out_i] = neg_ht_weights[int(rel_i)]
            incl_out[out_i] = neg_incl_probs[int(rel_i)]

    alert_cut = float(np.quantile(s_el[pos_rel], 0.35))
    return CaseCohortPilot(
        sampled_indices=sampled_abs,
        sampling_weights=weights_out,
        inclusion_probabilities=incl_out,
        stratum_ids=bins[sampled_rel],
        is_matched_alert_stratum=scores[sampled_abs] >= alert_cut,
        fraction_of_cohort_measured=float(sampled_abs.size / n_total),
        fraction_of_eligible_measured=float(sampled_abs.size / max(1, elig_idx.size)),
        matching_mode=matching_mode,
    )
