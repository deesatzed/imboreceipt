"""Stage D (Part 1): Cross-fitted consensus residual strata and case-cohort pilot sampling."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

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
    confident_fp_mask: np.ndarray
    confident_fn_mask: np.ndarray
    matched_tp_mask: np.ndarray
    matched_tn_mask: np.ndarray
    alert_region_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class CaseCohortPilot:
    """Cheap case-cohort measurement design: 100% positives + m score-matched controls."""

    sampled_indices: np.ndarray
    sampling_weights: np.ndarray
    is_matched_alert_stratum: np.ndarray
    fraction_of_cohort_measured: float


def make_leak_free_cv_splits(
    y: np.ndarray,
    *,
    n_folds: int = 5,
    groups: np.ndarray | None = None,
    time_order: np.ndarray | None = None,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct group-aware or causal time-aware CV splits without entity/future leakage."""
    yv = np.asarray(y, dtype=int).ravel()
    n = yv.size
    n_pos = int(yv.sum())
    k = min(n_folds, max(2, n_pos), max(2, n - n_pos))

    if time_order is not None:
        # Strict causal forward-chaining chronological blocks:
        # Block 0 (earliest) is predicted from the second half of Block 0? NO — even stricter:
        # split Block 0 into two sub-halves where sub-half A -> sub-half B and for Block 0_A
        # use out-of-sample prior/sub-half B ONLY if needed, OR better: use causal expanding
        # folds for blocks 1..k-1 and predict block 0 via a within-block0 chronological split
        # (never touching blocks 1..k-1!). Thus ZERO future blocks ever predict past blocks!
        order = np.argsort(np.asarray(time_order, dtype=float), kind="mergesort")
        blocks = np.array_split(order, k)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        b0 = blocks[0]
        if b0.size >= 10 and 2 <= int(yv[b0].sum()) <= (b0.size - 2):
            mid = b0.size // 2
            b0_a, b0_b = b0[:mid], b0[mid:]
            if int(yv[b0_a].sum()) >= 1 and int(yv[b0_b].sum()) >= 1:
                # Within earliest burn-in block only (never using blocks 1..k-1!)
                splits.append((b0_b, b0_a))
                splits.append((b0_a, b0_b))
            else:
                skf0 = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
                for t0, v0 in skf0.split(np.zeros(b0.size), yv[b0]):
                    splits.append((b0[t0], b0[v0]))
        else:
            # Fallback if block 0 has <2 positives: combine blocks 0+1 for burn-in only
            b01 = np.concatenate(blocks[:2])
            skf01 = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            for t0, v0 in skf01.split(np.zeros(b01.size), yv[b01]):
                if np.any(np.isin(b01[v0], b0)):
                    splits.append((b01[t0], np.intersect1d(b01[v0], b0)))

        for j in range(1, len(blocks)):
            past_tr = np.concatenate(blocks[:j])
            va_j = blocks[j]
            if int(yv[past_tr].sum()) >= 1 and int((1 - yv[past_tr]).sum()) >= 1:
                splits.append((past_tr, va_j))
            else:
                # Rare edge case if earliest block had 0 positives
                skf_fb = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                return list(skf_fb.split(np.zeros(n), yv))
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
    """Compute out-of-fold predictions across linear + GBDT families to isolate stable errors.

    Respects entity ``groups`` (StratifiedGroupKFold) and ``time_order`` (causal expanding blocks)
    so OOF residuals never leak repeated patient visits or future regimes.
    """
    y = np.asarray(y_train, dtype=int).ravel()
    n = y.size

    oof_glm = np.zeros(n, dtype=float)
    oof_gbdt = np.zeros(n, dtype=float)

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

    oof_cons = 0.5 * (oof_glm + oof_gbdt)
    glm_hi = np.quantile(oof_glm, alert_tail_quantile)
    gbdt_hi = np.quantile(oof_gbdt, alert_tail_quantile)
    cons_hi = np.quantile(oof_cons, alert_tail_quantile)

    pos_scores = oof_cons[y == 1]
    fn_cut = float(np.quantile(pos_scores, 0.40)) if pos_scores.size > 0 else 0.1

    alert_region = oof_cons >= cons_hi
    conf_fp = (y == 0) & ((oof_glm >= glm_hi) & (oof_gbdt >= gbdt_hi) | alert_region)
    conf_fn = (y == 1) & (oof_glm <= fn_cut) & (oof_gbdt <= fn_cut)
    matched_tp = (y == 1) & alert_region
    matched_tn = (y == 0) & (oof_cons <= fn_cut)

    return ConsensusResidualStrata(
        oof_glm_prob=oof_glm,
        oof_gbdt_prob=oof_gbdt,
        oof_consensus_prob=oof_cons,
        confident_fp_mask=conf_fp,
        confident_fn_mask=conf_fn,
        matched_tp_mask=matched_tp,
        matched_tn_mask=matched_tn,
        alert_region_mask=alert_region,
    )


def sample_case_cohort_pilot(
    y_train: np.ndarray,
    oof_scores: np.ndarray,
    *,
    controls_per_positive: int = 4,
    observation_propensity: np.ndarray | None = None,
    seed: int = 42,
) -> CaseCohortPilot:
    """Sample all positives + score/propensity-matched controls for cheap channel evaluation."""
    return _sample_case_cohort_impl(
        y_train=y_train,
        oof_scores=oof_scores,
        controls_per_positive=controls_per_positive,
        observation_propensity=observation_propensity,
        seed=seed,
    )


def _sample_case_cohort_impl(
    y_train: np.ndarray,
    oof_scores: np.ndarray,
    controls_per_positive: int,
    observation_propensity: np.ndarray | None,
    seed: int,
) -> CaseCohortPilot:
    y = np.asarray(y_train, dtype=int).ravel()
    scores = np.asarray(oof_scores, dtype=float).ravel()
    n = y.size
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if pos_idx.size == 0 or neg_idx.size == 0:
        all_idx = np.arange(n, dtype=int)
        return CaseCohortPilot(
            sampled_indices=all_idx,
            sampling_weights=np.ones(n, dtype=float),
            is_matched_alert_stratum=np.ones(n, dtype=bool),
            fraction_of_cohort_measured=1.0,
        )

    target_n_ctrl = min(neg_idx.size, pos_idx.size * controls_per_positive)
    if target_n_ctrl >= neg_idx.size:
        all_idx = np.arange(n, dtype=int)
        alert_cut = float(np.quantile(scores, 0.85))
        return CaseCohortPilot(
            sampled_indices=all_idx,
            sampling_weights=np.ones(n, dtype=float),
            is_matched_alert_stratum=scores >= alert_cut,
            fraction_of_cohort_measured=1.0,
        )

    rng = np.random.default_rng(seed)
    # 75% of controls score-matched to positives / high-score tail; 25% random anchor controls
    n_matched = int(round(0.75 * target_n_ctrl))
    n_random = target_n_ctrl - n_matched

    # Match in standardized (logit score, observation propensity) space
    s_std = (scores - float(np.mean(scores))) / max(1e-6, float(np.std(scores)))
    if observation_propensity is not None:
        op = np.asarray(observation_propensity, dtype=float).ravel()
        op_std = (op - float(np.mean(op))) / max(1e-6, float(np.std(op)))
        match_coord = 0.75 * s_std + 0.25 * op_std
    else:
        match_coord = s_std

    neg_order = np.argsort(match_coord[neg_idx])
    sorted_neg_idx = neg_idx[neg_order]
    sorted_neg_coord = match_coord[sorted_neg_idx]

    chosen_neg_set: set[int] = set()
    # Greedy nearest-neighbor matching without replacement for each positive
    repeats = max(1, int(np.ceil(n_matched / max(1, pos_idx.size))))
    for p_i in pos_idx:
        target_val = match_coord[p_i]
        insert_pos = int(np.searchsorted(sorted_neg_coord, target_val))
        # Expand outward around insert_pos to find unselected negatives
        picked = 0
        offset = 0
        while picked < repeats and (insert_pos - offset >= 0 or insert_pos + offset < neg_idx.size):
            for cand_pos in (insert_pos + offset, insert_pos - offset):
                if 0 <= cand_pos < neg_idx.size:
                    cand_row = int(sorted_neg_idx[cand_pos])
                    if cand_row not in chosen_neg_set:
                        chosen_neg_set.add(cand_row)
                        picked += 1
                        if len(chosen_neg_set) >= n_matched or picked >= repeats:
                            break
            offset += 1
            if len(chosen_neg_set) >= n_matched:
                break
        if len(chosen_neg_set) >= n_matched:
            break

    remaining_neg = np.array([i for i in neg_idx if int(i) not in chosen_neg_set], dtype=int)
    if n_random > 0 and remaining_neg.size > 0:
        rand_pick = rng.choice(remaining_neg, size=min(n_random, remaining_neg.size), replace=False)
        for r_i in rand_pick:
            chosen_neg_set.add(int(r_i))

    chosen_neg = np.sort(np.array(list(chosen_neg_set), dtype=int))
    sampled = np.sort(np.concatenate([pos_idx, chosen_neg]))

    # Inverse sampling weights for population risk projection
    neg_weight = float(neg_idx.size) / max(1.0, float(chosen_neg.size))
    weights = np.where(y[sampled] == 1, 1.0, neg_weight)
    alert_cut = float(np.quantile(scores[pos_idx], 0.35))
    in_stratum = scores[sampled] >= alert_cut

    return CaseCohortPilot(
        sampled_indices=sampled,
        sampling_weights=weights,
        is_matched_alert_stratum=in_stratum,
        fraction_of_cohort_measured=float(sampled.size / max(1, n)),
    )
