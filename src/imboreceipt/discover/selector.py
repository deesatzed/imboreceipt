"""Stage D (Part 3): Horvitz-Thompson Conditional Information Gain & REG Channel Selectors."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from scipy.stats import fisher_exact, norm
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score

from imboreceipt.contract.decision import DecisionContract, instance_evidence_deficit
from imboreceipt.core.types import SemanticType
from imboreceipt.discover.strata import (
    CaseCohortPilot,
    ConsensusResidualStrata,
    make_leak_free_cv_splits,
)
from imboreceipt.frontier.baselines import TabularPreprocessor

_PROB_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class CandidateChannelSpec:
    """Descriptor for an acquirable or constructible feature channel Z_k."""

    channel_id: str
    columns: tuple[str, ...]
    acquisition_cost: float = 1.0
    mechanism_tag: str = "general"


@dataclass(frozen=True, slots=True)
class ChannelScoreRecord:
    """Evaluation summary for one candidate channel under a selection strategy."""

    channel_id: str
    strategy: str
    raw_score: float
    score_per_cost: float
    conditional_logloss_gain_nats: float
    reg_likelihood_ratio: float
    negative_tail_coincidence: float
    deficit_covered_nats: float
    p_value: float
    fdr_survives: bool
    temporal_cv_used: bool = False


def benjamini_hochberg_mask(p_values: list[float], fdr_q: float = 0.10) -> list[bool]:
    """Apply Benjamini-Hochberg multiplicity control to candidate p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    survives = [False] * m
    max_k = -1
    for rank_0, orig_idx in enumerate(order):
        thresh = ((rank_0 + 1) / m) * fdr_q
        if p_values[orig_idx] <= thresh:
            max_k = rank_0
    if max_k >= 0:
        for rank_0 in range(max_k + 1):
            survives[int(order[rank_0])] = True
    return survives


def _fit_oof_probs_on_splits(
    df_features: pd.DataFrame,
    y: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    sample_weight: np.ndarray | None = None,
    *,
    init_logit: np.ndarray | None = None,
    semantic_types: dict[str, SemanticType] | None = None,
    use_gbdt: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit identical offset-anchored + fold-calibrated learner out-of-fold on ``cv_splits``.

    When ``init_logit`` is supplied (e.g. out-of-fold baseline log-odds), both the baseline
    refinement model and augmented channel model warm-start from ``init_logit`` and apply
    fold-internal Horvitz-Thompson weighted logistic calibration on ``[init_logit, raw_margin]``,
    preventing tree-overfitting log-loss degradation and bagging-clone inflation on small pilots.
    """
    from scipy.special import expit

    n = y.size
    sw_all = (
        np.ones(n, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    )
    # Normalize weights to mean 1.0 for numerical stability in regularized fitting
    sw_norm = sw_all / max(1e-9, float(np.mean(sw_all)))
    w_prior = float(np.sum(sw_norm * y) / max(1e-9, float(np.sum(sw_norm))))
    w_prior = float(np.clip(w_prior, _PROB_EPS, 1.0 - _PROB_EPS))

    oof = (
        np.clip(expit(init_logit), _PROB_EPS, 1.0 - _PROB_EPS)
        if init_logit is not None
        else np.full(n, w_prior, dtype=float)
    )
    valid = np.zeros(n, dtype=bool)
    if not cv_splits:
        return oof, valid

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in cv_splits:
            if int(y[tr].sum()) < 2 or int((1 - y[tr]).sum()) < 2:
                continue
            prep = TabularPreprocessor(semantic_types=semantic_types).fit(df_features.iloc[tr])
            X_tr = prep.transform(df_features.iloc[tr])
            X_va = prep.transform(df_features.iloc[va])
            sw_tr = sw_norm[tr]

            if (not use_gbdt) or n < 100:
                if init_logit is not None:
                    X_tr_aug = np.column_stack([init_logit[tr], X_tr])
                    X_va_aug = np.column_stack([init_logit[va], X_va])
                else:
                    X_tr_aug, X_va_aug = X_tr, X_va
                clf_lr = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
                clf_lr.fit(X_tr_aug, y[tr], sample_weight=sw_tr)
                oof[va] = np.clip(clf_lr.predict_proba(X_va_aug)[:, 1], _PROB_EPS, 1.0 - _PROB_EPS)
            else:
                clf = LGBMClassifier(
                    n_estimators=40,
                    learning_rate=0.05,
                    max_depth=3,
                    num_leaves=7,
                    min_child_samples=max(5, min(15, tr.size // 20)),
                    reg_lambda=5.0,
                    max_delta_step=1.0,
                    random_state=seed,
                    n_jobs=4,
                    verbose=-1,
                )
                init_tr = init_logit[tr] if init_logit is not None else None
                init_va = init_logit[va] if init_logit is not None else None
                clf.fit(X_tr, y[tr], sample_weight=sw_tr, init_score=init_tr)
                raw_tr_margin = np.asarray(clf.predict(X_tr, raw_score=True), dtype=float)
                raw_va_margin = np.asarray(clf.predict(X_va, raw_score=True), dtype=float)
                if init_tr is not None and init_va is not None:
                    cal_X_tr = np.column_stack([init_tr, raw_tr_margin])
                    cal_X_va = np.column_stack([init_va, raw_va_margin])
                else:
                    cal_X_tr = raw_tr_margin.reshape(-1, 1)
                    cal_X_va = raw_va_margin.reshape(-1, 1)
                cal_lr = LogisticRegression(C=0.5, max_iter=400, random_state=seed)
                cal_lr.fit(cal_X_tr, y[tr], sample_weight=sw_tr)
                oof[va] = np.clip(cal_lr.predict_proba(cal_X_va)[:, 1], _PROB_EPS, 1.0 - _PROB_EPS)
            valid[va] = True
    return oof, valid


def _bernoulli_loglik(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, _PROB_EPS, 1.0 - _PROB_EPS)
    return y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)


def _weighted_gain_and_pvalue(
    ll_diff: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute Horvitz-Thompson weighted mean log-likelihood gain and one-sided normal p-value."""
    if ll_diff.size == 0:
        return 0.0, 1.0
    w = np.asarray(weights, dtype=float)
    w_sum = max(1e-9, float(np.sum(w)))
    mean_gain = float(np.sum(w * ll_diff) / w_sum)

    # Cluster-robust or observation-level variance of Horvitz-Thompson ratio estimator
    centered = w * (ll_diff - mean_gain)
    if groups is not None and groups.size == ll_diff.size:
        _, inv = np.unique(groups, return_inverse=True)
        cluster_totals = np.bincount(inv, weights=centered)
        n_eff = int(cluster_totals.size)
        if n_eff < 3:
            return mean_gain, (0.05 if mean_gain > 0 else 1.0)
        var_num = float(n_eff / max(1, n_eff - 1)) * float(np.sum(cluster_totals**2))
    else:
        n_eff = int(ll_diff.size)
        if n_eff < 3:
            return mean_gain, (0.05 if mean_gain > 0 else 1.0)
        var_num = float(n_eff / max(1, n_eff - 1)) * float(np.sum(centered**2))

    se = float(np.sqrt(max(0.0, var_num)) / w_sum)
    if se <= 1e-12:
        return mean_gain, (0.0 if mean_gain > 0 else 1.0)
    z_stat = mean_gain / se
    p_one_sided = float(norm.sf(z_stat))
    return mean_gain, p_one_sided


def score_channels_conditional_gain(
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    strata: ConsensusResidualStrata,
    pilot: CaseCohortPilot,
    *,
    base_semantic_types: dict[str, SemanticType] | None = None,
    groups_train: np.ndarray | None = None,
    time_order_train: np.ndarray | None = None,
    strategy_name: str = "pilot_cmi",
    fdr_q: float = 0.10,
    seed: int = 42,
) -> list[ChannelScoreRecord]:
    """Estimate Horvitz-Thompson weighted conditional log-likelihood gain G_m on pilot P.

    At the population truth,
        G_m = E[ log P(Y | X_base, Z_m) - log P(Y | X_base) ] = I(Y; Z_m | X_base).
    Both the baseline-only model ``[X_base, base_logit]`` and each augmented model
    ``[X_base, base_logit, Z_m]`` are trained with the EXACT SAME regularized GBDT learner,
    EXACT SAME leak-free (group/chronological) CV folds, and EXACT SAME Horvitz-Thompson
    inverse-probability weights ``w_i = 1 / pi_i`` on ``pilot.sampled_indices``:
    - Preserves full ``[X_base, Z_m]`` feature interactions (solving compressed-score blindness).
    - Eliminates sampling-design log-odds artifacts between baseline and augmented fits.
    - Never silently disables forward chronological cross-fitting when ``time_order_train`` is set.
    """
    y_full = np.asarray(y_train, dtype=int).ravel()
    idx = pilot.sampled_indices
    y_p = y_full[idx]
    w_p = pilot.sampling_weights
    grp_p = groups_train[idx] if groups_train is not None else None
    time_p = time_order_train[idx] if time_order_train is not None else None

    cv_splits = make_leak_free_cv_splits(y_p, n_folds=4, groups=grp_p, time_order=time_p, seed=seed)
    temporal_used = bool(time_p is not None and len(cv_splits) > 0)

    # Full baseline representation on pilot: raw X_base columns anchored on OOF base_logit offset
    base_df_p = X_base_train.iloc[idx].reset_index(drop=True).copy()
    base_p_oof = np.clip(strata.oof_consensus_prob[idx], _PROB_EPS, 1.0 - _PROB_EPS)
    base_logit_p = logit(base_p_oof)
    sem_p = dict(base_semantic_types) if base_semantic_types is not None else {}

    p0_oof, valid_0 = _fit_oof_probs_on_splits(
        base_df_p,
        y_p,
        cv_splits,
        sample_weight=w_p,
        init_logit=base_logit_p,
        semantic_types=sem_p,
        use_gbdt=True,
        seed=seed,
    )
    ll_0 = _bernoulli_loglik(y_p, p0_oof)

    raw_rows: list[tuple[CandidateChannelSpec, float, float]] = []
    for cand in candidates:
        Z_sub = Z_pool_train.iloc[idx][list(cand.columns)].reset_index(drop=True)
        aug_df = pd.concat([base_df_p, Z_sub], axis=1)
        pm_oof, valid_m = _fit_oof_probs_on_splits(
            aug_df,
            y_p,
            cv_splits,
            sample_weight=w_p,
            init_logit=base_logit_p,
            semantic_types=sem_p,
            use_gbdt=True,
            seed=seed,
        )
        eval_mask = valid_0 & valid_m
        if not np.any(eval_mask):
            raw_rows.append((cand, 0.0, 1.0))
            continue
        ll_m = _bernoulli_loglik(y_p, pm_oof)
        gain_m, pv_m = _weighted_gain_and_pvalue(
            (ll_m - ll_0)[eval_mask],
            w_p[eval_mask],
            groups=grp_p[eval_mask] if grp_p is not None else None,
        )
        raw_rows.append((cand, gain_m, pv_m))

    p_vals = [r[2] for r in raw_rows]
    fdr_flags = benjamini_hochberg_mask(p_vals, fdr_q=fdr_q)

    records: list[ChannelScoreRecord] = []
    for (cand, gain_m, pv_m), surv in zip(raw_rows, fdr_flags, strict=True):
        pos_gain = max(0.0, gain_m)
        records.append(
            ChannelScoreRecord(
                channel_id=cand.channel_id,
                strategy=strategy_name,
                raw_score=gain_m,
                score_per_cost=float(gain_m / max(1e-6, cand.acquisition_cost)),
                conditional_logloss_gain_nats=gain_m,
                reg_likelihood_ratio=float(np.exp(np.clip(pos_gain, 0.0, 10.0))),
                negative_tail_coincidence=1.0,
                deficit_covered_nats=pos_gain,
                p_value=pv_m,
                fdr_survives=surv and (gain_m > 0.0),
                temporal_cv_used=temporal_used,
            )
        )

    records.sort(key=lambda r: (r.fdr_survives, r.score_per_cost, r.raw_score), reverse=True)
    return records


def score_channels_credo_reg(
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    strata: ConsensusResidualStrata,
    pilot: CaseCohortPilot,
    contract: DecisionContract,
    *,
    base_semantic_types: dict[str, SemanticType] | None = None,
    groups_train: np.ndarray | None = None,
    time_order_train: np.ndarray | None = None,
    fdr_q: float = 0.10,
    seed: int = 42,
) -> list[ChannelScoreRecord]:
    """Rank candidate channels using CREDO composite REG + Horvitz-Thompson conditional gain.

    Reconciles empirical REG scoring with strict design hygiene:
    1. Fits baseline ``[X_base, base_logit]`` and augmented ``[X_base, base_logit, Z_m]`` models
       on identical CV splits with identical Horvitz-Thompson weights (preserving interactions
       with ``X_base`` and never silently disabling chronological CV).
    2. Combines Horvitz-Thompson conditional log-likelihood gain ``G_m`` with empirical
       alert-zone / rescue-zone likelihood ratios and negative-tail coincidence ``kappa_0``
       discounting (reported explicitly as an empirical composite design choice alongside pure
       ``pilot_cmi``).
    """
    y_full = np.asarray(y_train, dtype=int).ravel()
    prev = float(y_full.mean())

    idx = pilot.sampled_indices
    y_p = y_full[idx]
    w_p = pilot.sampling_weights
    base_p = np.clip(strata.oof_consensus_prob[idx], _PROB_EPS, 1.0 - _PROB_EPS)
    grp_p = groups_train[idx] if groups_train is not None else None
    time_p = time_order_train[idx] if time_order_train is not None else None
    deficits = instance_evidence_deficit(base_p, prev, contract.target_precision)

    cv_splits = make_leak_free_cv_splits(y_p, n_folds=4, groups=grp_p, time_order=time_p, seed=seed)
    temporal_used = bool(time_p is not None and len(cv_splits) > 0)

    base_df_p = X_base_train.iloc[idx].reset_index(drop=True).copy()
    base_logit_p = logit(base_p)
    sem_p = dict(base_semantic_types) if base_semantic_types is not None else {}

    p0_oof, valid_0 = _fit_oof_probs_on_splits(
        base_df_p,
        y_p,
        cv_splits,
        sample_weight=w_p,
        init_logit=base_logit_p,
        semantic_types=sem_p,
        use_gbdt=True,
        seed=seed,
    )
    ll_0 = _bernoulli_loglik(y_p, p0_oof)
    l0_oof = logit(p0_oof)

    eval_base = valid_0 if np.any(valid_0) else np.ones(y_p.size, dtype=bool)
    pos_in_eval = eval_base & (y_p == 1)
    alert_cut = float(np.quantile(p0_oof[pos_in_eval], 0.35)) if np.any(pos_in_eval) else 0.1
    in_alert = eval_base & (p0_oof >= alert_cut)
    in_fn_zone = eval_base & (~in_alert)

    raw_records: list[tuple[CandidateChannelSpec, float, float, float, float, float, float]] = []

    for cand in candidates:
        Z_sub = Z_pool_train.iloc[idx][list(cand.columns)].reset_index(drop=True)
        aug_df = pd.concat([base_df_p, Z_sub], axis=1)
        pm_oof, valid_m = _fit_oof_probs_on_splits(
            aug_df,
            y_p,
            cv_splits,
            sample_weight=w_p,
            init_logit=base_logit_p,
            semantic_types=sem_p,
            use_gbdt=True,
            seed=seed,
        )
        z_prob, _ = _fit_oof_probs_on_splits(
            Z_sub,
            y_p,
            cv_splits,
            sample_weight=w_p,
            use_gbdt=(Z_sub.shape[1] > 2),
            seed=seed + 7,
        )

        ev = valid_0 & valid_m
        if not np.any(ev):
            raw_records.append((cand, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0))
            continue

        ll_m = _bernoulli_loglik(y_p, pm_oof)
        gain_m, p_cmi = _weighted_gain_and_pvalue(
            (ll_m - ll_0)[ev],
            w_p[ev],
            groups=grp_p[ev] if grp_p is not None else None,
        )

        delta_llr = logit(pm_oof) - l0_oof
        neg_d = delta_llr[ev & (y_p == 0)]
        tau_b = float(np.quantile(neg_d, 0.88)) if neg_d.size > 0 else 0.0
        B_event = delta_llr > tau_b

        a_pos = in_alert & ev & (y_p == 1)
        a_neg = in_alert & ev & (y_p == 0)
        tp_ab = float(np.sum(B_event & a_pos))
        fn_ab = float(np.sum((~B_event) & a_pos))
        fp_ab = float(np.sum(w_p[a_neg] * B_event[a_neg]))
        tn_ab = float(np.sum(w_p[a_neg] * (~B_event[a_neg])))
        tpr_b_given_a = (tp_ab + 0.5) / (tp_ab + fn_ab + 1.0)
        fpr_b_given_a = (fp_ab + 0.5) / (fp_ab + tn_ab + 1.0)
        reg_confirm = float(tpr_b_given_a / max(1e-4, fpr_b_given_a))

        _, p_confirm = fisher_exact(
            [
                [int(tp_ab), int(fn_ab)],
                [int(np.sum(B_event & a_neg)), int(np.sum((~B_event) & a_neg))],
            ],
            alternative="greater",
        )

        r_pos = in_fn_zone & ev & (y_p == 1)
        r_neg = in_fn_zone & ev & (y_p == 0)
        tau_rescue = float(np.quantile(neg_d, 0.94)) if neg_d.size > 0 else 0.0
        R_event = delta_llr > tau_rescue
        tp_r = float(np.sum(R_event & r_pos))
        fn_r = float(np.sum((~R_event) & r_pos))
        fp_r = float(np.sum(w_p[r_neg] * R_event[r_neg]))
        tn_r = float(np.sum(w_p[r_neg] * (~R_event[r_neg])))
        reg_rescue = float(
            ((tp_r + 0.5) / (tp_r + fn_r + 1.0)) / max(1e-4, (fp_r + 0.5) / (fp_r + tn_r + 1.0))
        )

        _, p_rescue = fisher_exact(
            [
                [int(tp_r), int(fn_r)],
                [int(np.sum(R_event & r_neg)), int(np.sum((~R_event) & r_neg))],
            ],
            alternative="greater",
        )
        p_best = float(min(1.0, 3.0 * min(p_cmi, p_confirm, p_rescue)))

        # Negative-tail coincidence kappa_0 among evaluated negatives
        neg_only = ev & (y_p == 0)
        neg_z = z_prob[neg_only]
        tau_z_raw = float(np.quantile(neg_z, 0.88)) if neg_z.size > 0 else 0.5
        B_raw_neg = neg_z > tau_z_raw
        w_neg = w_p[neg_only]
        w_neg_sum = max(1e-9, float(np.sum(w_neg)))
        p_a_neg = float(np.sum(w_neg * in_alert[neg_only]) / w_neg_sum)
        p_b_neg = float(np.sum(w_neg * B_raw_neg) / w_neg_sum)
        p_ab_neg = float(np.sum(w_neg * (in_alert[neg_only] & B_raw_neg)) / w_neg_sum)
        kappa_0 = float((p_ab_neg + 1e-4) / (p_a_neg * p_b_neg + 1e-4))

        eff_reg = max(reg_confirm, reg_rescue)
        log_reg = max(0.0, float(np.log(max(1.0, eff_reg))))
        pos_def = deficits[ev & (y_p == 1)]
        deficit_cov = float(np.mean(np.minimum(pos_def, log_reg))) if pos_def.size > 0 else log_reg

        # Empirical composite combines HT conditional log-loss gain G_m with kappa_0 discount:
        # Note that unlike collinearity discounting on standalone Z, G_m > 0 preserves channels
        # that correlate with X_base yet carry genuine conditional residual signal!
        orth_factor = 1.0 / max(0.65, float(np.sqrt(max(0.25, kappa_0))))
        composite = float(
            (max(0.0, gain_m) * 10.0 + 0.35 * log_reg + 0.15 * deficit_cov) * orth_factor
        )
        if gain_m <= 0.0:
            composite = float(gain_m)
        raw_records.append((cand, composite, gain_m, eff_reg, kappa_0, deficit_cov, p_best))

    p_vals = [r[6] for r in raw_records]
    fdr_flags = benjamini_hochberg_mask(p_vals, fdr_q=fdr_q)

    scored: list[ChannelScoreRecord] = []
    for (cand, comp, gm, reg_v, kap, def_c, pv), surv in zip(raw_records, fdr_flags, strict=True):
        scored.append(
            ChannelScoreRecord(
                channel_id=cand.channel_id,
                strategy="credo_reg",
                raw_score=comp,
                score_per_cost=float(comp / max(1e-6, cand.acquisition_cost)),
                conditional_logloss_gain_nats=gm,
                reg_likelihood_ratio=reg_v,
                negative_tail_coincidence=kap,
                deficit_covered_nats=def_c,
                p_value=pv,
                fdr_survives=surv and (gm > 0.0),
                temporal_cv_used=temporal_used,
            )
        )

    scored.sort(key=lambda r: (r.fdr_survives, r.score_per_cost, r.raw_score), reverse=True)
    return scored


def _weighted_binned_mi(y: np.ndarray, col_v: np.ndarray, weights: np.ndarray | None) -> float:
    """Compute weighted (or unweighted) Miller-Madow bias-corrected quantile-binned MI."""
    q_grid = np.linspace(0.0, 1.0, 33)
    edges = np.unique(np.quantile(col_v, q_grid))
    bins = (
        np.searchsorted(edges[1:-1], col_v, side="right")
        if edges.size > 2
        else (col_v > edges[0]).astype(int)
    )
    if weights is None:
        raw_mi = float(mutual_info_score(y, bins))
        n_eff = max(1, y.size)
    else:
        w = np.asarray(weights, dtype=float)
        w_tot = max(1e-9, float(np.sum(w)))
        n_bins = int(np.max(bins)) + 1
        joint = np.zeros((2, n_bins), dtype=float)
        np.add.at(joint, (y, bins), w)
        joint /= w_tot
        py = joint.sum(axis=1, keepdims=True)
        pb = joint.sum(axis=0, keepdims=True)
        denom = py @ pb
        nz = (joint > 0) & (denom > 0)
        raw_mi = float(np.sum(joint[nz] * np.log(joint[nz] / denom[nz])))
        n_eff = max(1, int((w_tot**2) / max(1e-9, float(np.sum(w**2)))))

    n_bins_eff = max(1, int(np.unique(bins).size))
    mm_bias = float(n_bins_eff - 1) / float(2 * n_eff)
    return max(0.0, raw_mi - mm_bias)


def score_channels_baseline_strategy(
    strategy: str,
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    strata: ConsensusResidualStrata,
    *,
    pilot: CaseCohortPilot | None = None,
    base_semantic_types: dict[str, SemanticType] | None = None,
    groups_train: np.ndarray | None = None,
    time_order_train: np.ndarray | None = None,
    seed: int = 42,
) -> list[ChannelScoreRecord]:
    """Channel-ranking comparators for single-factor ablation experiments.

    Supported ``strategy`` values:
    - ``global_mi``: Full-cohort (100% rows) marginal mutual information ``I(Y; Z)``.
    - ``pilot_mi``: Marginal MI evaluated on the exact ``pilot`` rows with HT weights.
    - ``unmatched_case_control_mi``: Classical unmatched case-control marginal MI.
    - ``full_cohort_cmi``: Full-cohort (100% rows) conditional log-likelihood gain ``G_m``.
    - ``global_shap_gain``: Global GBDT split gain of ``Z`` in joint model ``[X_base, Z]``.
    - ``global_residual``: Full-cohort OOF residual correlation.
    - ``random``: Uniform random ranking under equal acquisition budget.
    """
    y = np.asarray(y_train, dtype=int).ravel()
    rng = np.random.default_rng(seed)
    records: list[ChannelScoreRecord] = []

    if strategy == "full_cohort_cmi":
        elig = np.flatnonzero(strata.oof_valid_mask)
        if elig.size == 0:
            elig = np.arange(y.size, dtype=int)
        full_pilot = CaseCohortPilot(
            sampled_indices=elig,
            sampling_weights=np.ones(elig.size, dtype=float),
            inclusion_probabilities=np.ones(elig.size, dtype=float),
            stratum_ids=np.zeros(elig.size, dtype=int),
            is_matched_alert_stratum=np.ones(elig.size, dtype=bool),
            fraction_of_cohort_measured=1.0,
            fraction_of_eligible_measured=1.0,
            matching_mode="full_cohort",
        )
        return score_channels_conditional_gain(
            X_base_train,
            Z_pool_train,
            y,
            candidates,
            strata,
            full_pilot,
            base_semantic_types=base_semantic_types,
            groups_train=groups_train,
            time_order_train=time_order_train,
            strategy_name="full_cohort_cmi",
            seed=seed,
        )

    if strategy == "global_shap_gain":
        all_z_cols = [c for cand in candidates for c in cand.columns]
        joint_df = pd.concat(
            [X_base_train.reset_index(drop=True), Z_pool_train[all_z_cols].reset_index(drop=True)],
            axis=1,
        )
        prep = TabularPreprocessor().fit(joint_df)
        X_j = prep.transform(joint_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gbdt = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=23,
                max_delta_step=1.0,
                importance_type="gain",
                random_state=seed,
                n_jobs=4,
                verbose=-1,
            )
            gbdt.fit(X_j, y)
        gains = gbdt.feature_importances_.astype(float)
        feat_names = prep.feature_names_out_
        for cand in candidates:
            c_gain = 0.0
            for col in cand.columns:
                for fn, g in zip(feat_names, gains, strict=True):
                    if fn == col or fn.startswith(f"{col}_") or fn == f"miss__{col}":
                        c_gain += float(g)
            records.append(
                ChannelScoreRecord(
                    channel_id=cand.channel_id,
                    strategy=strategy,
                    raw_score=c_gain,
                    score_per_cost=float(c_gain / max(1e-6, cand.acquisition_cost)),
                    conditional_logloss_gain_nats=0.0,
                    reg_likelihood_ratio=1.0,
                    negative_tail_coincidence=1.0,
                    deficit_covered_nats=0.0,
                    p_value=0.05,
                    fdr_survives=True,
                )
            )
    elif strategy in {"global_mi", "pilot_mi", "unmatched_case_control_mi"}:
        w_eval: np.ndarray | None = None
        if strategy == "pilot_mi" and pilot is not None:
            sub_idx = pilot.sampled_indices
            y_eval = y[sub_idx]
            Z_eval = Z_pool_train.iloc[sub_idx].reset_index(drop=True)
            w_eval = pilot.sampling_weights
        elif strategy == "unmatched_case_control_mi":
            pos_i = np.flatnonzero(y == 1)
            neg_i = np.flatnonzero(y == 0)
            n_ctrl = min(neg_i.size, pos_i.size * 4)
            ctrl_i = rng.choice(neg_i, size=n_ctrl, replace=False) if n_ctrl < neg_i.size else neg_i
            sub_idx = np.sort(np.concatenate([pos_i, ctrl_i]))
            y_eval = y[sub_idx]
            Z_eval = Z_pool_train.iloc[sub_idx].reset_index(drop=True)
        else:
            y_eval = y
            Z_eval = Z_pool_train

        for cand in candidates:
            prep = TabularPreprocessor().fit(Z_eval[list(cand.columns)])
            Zm = prep.transform(Z_eval[list(cand.columns)])
            if w_eval is None and y_eval.size <= 35000:
                col_mis = np.asarray(
                    mutual_info_classif(Zm, y_eval, random_state=seed), dtype=float
                )
            else:
                col_mis = np.asarray(
                    [_weighted_binned_mi(y_eval, Zm[:, j], w_eval) for j in range(Zm.shape[1])],
                    dtype=float,
                )

            top_k = min(2, max(1, col_mis.size))
            mi = float(np.mean(np.sort(col_mis)[-top_k:]))
            records.append(
                ChannelScoreRecord(
                    channel_id=cand.channel_id,
                    strategy=strategy,
                    raw_score=mi,
                    score_per_cost=float(mi / max(1e-6, cand.acquisition_cost)),
                    conditional_logloss_gain_nats=0.0,
                    reg_likelihood_ratio=1.0,
                    negative_tail_coincidence=1.0,
                    deficit_covered_nats=0.0,
                    p_value=0.05,
                    fdr_survives=True,
                )
            )
    elif strategy == "global_residual":
        cv_splits = make_leak_free_cv_splits(
            y, n_folds=4, groups=groups_train, time_order=time_order_train, seed=seed
        )
        resid = y.astype(float) - strata.oof_consensus_prob
        for cand in candidates:
            z_prob, val_m = _fit_oof_probs_on_splits(
                Z_pool_train[list(cand.columns)],
                y,
                cv_splits,
                use_gbdt=(len(cand.columns) > 2),
                seed=seed,
            )
            ev = val_m & strata.oof_valid_mask
            if np.sum(ev) > 5 and np.std(z_prob[ev]) > 1e-8 and np.std(resid[ev]) > 1e-8:
                cov = float(abs(np.corrcoef(resid[ev], z_prob[ev])[0, 1]))
            else:
                cov = 0.0
            if not np.isfinite(cov):
                cov = 0.0
            records.append(
                ChannelScoreRecord(
                    channel_id=cand.channel_id,
                    strategy=strategy,
                    raw_score=cov,
                    score_per_cost=float(cov / max(1e-6, cand.acquisition_cost)),
                    conditional_logloss_gain_nats=0.0,
                    reg_likelihood_ratio=1.0,
                    negative_tail_coincidence=1.0,
                    deficit_covered_nats=0.0,
                    p_value=0.05,
                    fdr_survives=True,
                    temporal_cv_used=bool(time_order_train is not None and len(cv_splits) > 0),
                )
            )
    elif strategy == "random":
        for cand in candidates:
            sc = float(rng.uniform(0.01, 1.0))
            records.append(
                ChannelScoreRecord(
                    channel_id=cand.channel_id,
                    strategy=strategy,
                    raw_score=sc,
                    score_per_cost=sc,
                    conditional_logloss_gain_nats=0.0,
                    reg_likelihood_ratio=1.0,
                    negative_tail_coincidence=1.0,
                    deficit_covered_nats=0.0,
                    p_value=0.5,
                    fdr_survives=True,
                )
            )
    else:
        raise ValueError(f"Unknown channel selection strategy: {strategy}")

    records.sort(key=lambda r: (r.score_per_cost, r.raw_score), reverse=True)
    return records


def select_channels_within_budget(
    ranked: list[ChannelScoreRecord],
    specs: dict[str, CandidateChannelSpec],
    max_budget: float,
    require_fdr: bool = False,
) -> list[CandidateChannelSpec]:
    """Greedily acquire highest-value channels within total acquisition dollar budget."""
    chosen: list[CandidateChannelSpec] = []
    spent = 0.0
    for rec in ranked:
        if require_fdr and not rec.fdr_survives:
            continue
        spec = specs[rec.channel_id]
        if spent + spec.acquisition_cost <= max_budget + 1e-9:
            chosen.append(spec)
            spent += spec.acquisition_cost
    # If strict BH-FDR left unspent budget on a low-count pilot, fill remaining budget with
    # top-ranked positive-gain candidates so equal-budget comparisons spend identical budget.
    if spent < max_budget - 1e-9 and ranked:
        chosen_set = {c.channel_id for c in chosen}
        for rec in ranked:
            if rec.channel_id in chosen_set:
                continue
            spec = specs[rec.channel_id]
            if spent + spec.acquisition_cost <= max_budget + 1e-9 and rec.raw_score > 0.0:
                chosen.append(spec)
                chosen_set.add(rec.channel_id)
                spent += spec.acquisition_cost
    return chosen


__all__ = [
    "CandidateChannelSpec",
    "ChannelScoreRecord",
    "SemanticType",
    "benjamini_hochberg_mask",
    "score_channels_baseline_strategy",
    "score_channels_conditional_gain",
    "score_channels_credo_reg",
    "select_channels_within_budget",
]
