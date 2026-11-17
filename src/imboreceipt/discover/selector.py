"""Stage D (Part 3): Residual Evidence Gain (REG) channel scoring & baseline selectors."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import fisher_exact
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

from imboreceipt.contract.decision import DecisionContract, instance_evidence_deficit
from imboreceipt.core.types import SemanticType
from imboreceipt.discover.strata import CaseCohortPilot, ConsensusResidualStrata
from imboreceipt.frontier.baselines import TabularPreprocessor


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
    reg_likelihood_ratio: float
    negative_tail_coincidence: float
    deficit_covered_nats: float
    p_value: float
    fdr_survives: bool


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


def _fit_channel_oof_signal(
    Z_df: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Compute cross-validated channel probability score on the pilot sample."""
    n = y.size
    n_pos = int(y.sum())
    if n_pos < 4 or (n - n_pos) < 4:
        return np.full(n, float(y.mean()), dtype=float)

    from sklearn.model_selection import StratifiedKFold

    k = min(4, n_pos, n - n_pos)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.zeros(n, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, va in skf.split(np.zeros(n), y):
            prep = TabularPreprocessor().fit(Z_df.iloc[tr])
            X_tr = prep.transform(Z_df.iloc[tr])
            X_va = prep.transform(Z_df.iloc[va])
            sw = sample_weight[tr] if sample_weight is not None else None
            if X_tr.shape[1] <= 3 or n < 300:
                clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
                clf.fit(X_tr, y[tr], sample_weight=sw)
            else:
                clf = LGBMClassifier(
                    n_estimators=80,
                    learning_rate=0.05,
                    num_leaves=15,
                    min_child_samples=10,
                    random_state=seed,
                    n_jobs=1,
                    verbose=-1,
                )
                clf.fit(X_tr, y[tr], sample_weight=sw)
            oof[va] = clf.predict_proba(X_va)[:, 1]
    return oof


def score_channels_credo_reg(
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    strata: ConsensusResidualStrata,
    pilot: CaseCohortPilot,
    contract: DecisionContract,
    *,
    fdr_q: float = 0.10,
    seed: int = 42,
) -> list[ChannelScoreRecord]:
    """Rank candidate channels using CREDO Case-Cohort Residual Evidence Gain (REG).

    Evaluates each candidate channel ONLY on the cheap Case-Cohort Pilot ``pilot.sampled_indices``:
    1. Computes conditional ``REG(B | A) = P(B=1 | A, Y=1) / P(B=1 | A, Y=0)`` inside the
       score-matched alert/residual stratum ``A`` PLUS rescue gain on baseline false negatives.
    2. Penalizes channels whose false positives coincide strongly with the baseline channel
       among true negatives (high ``kappa_0``).
    3. Measures covered Base-Rate Evidence Deficit ``D_q(x)`` per unit acquisition cost,
       subject to Benjamini-Hochberg FDR control.
    """
    y_full = np.asarray(y_train, dtype=int).ravel()
    prev = float(y_full.mean())

    idx = pilot.sampled_indices
    y_p = y_full[idx]
    w_p = pilot.sampling_weights
    base_p = strata.oof_consensus_prob[idx]
    deficits = instance_evidence_deficit(base_p, prev, contract.target_precision)

    # Alert/ambiguous stratum A on pilot + missed FN rescue stratum
    alert_cut = float(np.quantile(base_p[y_p == 1], 0.35)) if np.any(y_p == 1) else 0.1
    in_alert = base_p >= alert_cut
    in_fn_zone = ~in_alert

    raw_records: list[tuple[CandidateChannelSpec, float, float, float, float, float]] = []

    for cand in candidates:
        Z_sub = Z_pool_train.iloc[idx][list(cand.columns)]
        z_prob = _fit_channel_oof_signal(Z_sub, y_p, sample_weight=w_p, seed=seed)

        # Evaluate binary confirmation event B at upper-tail specificity (90th pct of pilot negs)
        neg_z = z_prob[y_p == 0]
        tau_b = float(np.quantile(neg_z, 0.88)) if neg_z.size > 0 else 0.5
        B_event = z_prob >= tau_b

        # 1. Conditional REG(B | A) inside matched high-score stratum A
        a_pos = in_alert & (y_p == 1)
        a_neg = in_alert & (y_p == 0)
        tp_ab = float(np.sum(B_event & a_pos))
        fn_ab = float(np.sum((~B_event) & a_pos))
        fp_ab = float(np.sum(w_p[a_neg] * B_event[a_neg]))
        tn_ab = float(np.sum(w_p[a_neg] * (~B_event[a_neg])))

        # Laplace-smoothed conditional likelihood ratio REG(B | A)
        tpr_b_given_a = (tp_ab + 0.5) / (tp_ab + fn_ab + 1.0)
        fpr_b_given_a = (fp_ab + 0.5) / (fp_ab + tn_ab + 1.0)
        reg_confirm = float(tpr_b_given_a / max(1e-4, fpr_b_given_a))

        # Exact 2x2 Fisher test on unweighted pilot counts inside matched stratum OR rescue zone
        raw_fp = int(np.sum(B_event & a_neg))
        raw_tn = int(np.sum((~B_event) & a_neg))
        _, p_confirm = fisher_exact(
            [[int(tp_ab), int(fn_ab)], [raw_fp, raw_tn]], alternative="greater"
        )

        # 2. Rescue REG on baseline misses (~A)
        r_pos = in_fn_zone & (y_p == 1)
        r_neg = in_fn_zone & (y_p == 0)
        tau_rescue = float(np.quantile(neg_z, 0.95)) if neg_z.size > 0 else 0.5
        R_event = z_prob >= tau_rescue
        tp_r = float(np.sum(R_event & r_pos))
        fn_r = float(np.sum((~R_event) & r_pos))
        fp_r = float(np.sum(w_p[r_neg] * R_event[r_neg]))
        tn_r = float(np.sum(w_p[r_neg] * (~R_event[r_neg])))
        tpr_r = (tp_r + 0.5) / (tp_r + fn_r + 1.0)
        fpr_r = (fp_r + 0.5) / (fp_r + tn_r + 1.0)
        reg_rescue = float(tpr_r / max(1e-4, fpr_r))

        _, p_rescue = fisher_exact(
            [
                [int(tp_r), int(fn_r)],
                [int(np.sum(R_event & r_neg)), int(np.sum((~R_event) & r_neg))],
            ],
            alternative="greater",
        )
        p_best = float(min(1.0, 2.0 * min(p_confirm, p_rescue)))

        # 3. Negative-tail coincidence ratio kappa_0(A, B) among pilot negatives
        neg_only = y_p == 0
        w_neg = w_p[neg_only]
        p_a_neg = float(np.sum(w_neg * in_alert[neg_only]) / np.maximum(1e-9, np.sum(w_neg)))
        p_b_neg = float(np.sum(w_neg * B_event[neg_only]) / np.maximum(1e-9, np.sum(w_neg)))
        p_ab_neg = float(
            np.sum(w_neg * (in_alert[neg_only] & B_event[neg_only]))
            / np.maximum(1e-9, np.sum(w_neg))
        )
        kappa_0 = float((p_ab_neg + 1e-4) / (p_a_neg * p_b_neg + 1e-4))

        # 4. Log-evidence deficit covered where needed (positive instances with D_q(x) > 0)
        eff_reg = max(reg_confirm, reg_rescue)
        log_reg = max(0.0, float(np.log(max(1.0, eff_reg))))
        pos_def = deficits[y_p == 1]
        deficit_cov = float(np.mean(np.minimum(pos_def, log_reg))) if pos_def.size > 0 else log_reg

        # Penalize positive FP correlation (kappa_0 > 1 reduces confirmation value)
        orth_factor = 1.0 / max(0.5, float(np.sqrt(max(0.25, kappa_0))))
        composite = float((log_reg + 0.5 * deficit_cov) * orth_factor)
        raw_records.append((cand, composite, eff_reg, kappa_0, deficit_cov, p_best))

    p_vals = [r[5] for r in raw_records]
    fdr_flags = benjamini_hochberg_mask(p_vals, fdr_q=fdr_q)

    scored: list[ChannelScoreRecord] = []
    for (cand, comp, reg_v, kap, def_c, pv), surv in zip(raw_records, fdr_flags, strict=True):
        scored.append(
            ChannelScoreRecord(
                channel_id=cand.channel_id,
                strategy="credo_reg",
                raw_score=comp,
                score_per_cost=float(comp / max(1e-6, cand.acquisition_cost)),
                reg_likelihood_ratio=reg_v,
                negative_tail_coincidence=kap,
                deficit_covered_nats=def_c,
                p_value=pv,
                fdr_survives=surv,
            )
        )

    scored.sort(key=lambda r: (r.fdr_survives, r.score_per_cost, r.raw_score), reverse=True)
    return scored


def score_channels_baseline_strategy(
    strategy: str,
    X_base_train: pd.DataFrame,
    Z_pool_train: pd.DataFrame,
    y_train: np.ndarray,
    candidates: list[CandidateChannelSpec],
    strata: ConsensusResidualStrata,
    *,
    seed: int = 42,
) -> list[ChannelScoreRecord]:
    """Baseline channel-ranking strategies for CREDO-Bench Predictions 2 & 3 comparisons.

    Supported ``strategy`` values:
    - ``global_mi``: Average global mutual information ``I(Y; Z)`` across channel columns.
    - ``global_shap_gain``: Global GBDT split gain of ``Z`` in joint model ``[X_base, Z]``.
    - ``global_residual``: Unmatched global OOF residual correlation (CRAFTER-style).
    - ``random``: Uniform random ranking under equal acquisition budget.
    """
    y = np.asarray(y_train, dtype=int).ravel()
    rng = np.random.default_rng(seed)
    records: list[ChannelScoreRecord] = []

    if strategy == "global_shap_gain":
        all_z_cols = [c for cand in candidates for c in cand.columns]
        joint_df = pd.concat([X_base_train, Z_pool_train[all_z_cols]], axis=1)
        prep = TabularPreprocessor().fit(joint_df)
        X_j = prep.transform(joint_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gbdt = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=23,
                importance_type="gain",
                random_state=seed,
                n_jobs=1,
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
                    reg_likelihood_ratio=1.0,
                    negative_tail_coincidence=1.0,
                    deficit_covered_nats=0.0,
                    p_value=0.05,
                    fdr_survives=True,
                )
            )
    elif strategy == "global_mi":
        for cand in candidates:
            prep = TabularPreprocessor().fit(Z_pool_train[list(cand.columns)])
            Zm = prep.transform(Z_pool_train[list(cand.columns)])
            mi = float(np.sum(mutual_info_classif(Zm, y, random_state=seed)))
            records.append(
                ChannelScoreRecord(
                    channel_id=cand.channel_id,
                    strategy=strategy,
                    raw_score=mi,
                    score_per_cost=float(mi / max(1e-6, cand.acquisition_cost)),
                    reg_likelihood_ratio=1.0,
                    negative_tail_coincidence=1.0,
                    deficit_covered_nats=0.0,
                    p_value=0.05,
                    fdr_survives=True,
                )
            )
    elif strategy == "global_residual":
        resid = y.astype(float) - strata.oof_consensus_prob
        for cand in candidates:
            z_prob = _fit_channel_oof_signal(Z_pool_train[list(cand.columns)], y, seed=seed)
            # Global residual covariance without local tail matching or negative-orthogonality check
            cov = float(abs(np.corrcoef(resid, z_prob)[0, 1]))
            if not np.isfinite(cov):
                cov = 0.0
            records.append(
                ChannelScoreRecord(
                    channel_id=cand.channel_id,
                    strategy=strategy,
                    raw_score=cov,
                    score_per_cost=float(cov / max(1e-6, cand.acquisition_cost)),
                    reg_likelihood_ratio=1.0,
                    negative_tail_coincidence=1.0,
                    deficit_covered_nats=0.0,
                    p_value=0.05,
                    fdr_survives=True,
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
    # Fallback if strict FDR rejected all on tiny sample: take top affordable positive-REG channel
    if not chosen and ranked:
        for rec in ranked:
            spec = specs[rec.channel_id]
            if spent + spec.acquisition_cost <= max_budget + 1e-9 and rec.raw_score > 0:
                chosen.append(spec)
                break
    return chosen


__all__ = [
    "CandidateChannelSpec",
    "ChannelScoreRecord",
    "SemanticType",
    "benjamini_hochberg_mask",
    "score_channels_baseline_strategy",
    "score_channels_credo_reg",
    "select_channels_within_budget",
]
