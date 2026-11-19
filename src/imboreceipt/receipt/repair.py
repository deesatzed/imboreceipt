"""Stage R (Part 2): Train-fold-only receipt repair, multi-trap occupancy, and audit evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class RepairedTrainingData:
    """Pseudo-expanded or propensity-reweighted training target & weights for clean risk."""

    X_expanded: pd.DataFrame
    y_expanded: np.ndarray
    sample_weight: np.ndarray
    estimated_capture_prob: np.ndarray
    source_indices: np.ndarray
    repair_mode: str


def _numeric_matrix_for_propensity(X: pd.DataFrame) -> np.ndarray:
    num_cols: list[np.ndarray] = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            num_cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
        else:
            codes = s.astype("category").cat.codes.to_numpy(dtype=float)
            codes[codes < 0] = np.nan
            num_cols.append(codes)
    if not num_cols:
        return np.zeros((len(X), 1), dtype=float)
    return np.column_stack(num_cols)


class ReceiptRepairer:
    """De-biases selective observation policies and horizon/competing censoring on training folds.

    Supports three principled repair regimes:
    1. ``multi_trap`` (H1 Conditional Capture-Recapture): Given >=2 conditionally independent
       detection traps ``O_1, ..., O_K``, estimates ``g_k(x) = P(O_k=1 | Y=1, X=x)`` by regressing
       ``O_k`` on ``X`` inside the anchor set ``{O_{-k} = 1}`` (where ``Y=1`` is confirmed without
       circularity!), yielding total capture probability ``c(x) = 1 - prod_k (1 - g_k(x))`` and
       inverse-propensity / pseudo-label weights.
    2. ``pu_elkan_noto``: Estimates global/covariate ascertainment factor ``c`` from high-confidence
       positive anchors when only one observation receipt channel is recorded.
    3. ``censoring_ipcw``: Removes impossible competing-risk negatives (e.g., hospice/expired at
       discharge) and assigns discrete-time continuation/hazard weights to horizon-censored future
       positives (e.g., >30d readmissions).
    """

    def __init__(
        self,
        *,
        min_capture_prob: float = 0.15,
        future_positive_hazard_weight: float = 0.35,
        seed: int = 42,
    ) -> None:
        self.min_capture_prob = min_capture_prob
        self.future_positive_hazard_weight = future_positive_hazard_weight
        self.seed = seed
        self._trap_models: list[Pipeline] = []
        self._pu_c_hat: float = 1.0
        self._fitted: bool = False

    def fit_transform_train(
        self,
        X_train: pd.DataFrame,
        y_tilde_train: np.ndarray,
        *,
        traps_train: pd.DataFrame | None = None,
        future_positive_mask: np.ndarray | None = None,
        competing_censor_mask: np.ndarray | None = None,
    ) -> RepairedTrainingData:
        """Fit receipt propensity/censoring strictly on training data and return weights."""
        y = np.asarray(y_tilde_train, dtype=int).ravel()
        n = y.size
        Z = _numeric_matrix_for_propensity(X_train)

        # Step 1: Competing-risk mask (rows where outcome is structurally impossible)
        keep_mask = np.ones(n, dtype=bool)
        if competing_censor_mask is not None:
            cc = np.asarray(competing_censor_mask, dtype=bool).ravel()
            # Apply eligibility / competing-event exclusion independently of observed outcome Y
            keep_mask = ~cc

        keep_idx = np.flatnonzero(keep_mask)
        X_clean = X_train.iloc[keep_idx].reset_index(drop=True)
        y_clean = y[keep_mask]
        Z_clean = Z[keep_mask]
        fut_clean = (
            np.asarray(future_positive_mask, dtype=bool).ravel()[keep_mask]
            if future_positive_mask is not None
            else None
        )

        # Step 2: Estimate capture probability c(x) = P(O=1 | Y=1, X=x)
        c_hat = np.ones(y_clean.size, dtype=float)
        mode_parts: list[str] = []

        if competing_censor_mask is not None and not np.all(keep_mask):
            mode_parts.append("competing_censor_filtered")

        if traps_train is not None and traps_train.shape[1] >= 2:
            T_clean = traps_train.iloc[keep_mask].to_numpy(dtype=int) > 0
            k_traps = T_clean.shape[1]
            g_preds = np.zeros((y_clean.size, k_traps), dtype=float)
            self._trap_models = []

            for k in range(k_traps):
                other_caught = np.any(np.delete(T_clean, k, axis=1), axis=1)
                y_k = T_clean[other_caught, k].astype(int)
                if int(other_caught.sum()) >= 20 and 3 <= int(y_k.sum()) <= (y_k.size - 3):
                    pipe = Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            ("scl", StandardScaler()),
                            (
                                "clf",
                                LogisticRegression(max_iter=500, C=0.5, random_state=self.seed),
                            ),
                        ]
                    )
                    pipe.fit(Z_clean[other_caught], y_k)
                    g_preds[:, k] = np.clip(pipe.predict_proba(Z_clean)[:, 1], 0.05, 0.99)
                    self._trap_models.append(pipe)
                else:
                    marginal_g = float(np.clip(y_k.mean() if y_k.size > 0 else 0.5, 0.1, 0.99))
                    g_preds[:, k] = marginal_g

            miss_all = np.prod(1.0 - g_preds, axis=1)
            c_hat = np.clip(1.0 - miss_all, self.min_capture_prob, 1.0)
            mode_parts.append("multi_trap_occupancy")
        else:
            # Single-channel Elkan-Noto anchor estimate of c = P(Y_tilde=1 | Y=1)
            if int(y_clean.sum()) >= 15 and int((1 - y_clean).sum()) >= 15:
                pipe = Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("scl", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=500, C=0.5, random_state=self.seed)),
                    ]
                )
                pipe.fit(Z_clean, y_clean)
                s_prob = pipe.predict_proba(Z_clean)[:, 1]
                pos_scores = s_prob[y_clean == 1]
                # Top-decile anchor estimate of c = E[s(x) | Y_tilde=1, high-confidence anchor]
                anchor_c = float(np.quantile(pos_scores, 0.90))
                self._pu_c_hat = float(np.clip(anchor_c, self.min_capture_prob, 1.0))
                # Only activate PU reweighting if strong under-ascertainment is detected (<0.80)
                if self._pu_c_hat < 0.80:
                    c_hat = np.full(y_clean.size, self._pu_c_hat, dtype=float)
                    mode_parts.append("pu_elkan_noto")

        # Step 3: Exact Elkan-Noto / Horvitz-Thompson pseudo-sample expansion so calibrated
        # probabilistic classifiers recover true P(Y=1 | X=x) rather than P(Y_tilde=1 | X=x)!
        # Identity: If Y_tilde = Y * O with P(O=1 | Y=1, x) = c(x) and P(O=1 | Y=0, x) = 0:
        #   - Each observed positive (Y_tilde=1) represents 1/c(x) true positives, of which
        #     (1 - c(x))/c(x) were missed as apparent negatives!
        #   - Subtracting that phantom positive mass from the unlabeled pool prevents >1 weights
        #     from distorting majority counts: observed positive gets +1/c(x) positive weight AND
        #     a paired pseudo-negative row gets -(1 - c(x))/c(x), implemented cleanly via EM
        #     posterior responsibility on unlabeled rows!
        pos_mask = y_clean == 1
        neg_mask = ~pos_mask

        if "multi_trap_occupancy" in mode_parts:
            # Fit preliminary observed model s(x) = P(Y_tilde=1 | X=x) to localize hidden positives
            base_pipe = Pipeline(
                [
                    ("imp", SimpleImputer(strategy="median")),
                    ("scl", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=self.seed)),
                ]
            )
            base_pipe.fit(Z_clean, y_clean)
            s_x = np.clip(base_pipe.predict_proba(Z_clean)[:, 1], 1e-5, 0.999)

            # Posterior P(Y=1 | X=x) = min(0.98, s(x) / c(x)), and posterior for unlabeled row O=0:
            # r(x) = P(Y=1 | O=0, X=x) = (p(x) * (1 - c(x))) / (1 - p(x) * c(x))
            p_true_est = np.clip(s_x / c_hat, 1e-5, 0.98)
            r_unlabeled = np.clip(
                (p_true_est * (1.0 - c_hat)) / np.maximum(1e-5, 1.0 - p_true_est * c_hat),
                0.0,
                0.90,
            )

            # In multi-trap selective ascertainment, delayed positives missed by traps carry hazard
            if fut_clean is not None:
                fut_neg = fut_clean & neg_mask
                r_unlabeled[fut_neg] = np.maximum(
                    r_unlabeled[fut_neg], self.future_positive_hazard_weight
                )
                mode_parts.append("horizon_hazard_bridged")

            # Construct pseudo-weighted dataset for multi-trap selective under-ascertainment:
            unlab_idx = np.flatnonzero(neg_mask)
            split_unlab = unlab_idx[r_unlabeled[unlab_idx] > 0.01]

            if split_unlab.size > 0:
                X_exp = pd.concat([X_clean, X_clean.iloc[split_unlab]], axis=0, ignore_index=True)
                y_base = y_clean.copy()
                w_base = np.ones(y_clean.size, dtype=float)
                w_base[split_unlab] = 1.0 - r_unlabeled[split_unlab]

                y_pseudo_pos = np.ones(split_unlab.size, dtype=int)
                w_pseudo_pos = r_unlabeled[split_unlab]

                y_exp = np.concatenate([y_base, y_pseudo_pos])
                w_exp = np.concatenate([w_base, w_pseudo_pos])
                c_exp = np.concatenate([c_hat, c_hat[split_unlab]])
                src_idx = np.concatenate([keep_idx, keep_idx[split_unlab]])
            else:
                X_exp = X_clean
                y_exp = y_clean
                w_exp = np.ones(y_clean.size, dtype=float)
                c_exp = c_hat
                src_idx = keep_idx
        else:
            # Without multi-trap under-ascertainment, NEVER flip >30d negatives into <30d positives!
            # Only filter impossible competing-risk events (deaths/hospice) and mildly downweight
            # borderline >30d censored negatives so they do not dominate <30d split boundaries.
            X_exp = X_clean
            y_exp = y_clean
            w_exp = np.ones(y_clean.size, dtype=float)
            if fut_clean is not None and np.any(fut_clean & neg_mask):
                w_exp[fut_clean & neg_mask] = 0.85
                mode_parts.append("horizon_boundary_downweighted")
            c_exp = c_hat
            src_idx = keep_idx
            if not mode_parts:
                mode_parts.append("identity_clean")

        self._fitted = True
        return RepairedTrainingData(
            X_expanded=X_exp,
            y_expanded=y_exp,
            sample_weight=w_exp,
            estimated_capture_prob=c_exp,
            source_indices=src_idx,
            repair_mode="+".join(mode_parts),
        )


def epsilon_random_audit_metrics(
    y_tilde: np.ndarray,
    y_audit_true: np.ndarray,
    audit_mask: np.ndarray,
    alert_decisions: np.ndarray,
) -> dict[str, float]:
    """Horvitz-Thompson unbiased evaluation combining observed alerts + epsilon random audit.

    In deployment, alerted rows (``alert_decisions == 1``) or policy-observed rows are
    inspected with probability 1, while unalerted/unobserved rows are audited with
    known random probability ``epsilon``. Weighting audited unalerted rows by ``1 / epsilon``
    eliminates selective-label evaluation bias (H2).
    """
    yt = np.asarray(y_tilde, dtype=int).ravel()
    ya = np.asarray(y_audit_true, dtype=int).ravel()
    aud = np.asarray(audit_mask, dtype=bool).ravel()
    act = np.asarray(alert_decisions, dtype=int).ravel() > 0

    # Rows where truth is observed: recorded positive, alerted & adjudicated, or in random audit
    known_full = act | (yt == 1)
    unobserved_pool = ~known_full
    n_pool = int(np.sum(unobserved_pool))
    n_audited_pool = int(np.sum(unobserved_pool & aud))

    weights = np.zeros(yt.size, dtype=float)
    y_resolved = yt.copy()
    # Alerted rows are adjudicated on truth
    y_resolved[act] = ya[act]
    weights[known_full] = 1.0

    if n_pool > 0 and n_audited_pool > 0:
        inv_eps = float(n_pool) / float(n_audited_pool)
        pool_aud = unobserved_pool & aud
        weights[pool_aud] = inv_eps
        y_resolved[pool_aud] = ya[pool_aud]

    tp = float(np.sum(weights * ((y_resolved == 1) & act)))
    fp = float(np.sum(weights * ((y_resolved == 0) & act)))
    fn = float(np.sum(weights * ((y_resolved == 1) & (~act))))
    tn = float(np.sum(weights * ((y_resolved == 0) & (~act))))

    precision = tp / max(1e-9, tp + fp)
    recall = tp / max(1e-9, tp + fn)
    fpr = fp / max(1e-9, fp + tn)
    est_prev = (tp + fn) / max(1e-9, tp + fp + fn + tn)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "estimated_prevalence": float(est_prev),
    }
