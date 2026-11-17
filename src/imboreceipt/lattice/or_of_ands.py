"""Stage O (Part 2): Asymmetric Confirm-and-Rescue OR-of-ANDs Evidence Lattice."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from imboreceipt.core.types import SemanticType
from imboreceipt.frontier.baselines import (
    FittedDecoderArm,
    PlattLogitCalibrator,
    TabularPreprocessor,
    fit_single_calibrated_gbdt,
)
from imboreceipt.frontier.ceilings import select_neyman_pearson_threshold
from imboreceipt.lattice.coincidence import (
    TailCoincidenceReport,
    estimate_tail_coincidence_surrogate,
)


@dataclass(slots=True)
class MechanismBranch:
    """One confirmation + rescue pair associated with an acquired channel Z_m."""

    channel_id: str
    confirm_arm: FittedDecoderArm
    rescue_arm: FittedDecoderArm
    coincidence_report: TailCoincidenceReport
    orthogonality_weight: float
    tau_scout: float = 0.5
    tau_confirm: float = 0.5
    tau_rescue: float = 0.9


class ConfirmRescueLattice:
    """Asymmetric OR-of-ANDs Evidence Lattice separating Confirmation from Rescue.

    Boolean decision rule at certified FPR budget ``alpha``:
      ``Y_hat = OR_m (A_scout >= tau_A_m AND B_confirm_m >= tau_B_m) OR_m (C_rescue_m >= tau_C_m)``

    Key structural properties:
    1. **Confirmation models ``B_confirm_m``** are trained STRICTLY on acquired channel ``Z_m``
       alone (not concatenated with ``X_base``), preserving negative-tail independence ``kappa_0``
       so specificity multiplies (H3 Two-Signal Coincidence Rule).
    2. **Rescue models ``C_rescue_m``** are trained on ``[X_base, Z_m]`` with upweighted scout
       false-negative instances to recover distinct Anna Karenina minority subtypes missed by
       ``X_base`` without polluting confirmation thresholds.
    3. Emits both exact Boolean lattice alerts AND a calibrated continuous log-odds lattice
       posterior for full precision-recall frontier evaluation.
    """

    def __init__(
        self,
        *,
        max_fpr_budget: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.max_fpr_budget = max_fpr_budget
        self.seed = seed
        self.scout_arm_: FittedDecoderArm | None = None
        self.branches_: list[MechanismBranch] = []
        self._meta_calibrator: LogisticRegression | None = None
        self._prevalence: float = 0.05
        self._fitted = False

    def fit(
        self,
        X_base_train: pd.DataFrame,
        Z_channels_train: dict[str, pd.DataFrame],
        y_train: np.ndarray,
        X_base_cal: pd.DataFrame,
        Z_channels_cal: dict[str, pd.DataFrame],
        y_cal: np.ndarray,
        X_base_val: pd.DataFrame,
        Z_channels_val: dict[str, pd.DataFrame],
        y_val: np.ndarray,
        *,
        base_semantic_types: dict[str, SemanticType] | None = None,
        sample_weight_train: np.ndarray | None = None,
    ) -> ConfirmRescueLattice:
        """Fit Scout, Confirmation, and Rescue branches and optimize lattice thresholds on val."""
        y_tr = np.asarray(y_train, dtype=int).ravel()
        y_c = np.asarray(y_cal, dtype=int).ravel()
        y_v = np.asarray(y_val, dtype=int).ravel()
        self._prevalence = float(np.clip(y_tr.mean(), 1e-5, 0.99))

        # 1. Fit Scout on X_base
        self.scout_arm_ = fit_single_calibrated_gbdt(
            X_base_train,
            y_tr,
            X_base_cal,
            y_c,
            semantic_types=base_semantic_types,
            sample_weight=sample_weight_train,
            arm_name="lattice_scout",
            seed=self.seed,
        )
        scout_tr_p = self.scout_arm_.predict_proba(X_base_train)
        scout_val_p = self.scout_arm_.predict_proba(X_base_val)

        # Identify training false-negative misses of the scout to upweight Rescue training
        pos_med = float(np.median(scout_tr_p[y_tr == 1])) if np.any(y_tr == 1) else 0.2
        fn_miss_tr = (y_tr == 1) & (scout_tr_p <= pos_med)
        rescue_weights = (
            sample_weight_train.copy()
            if sample_weight_train is not None
            else np.ones(y_tr.size, dtype=float)
        )
        rescue_weights[fn_miss_tr] *= 2.5

        self.branches_ = []
        n_ch = max(1, len(Z_channels_train))

        for idx, (ch_id, Z_tr) in enumerate(Z_channels_train.items()):
            Z_c = Z_channels_cal[ch_id]
            Z_v = Z_channels_val[ch_id]

            # Fit Confirmation arm strictly on Z_m alone for orthogonal FP errors!
            prep_b = TabularPreprocessor().fit(Z_tr)
            Zb_tr = prep_b.transform(Z_tr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if Zb_tr.shape[1] <= 3:
                    clf_b: object = LogisticRegression(
                        C=1.0, max_iter=600, random_state=self.seed + idx
                    )
                    clf_b.fit(Zb_tr, y_tr, sample_weight=sample_weight_train)  # type: ignore[attr-defined]
                else:
                    clf_b = LGBMClassifier(
                        n_estimators=120,
                        learning_rate=0.05,
                        num_leaves=23,
                        min_child_samples=15,
                        random_state=self.seed + idx,
                        n_jobs=1,
                        verbose=-1,
                    )
                    clf_b.fit(Zb_tr, y_tr, sample_weight=sample_weight_train)  # type: ignore[attr-defined]
            arm_b = FittedDecoderArm(f"confirm_{ch_id}", prep_b, [clf_b], None)
            arm_b.calibrator = PlattLogitCalibrator().fit(arm_b.predict_raw_proba(Z_c), y_c)

            # Audit negative-tail coincidence kappa_0(A, B_m) on validation negatives
            b_val_p = arm_b.predict_proba(Z_v)
            neg_v = y_v == 0
            coinc = estimate_tail_coincidence_surrogate(
                scout_val_p[neg_v],
                b_val_p[neg_v],
                alpha_a=min(0.10, self.max_fpr_budget * 5.0),
                alpha_b=0.10,
                n_candidate_pairs_tested=n_ch,
                seed=self.seed + idx,
            )
            # Channels with lower negative-tail coincidence get higher confirmation weight!
            orth_w = float(
                1.0 / max(0.35, np.sqrt(max(0.1, coinc.negative_tail_coincidence_kappa0)))
            )

            # Fit Rescue arm on [X_base, Z_m] with FN-upweighted training
            XZ_tr = pd.concat(
                [X_base_train.reset_index(drop=True), Z_tr.reset_index(drop=True)], axis=1
            )
            XZ_c = pd.concat(
                [X_base_cal.reset_index(drop=True), Z_c.reset_index(drop=True)], axis=1
            )
            arm_c = fit_single_calibrated_gbdt(
                XZ_tr,
                y_tr,
                XZ_c,
                y_c,
                sample_weight=rescue_weights,
                arm_name=f"rescue_{ch_id}",
                seed=self.seed + 100 + idx,
            )

            self.branches_.append(
                MechanismBranch(
                    channel_id=ch_id,
                    confirm_arm=arm_b,
                    rescue_arm=arm_c,
                    coincidence_report=coinc,
                    orthogonality_weight=orth_w,
                )
            )

        # 2. Optimize Boolean OR-of-ANDs thresholds on Validation fold under FPR <= max_fpr_budget
        self._optimize_boolean_thresholds_on_val(X_base_val, Z_channels_val, y_v)

        # 3. Fit calibrated continuous lattice log-odds combiner on Cal + Val features
        # (Strictly leak-free with respect to untouched Test!)
        feat_cal = self._lattice_log_odds_features(X_base_cal, Z_channels_cal)
        feat_val = self._lattice_log_odds_features(X_base_val, Z_channels_val)
        F_cv = np.vstack([feat_cal, feat_val])
        y_cv = np.concatenate([y_c, y_v])
        self._meta_calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=self.seed)
        self._meta_calibrator.fit(F_cv, y_cv)

        self._fitted = True
        return self

    def _optimize_boolean_thresholds_on_val(
        self,
        X_base_val: pd.DataFrame,
        Z_channels_val: dict[str, pd.DataFrame],
        y_val: np.ndarray,
    ) -> None:
        assert self.scout_arm_ is not None
        if not self.branches_:
            return

        scout_p = self.scout_arm_.predict_proba(X_base_val)
        neg_mask = y_val == 0
        pos_mask = y_val == 1
        n_pos = max(1, int(np.sum(pos_mask)))
        n_neg = max(1, int(np.sum(neg_mask)))

        b_probs = [
            br.confirm_arm.predict_proba(Z_channels_val[br.channel_id]) for br in self.branches_
        ]
        c_probs = [
            br.rescue_arm.predict_proba(
                pd.concat(
                    [
                        X_base_val.reset_index(drop=True),
                        Z_channels_val[br.channel_id].reset_index(drop=True),
                    ],
                    axis=1,
                )
            )
            for br in self.branches_
        ]

        best_tpr = -1.0
        best_params: tuple[float, list[float], list[float]] | None = None

        m_br = len(self.branches_)
        # Search split of FPR budget between Confirm (AND) and Rescue (OR) + scout loosening factor
        for rescue_share in (0.0, 0.20, 0.40, 0.60):
            confirm_budget = self.max_fpr_budget * (1.0 - rescue_share)
            rescue_budget_each = (self.max_fpr_budget * rescue_share) / m_br

            for scout_mult in (2.5, 4.0, 7.0, 12.0):
                alpha_scout = min(0.35, max(self.max_fpr_budget, confirm_budget * scout_mult))
                tau_a = select_neyman_pearson_threshold(y_val, scout_p, alpha_scout)
                a_hit = scout_p >= tau_a

                # Conditional FPR budget for each B_m inside {A >= tau_a, Y=0}
                tau_bs: list[float] = []
                confirm_or = np.zeros(y_val.size, dtype=bool)
                a_neg_idx = np.flatnonzero(a_hit & neg_mask)

                for bp in b_probs:
                    target_cond_fpr = min(
                        0.80,
                        (confirm_budget / m_br) / max(1e-5, float(np.mean(a_hit[neg_mask]))),
                    )
                    if a_neg_idx.size >= 5:
                        tb = float(np.quantile(bp[a_neg_idx], max(0.0, 1.0 - target_cond_fpr)))
                    else:
                        tb = select_neyman_pearson_threshold(y_val, bp, confirm_budget / m_br)
                    tau_bs.append(tb)
                    confirm_or |= a_hit & (bp >= tb)

                tau_cs: list[float] = []
                rescue_or = np.zeros(y_val.size, dtype=bool)
                for cp in c_probs:
                    if rescue_budget_each <= 1e-7:
                        tc = float(np.max(cp) + 1.0)
                    else:
                        tc = select_neyman_pearson_threshold(y_val, cp, rescue_budget_each)
                    tau_cs.append(tc)
                    rescue_or |= cp >= tc

                final_alert = confirm_or | rescue_or
                realized_fpr = float(np.sum(final_alert & neg_mask) / n_neg)
                if realized_fpr <= self.max_fpr_budget + (0.5 / n_neg):
                    tpr = float(np.sum(final_alert & pos_mask) / n_pos)
                    if tpr > best_tpr:
                        best_tpr = tpr
                        best_params = (tau_a, tau_bs, tau_cs)

        if best_params is not None:
            tau_a, tau_bs, tau_cs = best_params
            for m_idx, br in enumerate(self.branches_):
                br.tau_scout = tau_a
                br.tau_confirm = tau_bs[m_idx]
                br.tau_rescue = tau_cs[m_idx]

    def _lattice_log_odds_features(
        self,
        X_base: pd.DataFrame,
        Z_channels: dict[str, pd.DataFrame],
    ) -> np.ndarray:
        assert self.scout_arm_ is not None
        pi_logit = float(logit(self._prevalence))
        p_scout = np.clip(self.scout_arm_.predict_proba(X_base), 1e-6, 1.0 - 1e-6)
        llr_scout = logit(p_scout) - pi_logit

        cols: list[np.ndarray] = [llr_scout]
        if self.branches_:
            confirm_sum = np.zeros(len(X_base), dtype=float)
            rescue_max = np.full(len(X_base), -10.0, dtype=float)
            and_synergy = np.zeros(len(X_base), dtype=float)

            for br in self.branches_:
                Zc = Z_channels[br.channel_id]
                pb = np.clip(br.confirm_arm.predict_proba(Zc), 1e-6, 1.0 - 1e-6)
                llr_b = logit(pb) - pi_logit
                confirm_sum += br.orthogonality_weight * llr_b
                # Conjunctive confirmation evidence triggers when BOTH channels are positive!
                and_synergy += (
                    br.orthogonality_weight * np.maximum(0.0, llr_scout) * np.maximum(0.0, llr_b)
                )

                XZ = pd.concat([X_base.reset_index(drop=True), Zc.reset_index(drop=True)], axis=1)
                pc = np.clip(br.rescue_arm.predict_proba(XZ), 1e-6, 1.0 - 1e-6)
                llr_c = logit(pc) - pi_logit
                rescue_max = np.maximum(rescue_max, llr_c)

            cols.extend([confirm_sum, and_synergy, rescue_max])

        return np.column_stack(cols)

    def predict_boolean_lattice(
        self,
        X_base: pd.DataFrame,
        Z_channels: dict[str, pd.DataFrame],
    ) -> np.ndarray:
        """Evaluate exact Boolean OR-of-ANDs decision rule at the certified FPR budget."""
        if not self._fitted or self.scout_arm_ is None:
            raise RuntimeError("ConfirmRescueLattice must be fit before prediction.")
        scout_p = self.scout_arm_.predict_proba(X_base)
        if not self.branches_:
            return (scout_p >= 0.5).astype(int)

        alert = np.zeros(len(X_base), dtype=bool)
        for br in self.branches_:
            Zc = Z_channels[br.channel_id]
            bp = br.confirm_arm.predict_proba(Zc)
            XZ = pd.concat([X_base.reset_index(drop=True), Zc.reset_index(drop=True)], axis=1)
            cp = br.rescue_arm.predict_proba(XZ)
            confirm_hit = (scout_p >= br.tau_scout) & (bp >= br.tau_confirm)
            rescue_hit = cp >= br.tau_rescue
            alert |= confirm_hit | rescue_hit
        return alert.astype(int)

    def predict_proba(
        self,
        X_base: pd.DataFrame,
        Z_channels: dict[str, pd.DataFrame],
    ) -> np.ndarray:
        """Predict calibrated posterior probability over the Confirm + Rescue lattice."""
        if not self._fitted or self.scout_arm_ is None:
            raise RuntimeError("ConfirmRescueLattice must be fit before prediction.")
        if not self.branches_ or self._meta_calibrator is None:
            return self.scout_arm_.predict_proba(X_base)
        F = self._lattice_log_odds_features(X_base, Z_channels)
        p = self._meta_calibrator.predict_proba(F)[:, 1]
        return np.asarray(expit(logit(np.clip(p, 1e-6, 1.0 - 1e-6))), dtype=float)
