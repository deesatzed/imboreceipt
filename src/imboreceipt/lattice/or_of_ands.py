"""Stage O (Part 2): Asymmetric Confirm-and-Rescue Lattice with Adaptive kappa_0 Routing."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from imboreceipt.core.types import SemanticType
from imboreceipt.discover.strata import make_leak_free_cv_splits
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

_LOGIT_EPS = 1e-15


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


def _oof_platt_probs(
    raw_probs: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    time_order: np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Cross-fit Platt calibration on calibration fold respecting groups/time order."""
    yv = np.asarray(y, dtype=int).ravel()
    rp = np.asarray(raw_probs, dtype=float).ravel()
    n_pos = int(yv.sum())
    n_neg = yv.size - n_pos
    if n_pos < 6 or n_neg < 6:
        return PlattLogitCalibrator().fit(rp, yv).predict_proba(rp)

    k = min(3, n_pos, n_neg)
    splits = make_leak_free_cv_splits(
        yv, n_folds=k, groups=groups, time_order=time_order, seed=seed
    )
    oof = np.full_like(rp, float(yv.mean()))
    for tr, va in splits:
        cal = PlattLogitCalibrator().fit(rp[tr], yv[tr])
        oof[va] = cal.predict_proba(rp[va])
    return oof


class ConfirmRescueLattice:
    """Asymmetric OR-of-ANDs Evidence Lattice with adaptive kappa_0-gated coupled fusion.

    Supports two architectural modes for controlled knockout ablation:
    1. ``mode="pure_orthogonal"``: Uses strictly separate-channel Confirmation ``B_m(Z_m)``
       (``A_scout AND B_m``) + per-channel FN-upweighted Rescue ``C_m([X_base, Z_m])``.
    2. ``mode="hybrid_kappa0"`` (default Full CREDO): Uses the LIGO negative-tail coincidence
       diagnostic ``kappa_0(A_scout, B_m)`` measured strictly on calibration negatives to
       determine whether acquired channels act as independent confirmers (``kappa_0 <= 1.8``)
       or coupled facets of a shared mechanism (``kappa_0 > 1.8``). Blends the orthogonal
       Confirm-Rescue lattice log-odds with a Joint Coupled Fusion arm ``J([X_base, Z_1..M])``
       via cross-fitted validation selection anchored on a ``kappa_0`` prior, leaving
       out-of-fold validation scores leak-free for threshold certification.
    """

    def __init__(
        self,
        *,
        max_fpr_budget: float = 0.01,
        mode: Literal["hybrid_kappa0", "pure_orthogonal"] = "hybrid_kappa0",
        kappa0_pivot: float = 1.80,
        seed: int = 42,
    ) -> None:
        self.max_fpr_budget = max_fpr_budget
        self.mode = mode
        self.kappa0_pivot = kappa0_pivot
        self.seed = seed
        self.scout_arm_: FittedDecoderArm | None = None
        self.branches_: list[MechanismBranch] = []
        self.coupled_arm_: FittedDecoderArm | None = None
        self._orth_meta_lr: LogisticRegression | None = None
        self._hybrid_calibrator: PlattLogitCalibrator | None = None
        self.mean_kappa0_: float = 1.0
        self.orthogonal_lattice_weight_: float = 1.0
        self._prevalence: float = 0.05
        self._fitted = False

    @staticmethod
    def _concat_all(X_base: pd.DataFrame, Z_channels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        parts = [X_base.reset_index(drop=True)]
        for ch_id in sorted(Z_channels.keys()):
            parts.append(Z_channels[ch_id].reset_index(drop=True))
        return pd.concat(parts, axis=1)

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
        groups_cal: np.ndarray | None = None,
        time_order_cal: np.ndarray | None = None,
    ) -> ConfirmRescueLattice:
        """Fit Scout, Confirmation, Rescue, and Coupled arms; route via kappa_0 without leakage."""
        y_tr = np.asarray(y_train, dtype=int).ravel()
        y_c = np.asarray(y_cal, dtype=int).ravel()
        y_v = np.asarray(y_val, dtype=int).ravel()
        self._prevalence = float(np.clip(y_c.mean() if y_c.sum() > 0 else y_tr.mean(), 1e-5, 0.99))

        # 1. Fit Scout on X_base_train
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
        scout_cal_oof = _oof_platt_probs(
            self.scout_arm_.predict_raw_proba(X_base_cal),
            y_c,
            groups=groups_cal,
            time_order=time_order_cal,
            seed=self.seed,
        )

        # Upweight Scout false negatives on training fold for Rescue learners
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
        cal_b_oof_list: list[np.ndarray] = []
        cal_c_oof_list: list[np.ndarray] = []

        for idx, (ch_id, Z_tr) in enumerate(Z_channels_train.items()):
            Z_c = Z_channels_cal[ch_id]

            # Fit Confirmation arm strictly on Z_m alone
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
                        max_delta_step=1.0,
                        random_state=self.seed + idx,
                        n_jobs=4,
                        verbose=-1,
                    )
                    clf_b.fit(Zb_tr, y_tr, sample_weight=sample_weight_train)  # type: ignore[attr-defined]
            arm_b = FittedDecoderArm(f"confirm_{ch_id}", prep_b, [clf_b], None)
            raw_b_cal = arm_b.predict_raw_proba(Z_c)
            b_cal_oof = _oof_platt_probs(
                raw_b_cal,
                y_c,
                groups=groups_cal,
                time_order=time_order_cal,
                seed=self.seed + idx,
            )
            arm_b.calibrator = PlattLogitCalibrator().fit(raw_b_cal, y_c)
            cal_b_oof_list.append(b_cal_oof)

            # Audit negative-tail coincidence kappa_0(A, B_m) strictly on CALIBRATION negatives
            neg_c = y_c == 0
            coinc = estimate_tail_coincidence_surrogate(
                scout_cal_oof[neg_c],
                b_cal_oof[neg_c],
                alpha_a=min(0.10, self.max_fpr_budget * 5.0),
                alpha_b=0.10,
                n_candidate_pairs_tested=n_ch,
                seed=self.seed + idx,
            )
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
            c_cal_oof = _oof_platt_probs(
                arm_c.predict_raw_proba(XZ_c),
                y_c,
                groups=groups_cal,
                time_order=time_order_cal,
                seed=self.seed + 100 + idx,
            )
            cal_c_oof_list.append(c_cal_oof)

            self.branches_.append(
                MechanismBranch(
                    channel_id=ch_id,
                    confirm_arm=arm_b,
                    rescue_arm=arm_c,
                    coincidence_report=coinc,
                    orthogonality_weight=orth_w,
                )
            )

        if not self.branches_:
            self._fitted = True
            return self

        self.mean_kappa0_ = float(
            np.mean(
                [br.coincidence_report.negative_tail_coincidence_kappa0 for br in self.branches_]
            )
        )

        # 2. Fit Orthogonal Confirm-Rescue Meta-Combiner strictly on cross-fitted CAL features
        F_cal_oof = self._assemble_orth_features_from_probs(
            scout_cal_oof, cal_b_oof_list, cal_c_oof_list
        )
        self._orth_meta_lr = LogisticRegression(C=0.8, max_iter=500, random_state=self.seed)
        if int(y_c.sum()) >= 4 and int((1 - y_c).sum()) >= 4:
            self._orth_meta_lr.fit(F_cal_oof, y_c)
        else:
            self._orth_meta_lr = None

        # 3. Optimize Boolean OR-of-ANDs thresholds on untouched Validation fold
        self._optimize_boolean_thresholds_on_val(X_base_val, Z_channels_val, y_v)

        # 4. Adaptive kappa_0-Gated Coupled Fusion (when mode == "hybrid_kappa0")
        if self.mode == "pure_orthogonal":
            self.orthogonal_lattice_weight_ = 1.0
            self.coupled_arm_ = None
            self._hybrid_calibrator = None
        else:
            all_tr = self._concat_all(X_base_train, Z_channels_train)
            all_cal = self._concat_all(X_base_cal, Z_channels_cal)

            self.coupled_arm_ = fit_single_calibrated_gbdt(
                all_tr,
                y_tr,
                all_cal,
                y_c,
                sample_weight=sample_weight_train,
                arm_name="coupled_joint_fusion",
                seed=self.seed + 300,
            )

            # Cross-fitted OOF predictions on CAL (leaving VAL 100% untouched for thresholds!)
            p_coup_cal_oof = _oof_platt_probs(
                self.coupled_arm_.predict_raw_proba(all_cal),
                y_c,
                groups=groups_cal,
                time_order=time_order_cal,
                seed=self.seed + 300,
            )
            if int(y_c.sum()) >= 6 and int((1 - y_c).sum()) >= 6:
                cal_splits = make_leak_free_cv_splits(
                    y_c,
                    n_folds=3,
                    groups=groups_cal,
                    time_order=time_order_cal,
                    seed=self.seed,
                )
                p_orth_cal_oof = np.full(y_c.size, float(y_c.mean()), dtype=float)
                for tr_k, va_k in cal_splits:
                    lr_k = LogisticRegression(C=0.8, max_iter=500, random_state=self.seed)
                    lr_k.fit(F_cal_oof[tr_k], y_c[tr_k])
                    p_orth_cal_oof[va_k] = lr_k.predict_proba(F_cal_oof[va_k])[:, 1]
            else:
                p_orth_cal_oof = self._predict_orth_proba(X_base_cal, Z_channels_cal)

            llr_orth_c = logit(np.clip(p_orth_cal_oof, _LOGIT_EPS, 1.0 - _LOGIT_EPS))
            llr_coup_c = logit(np.clip(p_coup_cal_oof, _LOGIT_EPS, 1.0 - _LOGIT_EPS))

            # Prior weight on orth lattice: ~0.85 when kappa_0 <= 1.1, ~0.15 when kappa_0 >= 2.5
            prior_w_orth = float(
                np.clip(1.0 - (self.mean_kappa0_ - 1.05) / max(0.3, 2.50 - 1.05), 0.10, 0.90)
            )

            best_obj = -1e9
            best_w = prior_w_orth
            if int(y_c.sum()) >= 6 and int((1 - y_c).sum()) >= 20:
                # Empirical-Bayes shrinkage scales inversely with calibration positives n_+^cal
                # preventing meta-selection overfitting on small calibration folds!
                shrink_strength = max(0.03, 3.0 / max(6.0, float(y_c.sum())))
                for cand_w in (0.0, 0.20, 0.40, 0.60, 0.80, 1.0):
                    blend_p = expit(cand_w * llr_orth_c + (1.0 - cand_w) * llr_coup_c)
                    ap = float(average_precision_score(y_c, blend_p))
                    tau_np = select_neyman_pearson_threshold(y_c, blend_p, self.max_fpr_budget)
                    np_tpr = float(np.mean(blend_p[y_c == 1] >= tau_np))
                    prior_pen = shrink_strength * ((cand_w - prior_w_orth) ** 2)
                    obj = 0.55 * ap + 0.45 * np_tpr - prior_pen
                    if obj > best_obj:
                        best_obj = obj
                        best_w = cand_w

            self.orthogonal_lattice_weight_ = best_w
            # Do NOT refit a 3rd post-selection Platt calibrator on the same y_c fold:
            # convex log-odds blend of two calibrated models stays calibrated!
            self._hybrid_calibrator = None

        self._fitted = True
        return self

    def _assemble_orth_features_from_probs(
        self,
        p_scout: np.ndarray,
        b_prob_list: list[np.ndarray],
        c_prob_list: list[np.ndarray],
    ) -> np.ndarray:
        pi_logit = float(logit(self._prevalence))
        ps = np.clip(p_scout, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
        llr_scout = logit(ps) - pi_logit

        confirm_sum = np.zeros(ps.size, dtype=float)
        rescue_max = np.full(ps.size, -30.0, dtype=float)
        and_synergy = np.zeros(ps.size, dtype=float)

        for br, pb_raw, pc_raw in zip(self.branches_, b_prob_list, c_prob_list, strict=True):
            pb = np.clip(pb_raw, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
            llr_b = logit(pb) - pi_logit
            confirm_sum += br.orthogonality_weight * llr_b
            and_synergy += (
                br.orthogonality_weight * np.maximum(0.0, llr_scout) * np.maximum(0.0, llr_b)
            )
            pc = np.clip(pc_raw, _LOGIT_EPS, 1.0 - _LOGIT_EPS)
            llr_c = logit(pc) - pi_logit
            rescue_max = np.maximum(rescue_max, llr_c)

        return np.column_stack([llr_scout, confirm_sum, and_synergy, rescue_max])

    def _predict_orth_proba(
        self,
        X_base: pd.DataFrame,
        Z_channels: dict[str, pd.DataFrame],
    ) -> np.ndarray:
        assert self.scout_arm_ is not None
        p_scout = self.scout_arm_.predict_proba(X_base)
        if not self.branches_ or self._orth_meta_lr is None:
            return p_scout
        b_list = [br.confirm_arm.predict_proba(Z_channels[br.channel_id]) for br in self.branches_]
        c_list = [
            br.rescue_arm.predict_proba(
                pd.concat(
                    [
                        X_base.reset_index(drop=True),
                        Z_channels[br.channel_id].reset_index(drop=True),
                    ],
                    axis=1,
                )
            )
            for br in self.branches_
        ]
        F = self._assemble_orth_features_from_probs(p_scout, b_list, c_list)
        return np.asarray(self._orth_meta_lr.predict_proba(F)[:, 1], dtype=float)

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
                        X_base.reset_index(drop=True)
                        for X_base in (X_base_val, Z_channels_val[br.channel_id])
                    ],
                    axis=1,
                )
            )
            for br in self.branches_
        ]

        best_tpr = -1.0
        best_params: tuple[float, list[float], list[float]] | None = None
        m_br = len(self.branches_)

        for rescue_share in (0.0, 0.20, 0.40, 0.60, 0.85):
            confirm_budget = self.max_fpr_budget * (1.0 - rescue_share)
            rescue_budget_each = (self.max_fpr_budget * rescue_share) / m_br

            for scout_mult in (2.5, 4.0, 7.0, 12.0):
                alpha_scout = min(0.35, max(self.max_fpr_budget, confirm_budget * scout_mult))
                tau_a = select_neyman_pearson_threshold(y_val, scout_p, alpha_scout)
                a_hit = scout_p >= tau_a

                tau_bs: list[float] = []
                confirm_or = np.zeros(y_val.size, dtype=bool)
                a_neg_idx = np.flatnonzero(a_hit & neg_mask)

                for bp in b_probs:
                    if confirm_budget <= 1e-7:
                        tb = float(np.max(bp) + 1.0)
                    else:
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

    def predict_boolean_lattice(
        self,
        X_base: pd.DataFrame,
        Z_channels: dict[str, pd.DataFrame],
    ) -> np.ndarray:
        """Evaluate exact Boolean OR-of-ANDs decision rule at the certified FPR budget."""
        if not self._fitted or self.scout_arm_ is not None and not self.branches_:
            if not self._fitted or self.scout_arm_ is None:
                raise RuntimeError("ConfirmRescueLattice must be fit before prediction.")
            return (self.scout_arm_.predict_proba(X_base) >= 0.5).astype(int)

        assert self.scout_arm_ is not None
        scout_p = self.scout_arm_.predict_proba(X_base)
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
        """Predict calibrated posterior probability over the Confirm + Rescue (+ Coupled) net."""
        if not self._fitted or self.scout_arm_ is None:
            raise RuntimeError("ConfirmRescueLattice must be fit before prediction.")
        if not self.branches_:
            return self.scout_arm_.predict_proba(X_base)

        p_orth = self._predict_orth_proba(X_base, Z_channels)
        if self.mode == "pure_orthogonal" or self.coupled_arm_ is None:
            return p_orth

        w = self.orthogonal_lattice_weight_
        if w >= 0.999:
            raw_blend = p_orth
        else:
            p_coup = self.coupled_arm_.predict_proba(self._concat_all(X_base, Z_channels))
            raw_blend = expit(
                w * logit(np.clip(p_orth, _LOGIT_EPS, 1.0 - _LOGIT_EPS))
                + (1.0 - w) * logit(np.clip(p_coup, _LOGIT_EPS, 1.0 - _LOGIT_EPS))
            )

        if self._hybrid_calibrator is not None:
            return self._hybrid_calibrator.predict_proba(raw_blend)
        return np.asarray(raw_blend, dtype=float)
