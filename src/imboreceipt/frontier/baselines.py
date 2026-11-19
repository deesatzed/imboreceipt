"""Stage E (Part 1): Leak-free preprocessing, calibrated baselines, and decoder-trick suite."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from lightgbm import LGBMClassifier
from scipy.special import logit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from imboreceipt.core.types import SemanticType, infer_semantic_types


class TabularPreprocessor:
    """Strict train-only fitted preprocessor adhering to ml-best-practices.

    - Tracks missingness rates on training data and emits explicit ``miss__<col>`` indicators
      whenever missingness is non-trivial (>0.5% and <99.5%).
    - Applies median imputation + standardization to ``CONTINUOUS`` and ``COUNT`` columns.
    - Applies unknown-safe ``OrdinalEncoder`` to ``ORDINAL`` columns.
    - Applies bounded-cardinality ``OneHotEncoder`` to ``NOMINAL`` columns.
    - Passes ``BINARY`` and ``RECEIPT_FLAG`` columns with zero/mode imputation.
    """

    def __init__(
        self,
        semantic_types: dict[str, SemanticType] | None = None,
        max_ohe_categories: int = 25,
    ) -> None:
        self.declared_types = semantic_types
        self.max_ohe_categories = max_ohe_categories
        self.semantic_types_: dict[str, SemanticType] = {}
        self.miss_cols_: list[str] = []
        self.num_cols_: list[str] = []
        self.ord_cols_: list[str] = []
        self.nom_cols_: list[str] = []
        self.bin_cols_: list[str] = []
        self._num_imp: SimpleImputer | None = None
        self._num_scl: StandardScaler | None = None
        self._ord_enc: OrdinalEncoder | None = None
        self._nom_enc: OneHotEncoder | None = None
        self._bin_imp: SimpleImputer | None = None
        self.feature_names_out_: list[str] = []
        self._fitted = False

    def fit(self, X_train: pd.DataFrame) -> TabularPreprocessor:
        if any(type(c) is not str for c in X_train.columns):
            X_train = X_train.rename(columns=str)
        self.semantic_types_ = infer_semantic_types(X_train, overrides=self.declared_types)
        self.miss_cols_ = []
        self.num_cols_ = []
        self.ord_cols_ = []
        self.nom_cols_ = []
        self.bin_cols_ = []

        for col in X_train.columns:
            stype = self.semantic_types_.get(col)
            if stype is None or stype in {SemanticType.ENTITY_ID, SemanticType.TIMESTAMP}:
                continue
            miss_rate = float(X_train[col].isna().mean())
            if 0.005 < miss_rate < 0.995:
                self.miss_cols_.append(col)

            if stype in {SemanticType.CONTINUOUS, SemanticType.COUNT}:
                self.num_cols_.append(col)
            elif stype == SemanticType.ORDINAL:
                self.ord_cols_.append(col)
            elif stype == SemanticType.NOMINAL:
                self.nom_cols_.append(col)
            elif stype in {SemanticType.BINARY, SemanticType.RECEIPT_FLAG}:
                self.bin_cols_.append(col)

        names: list[str] = []

        if self.num_cols_:
            num_df = X_train[self.num_cols_].apply(pd.to_numeric, errors="coerce")
            # Drop completely all-NaN numeric columns safely
            valid_num = [c for c in self.num_cols_ if num_df[c].notna().any()]
            self.num_cols_ = valid_num
            if self.num_cols_:
                self._num_imp = SimpleImputer(strategy="median")
                self._num_scl = StandardScaler()
                arr = self._num_imp.fit_transform(num_df[self.num_cols_])
                self._num_scl.fit(arr)
                names.extend(self.num_cols_)

        if self.bin_cols_:
            bin_df = X_train[self.bin_cols_].apply(pd.to_numeric, errors="coerce")
            valid_bin = [c for c in self.bin_cols_ if bin_df[c].notna().any()]
            self.bin_cols_ = valid_bin
            if self.bin_cols_:
                self._bin_imp = SimpleImputer(strategy="most_frequent")
                self._bin_imp.fit(bin_df[self.bin_cols_])
                names.extend(self.bin_cols_)

        if self.ord_cols_:
            ord_str = X_train[self.ord_cols_].fillna("__MISSING__").astype(str)
            self._ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self._ord_enc.fit(ord_str)
            names.extend(self.ord_cols_)

        if self.nom_cols_:
            nom_str = X_train[self.nom_cols_].fillna("__MISSING__").astype(str)
            self._nom_enc = OneHotEncoder(
                handle_unknown="ignore",
                max_categories=self.max_ohe_categories,
                sparse_output=False,
            )
            self._nom_enc.fit(nom_str)
            names.extend(self._nom_enc.get_feature_names_out(self.nom_cols_).tolist())

        names.extend([f"miss__{c}" for c in self.miss_cols_])
        self.feature_names_out_ = names
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform rows WITHOUT label access."""
        if not self._fitted:
            raise RuntimeError("TabularPreprocessor must be fit on X_train before transform.")
        if any(type(c) is not str for c in X.columns):
            X = X.rename(columns=str)

        blocks: list[np.ndarray] = []
        if self.num_cols_ and self._num_imp is not None and self._num_scl is not None:
            num_df = X[self.num_cols_].apply(pd.to_numeric, errors="coerce")
            blocks.append(self._num_scl.transform(self._num_imp.transform(num_df)))

        if self.bin_cols_ and self._bin_imp is not None:
            bin_df = X[self.bin_cols_].apply(pd.to_numeric, errors="coerce")
            blocks.append(self._bin_imp.transform(bin_df))

        if self.ord_cols_ and self._ord_enc is not None:
            ord_str = X[self.ord_cols_].fillna("__MISSING__").astype(str)
            blocks.append(self._ord_enc.transform(ord_str))

        if self.nom_cols_ and self._nom_enc is not None:
            nom_str = X[self.nom_cols_].fillna("__MISSING__").astype(str)
            blocks.append(self._nom_enc.transform(nom_str))

        if self.miss_cols_:
            miss_arr = np.column_stack([X[c].isna().to_numpy(dtype=float) for c in self.miss_cols_])
            blocks.append(miss_arr)

        if not blocks:
            return np.zeros((len(X), 1), dtype=float)
        return np.hstack(blocks).astype(np.float64, copy=False)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit strictly on ``X_train`` and transform ``X_train``."""
        return self.fit(X_train).transform(X_train)


_PLATT_EPS = 1e-15


class PlattLogitCalibrator:
    """Monotonic logit-scale Platt calibrator fit on natural-prevalence calibration fold."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=10.0, max_iter=500)
        self._fallback_prev = 0.05
        self._active = False

    def fit(self, raw_probs: np.ndarray, y_cal: np.ndarray) -> PlattLogitCalibrator:
        y = np.asarray(y_cal, dtype=int).ravel()
        self._fallback_prev = float(np.clip(y.mean(), 1e-5, 1.0 - 1e-5))
        if int(y.sum()) < 5 or int((1 - y).sum()) < 5:
            self._active = False
            return self
        z = logit(
            np.clip(np.asarray(raw_probs, dtype=float).ravel(), _PLATT_EPS, 1.0 - _PLATT_EPS)
        ).reshape(-1, 1)
        self._lr.fit(z, y)
        # Enforce monotone non-decreasing calibration slope
        if float(self._lr.coef_[0, 0]) <= 0.0:
            self._active = False
        else:
            self._active = True
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw_probs, dtype=float).ravel(), _PLATT_EPS, 1.0 - _PLATT_EPS)
        if not self._active:
            return p
        z = logit(p).reshape(-1, 1)
        return np.asarray(self._lr.predict_proba(z)[:, 1], dtype=float)


def _gaussian_copula_minority_augment(
    X_mat: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic minority samples via rank-Gaussian copula (latent-space baseline)."""
    pos = X_mat[y == 1]
    n_pos = pos.shape[0]
    n_neg = int(np.sum(y == 0))
    n_target_pos = int(round(n_neg * target_ratio))
    n_syn = max(0, n_target_pos - n_pos)
    if n_syn == 0 or n_pos < 6:
        return X_mat, y

    rng = np.random.default_rng(seed)
    d = pos.shape[1]
    # Empirical CDF -> normal scores per feature
    ranks = np.argsort(np.argsort(pos, axis=0), axis=0) + 1.0
    u = ranks / (n_pos + 1.0)
    z = norm.ppf(u)
    cov = np.cov(z, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + 0.05 * np.eye(d)

    z_syn = rng.multivariate_normal(np.zeros(d), cov, size=n_syn)
    u_syn = norm.cdf(z_syn)
    x_syn = np.empty((n_syn, d), dtype=float)
    for j in range(d):
        sorted_col = np.sort(pos[:, j])
        q_idx = np.clip(u_syn[:, j] * (n_pos - 1), 0.0, n_pos - 1.0)
        lo = np.floor(q_idx).astype(int)
        hi = np.ceil(q_idx).astype(int)
        frac = q_idx - lo
        x_syn[:, j] = sorted_col[lo] * (1.0 - frac) + sorted_col[hi] * frac

    X_aug = np.vstack([X_mat, x_syn])
    y_aug = np.concatenate([y, np.ones(n_syn, dtype=int)])
    return X_aug, y_aug


@dataclass(slots=True)
class FittedDecoderArm:
    """One trained + calibrated model arm for frontier evaluation."""

    arm_name: str
    preprocessor: TabularPreprocessor
    estimators: list[object]
    calibrator: PlattLogitCalibrator | None

    def predict_raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xm = self.preprocessor.transform(X)
        preds = [est.predict_proba(Xm)[:, 1] for est in self.estimators]  # type: ignore[attr-defined]
        return np.mean(np.column_stack(preds), axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw_proba(X)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(raw)


def _make_lgbm(seed: int = 42, scale_pos_weight: float = 1.0, n_jobs: int = 4) -> LGBMClassifier:
    """Create a Newton-stabilized GBDT (max_delta_step=1.0 prevents Hessian blowup)."""
    return LGBMClassifier(
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=15,
        max_delta_step=1.0,
        subsample_for_bin=50000,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )


def fit_single_calibrated_gbdt(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    *,
    semantic_types: dict[str, SemanticType] | None = None,
    sample_weight: np.ndarray | None = None,
    arm_name: str = "calibrated_gbdt",
    seed: int = 42,
) -> FittedDecoderArm:
    """Fit the canonical CREDO natural-prevalence calibrated GBDT baseline."""
    prep = TabularPreprocessor(semantic_types=semantic_types).fit(X_train)
    X_tr = prep.transform(X_train)
    clf = _make_lgbm(seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, np.asarray(y_train, dtype=int), sample_weight=sample_weight)
    arm = FittedDecoderArm(arm_name=arm_name, preprocessor=prep, estimators=[clf], calibrator=None)
    cal = PlattLogitCalibrator().fit(arm.predict_raw_proba(X_cal), y_cal)
    arm.calibrator = cal
    return arm


def fit_equal_capacity_monolithic_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    *,
    semantic_types: dict[str, SemanticType] | None = None,
    sample_weight: np.ndarray | None = None,
    arm_name: str = "monolithic_credo_ensemble",
    seed: int = 42,
) -> FittedDecoderArm:
    """Fit a 6-learner monolithic ensemble matching ConfirmRescueLattice's exact learner count.

    Controls against model-count / bagging variance-reduction confounding in ablations:
    trains 4 diverse seeded GBDTs + 1 FN-upweighted GBDT + 1 Penalized GLM on the joint
    table ``[X_base, Z]`` and calibrates on ``X_cal``.
    """
    y_tr = np.asarray(y_train, dtype=int).ravel()
    prep = TabularPreprocessor(semantic_types=semantic_types).fit(X_train)
    X_tr = prep.transform(X_train)

    learners: list[object] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 4 diverse GBDTs
        for idx, (n_l, col_f) in enumerate(((31, 1.0), (23, 0.8), (39, 0.85), (19, 0.9))):
            gb = LGBMClassifier(
                n_estimators=180,
                learning_rate=0.05,
                num_leaves=n_l,
                colsample_bytree=col_f,
                subsample=0.85,
                subsample_freq=1,
                min_child_samples=15,
                max_delta_step=1.0,
                subsample_for_bin=50000,
                random_state=seed + 17 * idx,
                n_jobs=4,
                verbose=-1,
            )
            gb.fit(X_tr, y_tr, sample_weight=sample_weight)
            learners.append(gb)

        # 5th learner: FN-upweighted GBDT (mirrors Rescue learner upweighting on joint table)
        p0 = learners[0].predict_proba(X_tr)[:, 1]  # type: ignore[attr-defined]
        pos_med = float(np.median(p0[y_tr == 1])) if np.any(y_tr == 1) else 0.2
        fn_miss = (y_tr == 1) & (p0 <= pos_med)
        rw = sample_weight.copy() if sample_weight is not None else np.ones(y_tr.size, dtype=float)
        rw[fn_miss] *= 2.5
        gb_rescue = _make_lgbm(seed=seed + 99)
        gb_rescue.fit(X_tr, y_tr, sample_weight=rw)
        learners.append(gb_rescue)

        # 6th learner: Penalized Logistic Regression
        glm = LogisticRegression(C=1.0, max_iter=600, random_state=seed)
        glm.fit(X_tr, y_tr, sample_weight=sample_weight)
        learners.append(glm)

    arm = FittedDecoderArm(
        arm_name=arm_name, preprocessor=prep, estimators=learners, calibrator=None
    )
    arm.calibrator = PlattLogitCalibrator().fit(arm.predict_raw_proba(X_cal), y_cal)
    return arm


def train_decoder_saturation_suite(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    *,
    semantic_types: dict[str, SemanticType] | None = None,
    include_extended_smote: bool = True,
    seed: int = 42,
) -> dict[str, FittedDecoderArm]:
    """Train standard imbalanced-learning decoder interventions on fixed channel X.

    Tests CREDO Prediction 1 (Frontier Saturation): compares natural-prevalence
    calibrated GBDT and GLM against SMOTE, Borderline-SMOTE, ADASYN, cost-sensitive
    reweighting, Gaussian-copula latent augmentation, and soft probability averaging.
    Note: resampling/reweighting arms are recorded BOTH raw (as deployed in most SMOTE
    papers, exposing severe calibration distortion) and post-recalibrated where needed.
    """
    y_tr = np.asarray(y_train, dtype=int).ravel()
    y_c = np.asarray(y_cal, dtype=int).ravel()
    prep = TabularPreprocessor(semantic_types=semantic_types).fit(X_train)
    X_tr = prep.transform(X_train)
    X_c = prep.transform(X_cal)

    arms: dict[str, FittedDecoderArm] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. Calibrated GLM (Penalized Logistic Regression)
        glm = LogisticRegression(C=1.0, max_iter=800, random_state=seed)
        glm.fit(X_tr, y_tr)
        cal_glm = PlattLogitCalibrator().fit(glm.predict_proba(X_c)[:, 1], y_c)
        arms["calibrated_glm"] = FittedDecoderArm("calibrated_glm", prep, [glm], cal_glm)

        # 2. Calibrated GBDT (Natural Prevalence)
        gbdt = _make_lgbm(seed=seed)
        gbdt.fit(X_tr, y_tr)
        cal_gbdt = PlattLogitCalibrator().fit(gbdt.predict_proba(X_c)[:, 1], y_c)
        arms["calibrated_gbdt"] = FittedDecoderArm("calibrated_gbdt", prep, [gbdt], cal_gbdt)

        # 3. Cost-Sensitive GBDT (Inverse class frequency weighting)
        n_pos = max(1, int(y_tr.sum()))
        n_neg = max(1, int((1 - y_tr).sum()))
        spw = float(n_neg / n_pos)
        cs_gbdt = _make_lgbm(seed=seed, scale_pos_weight=spw)
        cs_gbdt.fit(X_tr, y_tr)
        arms["cost_sensitive_gbdt_uncal"] = FittedDecoderArm(
            "cost_sensitive_gbdt_uncal", prep, [cs_gbdt], None
        )
        cal_cs = PlattLogitCalibrator().fit(cs_gbdt.predict_proba(X_c)[:, 1], y_c)
        arms["cost_sensitive_gbdt"] = FittedDecoderArm(
            "cost_sensitive_gbdt", prep, [cs_gbdt], cal_cs
        )

        # 4. SMOTE + GBDT (strictly inside X_tr)
        k_nb = min(5, max(1, n_pos - 1))
        if n_pos >= 3 and (n_pos / max(1, n_neg)) < 0.45:
            try:
                sm = SMOTE(k_neighbors=k_nb, sampling_strategy=0.5, random_state=seed)
                X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
            except ValueError:
                X_sm, y_sm = X_tr, y_tr
        else:
            X_sm, y_sm = X_tr, y_tr
        sm_gbdt = _make_lgbm(seed=seed)
        sm_gbdt.fit(X_sm, y_sm)
        arms["smote_gbdt_uncal"] = FittedDecoderArm("smote_gbdt_uncal", prep, [sm_gbdt], None)
        cal_sm = PlattLogitCalibrator().fit(sm_gbdt.predict_proba(X_c)[:, 1], y_c)
        arms["smote_gbdt"] = FittedDecoderArm("smote_gbdt", prep, [sm_gbdt], cal_sm)

        if include_extended_smote:
            # 5. Borderline-SMOTE + GBDT
            if n_pos >= 6 and (n_pos / max(1, n_neg)) < 0.45:
                try:
                    bsm = BorderlineSMOTE(
                        k_neighbors=k_nb, sampling_strategy=0.5, random_state=seed
                    )
                    X_bsm, y_bsm = bsm.fit_resample(X_tr, y_tr)
                except ValueError:
                    X_bsm, y_bsm = X_sm, y_sm
            else:
                X_bsm, y_bsm = X_sm, y_sm
            bsm_gbdt = _make_lgbm(seed=seed)
            bsm_gbdt.fit(X_bsm, y_bsm)
            cal_bsm = PlattLogitCalibrator().fit(bsm_gbdt.predict_proba(X_c)[:, 1], y_c)
            arms["borderline_smote_gbdt"] = FittedDecoderArm(
                "borderline_smote_gbdt", prep, [bsm_gbdt], cal_bsm
            )

            # 6. ADASYN + GBDT
            if n_pos >= 6 and (n_pos / max(1, n_neg)) < 0.45:
                try:
                    ada = ADASYN(n_neighbors=k_nb, sampling_strategy=0.5, random_state=seed)
                    X_ada, y_ada = ada.fit_resample(X_tr, y_tr)
                except (ValueError, RuntimeError):
                    X_ada, y_ada = X_sm, y_sm
            else:
                X_ada, y_ada = X_sm, y_sm
            ada_gbdt = _make_lgbm(seed=seed)
            ada_gbdt.fit(X_ada, y_ada)
            cal_ada = PlattLogitCalibrator().fit(ada_gbdt.predict_proba(X_c)[:, 1], y_c)
            arms["adasyn_gbdt"] = FittedDecoderArm("adasyn_gbdt", prep, [ada_gbdt], cal_ada)

            # 6b. CLIMB-style EasyEnsembleClassifier (Bagged Balanced AdaBoost)
            eec = EasyEnsembleClassifier(
                n_estimators=15,
                sampling_strategy="auto",
                random_state=seed,
                n_jobs=4,
            )
            try:
                eec.fit(X_tr, y_tr)
                cal_eec = PlattLogitCalibrator().fit(eec.predict_proba(X_c)[:, 1], y_c)
                arms["easy_ensemble"] = FittedDecoderArm("easy_ensemble", prep, [eec], cal_eec)
            except ValueError:
                arms["easy_ensemble"] = arms["smote_gbdt"]

            # 6c. CLIMB-style RUSBoostClassifier (Depth-3 Tree Base + Undersampled Boosting)
            rus = RUSBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, random_state=seed),
                n_estimators=60,
                learning_rate=0.1,
                sampling_strategy="auto",
                random_state=seed,
            )
            try:
                rus.fit(X_tr, y_tr)
                cal_rus = PlattLogitCalibrator().fit(rus.predict_proba(X_c)[:, 1], y_c)
                arms["rusboost"] = FittedDecoderArm("rusboost", prep, [rus], cal_rus)
            except ValueError:
                arms["rusboost"] = arms["easy_ensemble"]

        # 7. Latent Gaussian-Copula Augmentation + GBDT
        X_cop, y_cop = _gaussian_copula_minority_augment(X_tr, y_tr, target_ratio=0.4, seed=seed)
        cop_gbdt = _make_lgbm(seed=seed)
        cop_gbdt.fit(X_cop, y_cop)
        cal_cop = PlattLogitCalibrator().fit(cop_gbdt.predict_proba(X_c)[:, 1], y_c)
        arms["latent_copula_gbdt"] = FittedDecoderArm(
            "latent_copula_gbdt", prep, [cop_gbdt], cal_cop
        )

        # 8. Soft Probability-Averaging Ensemble (GLM + LightGBM + HistGB + BalancedRF)
        hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=seed)
        hgb.fit(X_tr, y_tr)
        brf = BalancedRandomForestClassifier(
            n_estimators=120,
            sampling_strategy="all",
            replacement=True,
            random_state=seed,
            n_jobs=4,
        )
        brf.fit(X_tr, y_tr)
        ens_members: list[object] = [glm, gbdt, hgb, brf]
        raw_ens_cal = np.mean(
            np.column_stack([m.predict_proba(X_c)[:, 1] for m in ens_members]),  # type: ignore[attr-defined]
            axis=1,
        )
        cal_ens = PlattLogitCalibrator().fit(raw_ens_cal, y_c)
        arms["soft_avg_ensemble"] = FittedDecoderArm(
            "soft_avg_ensemble", prep, ens_members, cal_ens
        )

    return arms
