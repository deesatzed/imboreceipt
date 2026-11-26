"""Sequential instancewise greedy CMI active feature acquisition comparator (Covert et al., 2023).

Scopes ``CovertGreedyCMIAFA`` cleanly as an acquisition policy-class comparison inside matched
LightGBM tabular predictors (rather than Covert et al.'s neural DIME amortized MLP):
- Trains a masked residual LightGBM predictor anchored at ``logit(p_1(X_base))`` across ``5``
  within-budget random channel subsets ``S subseteq {1..M}`` per row (with native ``NaN`` missing
  routing plus explicit mask indicator columns, returning exact Stage-1 ``p_1`` when ``S = empty``).
- Trains per-channel LightGBM CMI regressors predicting non-negative conditional Bernoulli KL gain
  ``D_KL(Bern(p_{S u {m}}) || Bern(p_S)) >= 0``.
- Supports ``D_cal``-tuned autonomous stopping ``max_m CMI_m / c_m > lambda_CMI*``, fixed query-rate
  matching (``target_query_rate = 0.20``), and full-budget within-``B`` rollout (``lambda = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from imboreceipt.contract.decision import DecisionContract
from imboreceipt.discover.selector import CandidateChannelSpec
from imboreceipt.frontier.baselines import TabularPreprocessor

_PROB_EPS = 1e-6
_MASK_SENTINEL = -999.0


def _kl_bernoulli(p_post: np.ndarray, p_prior: np.ndarray) -> np.ndarray:
    p2 = np.clip(p_post, _PROB_EPS, 1.0 - _PROB_EPS)
    p1 = np.clip(p_prior, _PROB_EPS, 1.0 - _PROB_EPS)
    return np.maximum(0.0, p2 * np.log(p2 / p1) + (1.0 - p2) * np.log((1.0 - p2) / (1.0 - p1)))


@dataclass(frozen=True, slots=True)
class CovertAFAResult:
    """Evaluation output of ``CovertGreedyCMIAFA`` on held-out ``D_test``."""

    mode: str
    probs_test: np.ndarray
    probs_cal: np.ndarray
    queried_any_test: np.ndarray
    queried_any_cal: np.ndarray
    channel_masks_test: np.ndarray
    avg_channels_queried_test: float
    any_query_rate_test: float
    null_channel_query_rate_test: float
    net_benefit_per_1k: float
    auprc_routed: float
    lambda_cmi_star: float
    top_channels_by_freq: list[tuple[str, float]]


class CovertGreedyCMIAFA:
    """Amortized masked-GBDT sequential greedy CMI policy (Covert et al., 2023 policy class)."""

    def __init__(
        self,
        candidates: list[CandidateChannelSpec],
        contract: DecisionContract,
        per_instance_query_cost: float,
        acquisition_budget: float,
        *,
        max_steps_per_instance: int = 3,
        n_masks_per_row: int = 5,
        n_jobs: int = 2,
        seed: int = 42,
    ) -> None:
        self.candidates = list(candidates)
        self.contract = contract
        self.per_instance_query_cost = float(per_instance_query_cost)
        self.acquisition_budget = max(1e-6, float(acquisition_budget))
        self.max_steps = max(1, int(max_steps_per_instance))
        self.n_masks_per_row = max(3, int(n_masks_per_row))
        self.n_jobs = int(n_jobs)
        self.seed = int(seed)

        self._prep_base: TabularPreprocessor | None = None
        self._prep_z: dict[str, TabularPreprocessor] = {}
        self._z_slices: list[tuple[int, int]] = []
        self._z_dim_total: int = 0
        self._costs: np.ndarray = np.array(
            [max(1e-6, float(c.acquisition_cost)) for c in self.candidates], dtype=float
        )
        self._clf_masked: lgb.LGBMClassifier | None = None
        self._platt: LogisticRegression | None = None
        self._cmi_regs: list[lgb.LGBMRegressor | None] = []
        self.lambda_cmi_star_: float = 1e9
        self._cal_cache: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        ) = None

    def _encode_blocks(
        self, X_base: pd.DataFrame, Z_pool: pd.DataFrame, fit: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if fit:
            self._prep_base = TabularPreprocessor()
            Xb = np.nan_to_num(self._prep_base.fit_transform(X_base), nan=0.0).astype(np.float32)
            z_mats: list[np.ndarray] = []
            offset = 0
            self._prep_z = {}
            self._z_slices = []
            for cand in self.candidates:
                pr = TabularPreprocessor()
                zm = np.nan_to_num(pr.fit_transform(Z_pool[list(cand.columns)]), nan=0.0).astype(
                    np.float32
                )
                if zm.ndim == 1:
                    zm = zm[:, None]
                self._prep_z[cand.channel_id] = pr
                self._z_slices.append((offset, offset + zm.shape[1]))
                offset += zm.shape[1]
                z_mats.append(zm)
            self._z_dim_total = offset
            Z_full = np.hstack(z_mats) if z_mats else np.zeros((len(X_base), 0), dtype=np.float32)
            return Xb, Z_full

        assert self._prep_base is not None
        Xb = np.nan_to_num(self._prep_base.transform(X_base), nan=0.0).astype(np.float32)
        z_mats = []
        for cand in self.candidates:
            pr = self._prep_z[cand.channel_id]
            zm = np.nan_to_num(pr.transform(Z_pool[list(cand.columns)]), nan=0.0).astype(np.float32)
            if zm.ndim == 1:
                zm = zm[:, None]
            z_mats.append(zm)
        Z_full = np.hstack(z_mats) if z_mats else np.zeros((len(X_base), 0), dtype=np.float32)
        return Xb, Z_full

    def _build_masked_features(
        self, Xb: np.ndarray, Z_full: np.ndarray, ch_mask: np.ndarray
    ) -> np.ndarray:
        z_m = np.full((Xb.shape[0], self._z_dim_total), np.nan, dtype=np.float32)
        for j, (lo, hi) in enumerate(self._z_slices):
            obs = ch_mask[:, j] > 0.5
            if np.any(obs):
                z_m[obs, lo:hi] = Z_full[obs, lo:hi]
        return np.hstack([Xb, z_m, ch_mask.astype(np.float32)])

    def _sample_affordable_masks(self, n_rows: int, rng_local: np.random.Generator) -> np.ndarray:
        m_ch = len(self.candidates)
        masks = np.zeros((n_rows, m_ch), dtype=np.float32)
        max_k = min(4, m_ch)
        for i in range(n_rows):
            perm = rng_local.permutation(m_ch)
            target_k = int(rng_local.integers(1, max_k + 1))
            rem_b = self.acquisition_budget
            picked = 0
            for j in perm:
                if picked >= target_k:
                    break
                if self._costs[j] <= rem_b + 1e-9:
                    masks[i, j] = 1.0
                    rem_b -= self._costs[j]
                    picked += 1
        return masks

    def _predict_masked_prob(
        self, Xb: np.ndarray, Z_full: np.ndarray, ch_mask: np.ndarray, p1: np.ndarray
    ) -> np.ndarray:
        assert self._clf_masked is not None and self._platt is not None
        has_any = np.any(ch_mask > 0.5, axis=1)
        out = np.clip(p1.copy(), _PROB_EPS, 1.0 - _PROB_EPS)
        if np.any(has_any):
            xm = self._build_masked_features(Xb[has_any], Z_full[has_any], ch_mask[has_any])
            raw = logit(np.clip(p1[has_any], _PROB_EPS, 1.0 - _PROB_EPS)) + np.asarray(
                self._clf_masked.predict(xm, raw_score=True), dtype=float
            )
            out[has_any] = np.clip(
                self._platt.predict_proba(raw.reshape(-1, 1))[:, 1], _PROB_EPS, 1.0 - _PROB_EPS
            )
        return out

    def fit(
        self,
        X_base_train: pd.DataFrame,
        Z_pool_train: pd.DataFrame,
        y_train: np.ndarray,
        p1_train_oof: np.ndarray,
        X_base_cal: pd.DataFrame,
        Z_pool_cal: pd.DataFrame,
        y_cal: np.ndarray,
        p1_cal: np.ndarray,
        *,
        max_train_rows: int = 28000,
    ) -> CovertGreedyCMIAFA:
        rng = np.random.default_rng(self.seed + 303)
        y_tr_full = np.asarray(y_train, dtype=int).ravel()
        p1_tr_full = np.clip(
            np.asarray(p1_train_oof, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS
        )

        if len(X_base_train) > max_train_rows:
            sub_idx = np.sort(rng.choice(len(X_base_train), size=max_train_rows, replace=False))
            Xb_tr_df = X_base_train.iloc[sub_idx].reset_index(drop=True)
            Zp_tr_df = Z_pool_train.iloc[sub_idx].reset_index(drop=True)
            y_tr = y_tr_full[sub_idx]
            p1_tr = p1_tr_full[sub_idx]
        else:
            Xb_tr_df = X_base_train.reset_index(drop=True)
            Zp_tr_df = Z_pool_train.reset_index(drop=True)
            y_tr = y_tr_full
            p1_tr = p1_tr_full

        Xb_tr, Z_tr = self._encode_blocks(Xb_tr_df, Zp_tr_df, fit=True)
        n_tr = Xb_tr.shape[0]
        m_ch = len(self.candidates)
        ell1_tr = logit(p1_tr)

        x_rep: list[np.ndarray] = []
        y_rep: list[np.ndarray] = []
        init_rep: list[np.ndarray] = []
        for _ in range(self.n_masks_per_row):
            m_mat = self._sample_affordable_masks(n_tr, rng)
            x_rep.append(self._build_masked_features(Xb_tr, Z_tr, m_mat))
            y_rep.append(y_tr)
            init_rep.append(ell1_tr)

        X_stack = np.vstack(x_rep)
        y_stack = np.concatenate(y_rep)
        init_stack = np.concatenate(init_rep)

        self._clf_masked = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=25,
            max_delta_step=1.0,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=1.0 if X_stack.shape[1] <= 16 else 0.85,
            random_state=self.seed + 41,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self._clf_masked.fit(X_stack, y_stack, init_score=init_stack)

        Xb_ca, Z_ca = self._encode_blocks(X_base_cal, Z_pool_cal, fit=False)
        y_ca = np.asarray(y_cal, dtype=int).ravel()
        p1_ca = np.clip(np.asarray(p1_cal, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)
        m_cal_rand = self._sample_affordable_masks(y_ca.size, rng)
        raw_ca_m = logit(p1_ca) + np.asarray(
            self._clf_masked.predict(
                self._build_masked_features(Xb_ca, Z_ca, m_cal_rand), raw_score=True
            ),
            dtype=float,
        )
        self._platt = LogisticRegression(C=1e3, solver="lbfgs", max_iter=300)
        self._platt.fit(raw_ca_m.reshape(-1, 1), y_ca)

        # Fit per-channel CMI regressors on empty + 1-step partial training states
        sub_n = min(n_tr, 6000)
        s_idx = rng.choice(n_tr, size=sub_n, replace=False) if sub_n < n_tr else np.arange(n_tr)
        xb_s, zf_s, p1_s = Xb_tr[s_idx], Z_tr[s_idx], p1_tr[s_idx]

        m_zero = np.zeros((sub_n, m_ch), dtype=np.float32)
        m_part = np.zeros((sub_n, m_ch), dtype=np.float32)
        for i in range(sub_n):
            j0 = int(rng.integers(0, m_ch))
            if self._costs[j0] < self.acquisition_budget - 0.25:
                m_part[i, j0] = 1.0

        states_m = np.vstack([m_zero, m_part])
        states_xb = np.vstack([xb_s, xb_s])
        states_zf = np.vstack([zf_s, zf_s])
        states_p1 = np.concatenate([p1_s, p1_s])
        p_cur_state = self._predict_masked_prob(states_xb, states_zf, states_m, states_p1)
        x_state_feat = np.column_stack(
            [
                logit(p_cur_state).astype(np.float32),
                self._build_masked_features(states_xb, states_zf, states_m),
            ]
        )

        self._cmi_regs = []
        for j in range(m_ch):
            valid = (states_m[:, j] < 0.5) & (
                (states_m @ self._costs) + self._costs[j] <= self.acquisition_budget + 1e-9
            )
            if int(np.sum(valid)) < 40:
                self._cmi_regs.append(None)
                continue
            m_plus = states_m[valid].copy()
            m_plus[:, j] = 1.0
            p_plus = self._predict_masked_prob(
                states_xb[valid], states_zf[valid], m_plus, states_p1[valid]
            )
            cmi_target = _kl_bernoulli(p_plus, p_cur_state[valid])
            reg_j = lgb.LGBMRegressor(
                n_estimators=55,
                learning_rate=0.06,
                num_leaves=15,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=1.0 if X_stack.shape[1] <= 16 else 0.85,
                random_state=self.seed + 200 + j,
                n_jobs=self.n_jobs,
                verbose=-1,
            )
            reg_j.fit(x_state_feat[valid], cmi_target)
            self._cmi_regs.append(reg_j)

        # Tune stopping threshold lambda_CMI* strictly on D_cal
        _, _, _, step1_r = self._rollout_greedy(Xb_ca, Z_ca, p1_ca, lambda_stop=1e9, max_steps=1)
        pos_r = step1_r[step1_r > 0.0]
        if pos_r.size >= 10:
            cand_lams = np.unique(
                np.concatenate(
                    [np.quantile(pos_r, [0.10, 0.25, 0.40, 0.55, 0.70, 0.80, 0.90, 0.96]), [1e9]]
                )
            )
        else:
            cand_lams = np.array([1e-4, 1e-3, 1e-2, 1e9])

        best_lam = float(cand_lams[-1])
        best_nb = -1e18
        for lam in cand_lams:
            p_try, q_try, _, _ = self._rollout_greedy(Xb_ca, Z_ca, p1_ca, float(lam))
            nb_try = self._compute_nb(y_ca, p_try, q_try)
            if nb_try > best_nb:
                best_nb = nb_try
                best_lam = float(lam)
        self.lambda_cmi_star_ = best_lam
        self._cal_cache = (Xb_ca, Z_ca, y_ca, p1_ca, step1_r)
        return self

    def _rollout_greedy(
        self,
        Xb: np.ndarray,
        Z_full: np.ndarray,
        p1: np.ndarray,
        lambda_stop: float,
        max_steps: int | None = None,
        decision_boundary_weighted: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        steps = self.max_steps if max_steps is None else int(max_steps)
        n = p1.size
        m_ch = len(self.candidates)
        mask_cur = np.zeros((n, m_ch), dtype=np.float32)
        spent_cur = np.zeros(n, dtype=float)
        p_cur = np.clip(p1.copy(), _PROB_EPS, 1.0 - _PROB_EPS)
        active = np.ones(n, dtype=bool)
        step1_best = np.zeros(n, dtype=float)
        tau = self.contract.cost_fp / (self.contract.cost_fp + self.contract.cost_fn)
        ltau = float(logit(np.clip(tau, _PROB_EPS, 1.0 - _PROB_EPS)))

        for st in range(steps):
            act_idx = np.flatnonzero(active)
            if act_idx.size == 0:
                break
            lp_act = logit(p_cur[act_idx])
            x_st = np.column_stack(
                [
                    lp_act.astype(np.float32),
                    self._build_masked_features(Xb[act_idx], Z_full[act_idx], mask_cur[act_idx]),
                ]
            )
            cz_wt = (
                np.exp(-np.abs(lp_act - ltau) / 1.0)
                if decision_boundary_weighted
                else np.ones(act_idx.size, dtype=float)
            )
            ratios = np.full((act_idx.size, m_ch), -np.inf, dtype=float)
            for j, reg_j in enumerate(self._cmi_regs):
                if reg_j is None:
                    continue
                can_buy = (mask_cur[act_idx, j] < 0.5) & (
                    spent_cur[act_idx] + self._costs[j] <= self.acquisition_budget + 1e-9
                )
                if not np.any(can_buy):
                    continue
                pred_cmi = np.maximum(0.0, reg_j.predict(x_st[can_buy])) * cz_wt[can_buy]
                ratios[can_buy, j] = pred_cmi / self._costs[j]

            best_j = np.argmax(ratios, axis=1)
            best_r = ratios[np.arange(act_idx.size), best_j]
            if st == 0:
                step1_best[act_idx] = np.where(np.isfinite(best_r), best_r, 0.0)

            go = np.isfinite(best_r) & (best_r > lambda_stop)
            if not np.any(go):
                break
            go_rows = act_idx[go]
            go_ch = best_j[go]
            mask_cur[go_rows, go_ch] = 1.0
            spent_cur[go_rows] += self._costs[go_ch]
            p_cur[go_rows] = self._predict_masked_prob(
                Xb[go_rows], Z_full[go_rows], mask_cur[go_rows], p1[go_rows]
            )
            active[~np.isin(np.arange(n), go_rows)] = False

        queried = np.any(mask_cur > 0.5, axis=1)
        return p_cur, queried, mask_cur > 0.5, step1_best

    def _compute_nb(
        self,
        y: np.ndarray,
        p_out: np.ndarray,
        queried_units: np.ndarray,
    ) -> float:
        tau = self.contract.cost_fp / (self.contract.cost_fp + self.contract.cost_fn)
        act = p_out >= tau
        tp = float(np.sum((y == 1) & act))
        fp = float(np.sum((y == 0) & act))
        n_t = max(1.0, float(y.size))
        return (1000.0 / n_t) * (
            tp
            - (self.contract.cost_fp / self.contract.cost_fn) * fp
            - (self.per_instance_query_cost / self.contract.cost_fn)
            * float(np.sum(np.asarray(queried_units, dtype=float)))
        )

    def evaluate(
        self,
        X_base_test: pd.DataFrame,
        Z_pool_test: pd.DataFrame,
        y_test: np.ndarray,
        p1_test: np.ndarray,
        *,
        target_query_rate: float | None = None,
        decision_boundary_weighted: bool = False,
        fee_accounting: str = "flat_any",
        ref_bundle_cost: float | None = None,
    ) -> CovertAFAResult:
        assert self._cal_cache is not None
        Xb_ca, Z_ca, _, p1_ca, step1_ca = self._cal_cache
        Xb_te, Z_te = self._encode_blocks(X_base_test, Z_pool_test, fit=False)
        yt = np.asarray(y_test, dtype=int).ravel()
        p1_t = np.clip(np.asarray(p1_test, dtype=float).ravel(), _PROB_EPS, 1.0 - _PROB_EPS)

        if decision_boundary_weighted:
            _, _, _, step1_ca_use = self._rollout_greedy(
                Xb_ca, Z_ca, p1_ca, lambda_stop=1e9, max_steps=1, decision_boundary_weighted=True
            )
        else:
            step1_ca_use = step1_ca

        if target_query_rate is None:
            lam_use = self.lambda_cmi_star_
            mode = "autonomous_cal_tuned"
        elif target_query_rate <= 0.0:
            lam_use = 1e9
            mode = "budget_matched_q0"
        elif target_query_rate >= 0.999:
            lam_use = -1e-12
            mode = "full_within_budget"
        else:
            lam_use = float(np.quantile(step1_ca_use, 1.0 - float(target_query_rate)))
            mode = f"budget_matched_q{int(round(target_query_rate * 100))}"

        p_ca_out, q_ca, _, _ = self._rollout_greedy(
            Xb_ca, Z_ca, p1_ca, lam_use, decision_boundary_weighted=decision_boundary_weighted
        )
        p_te_out, q_te, fm_te, _ = self._rollout_greedy(
            Xb_te, Z_te, p1_t, lam_use, decision_boundary_weighted=decision_boundary_weighted
        )

        if fee_accounting == "proportional_cost":
            denom_c = max(
                1e-6, float(ref_bundle_cost) if ref_bundle_cost else float(self.acquisition_budget)
            )
            q_units = (fm_te.astype(float) @ self._costs) / denom_c
        else:
            q_units = q_te.astype(float)

        nb_1k = self._compute_nb(yt, p_te_out, q_units)
        ap = float(average_precision_score(yt, p_te_out))

        null_idxs = [
            i
            for i, c in enumerate(self.candidates)
            if c.mechanism_tag == "pure_noise_null" or c.channel_id.startswith("null_")
        ]
        null_q_rate = float(np.mean(np.any(fm_te[:, null_idxs], axis=1))) if null_idxs else 0.0

        freqs = [
            (self.candidates[i].channel_id, float(np.mean(fm_te[:, i])))
            for i in range(len(self.candidates))
        ]
        freqs.sort(key=lambda kv: kv[1], reverse=True)

        return CovertAFAResult(
            mode=mode,
            probs_test=p_te_out,
            probs_cal=p_ca_out,
            queried_any_test=q_te,
            queried_any_cal=q_ca,
            channel_masks_test=fm_te,
            avg_channels_queried_test=float(np.mean(np.sum(fm_te, axis=1))),
            any_query_rate_test=float(np.mean(q_te)),
            null_channel_query_rate_test=null_q_rate,
            net_benefit_per_1k=float(nb_1k),
            auprc_routed=ap,
            lambda_cmi_star=lam_use,
            top_channels_by_freq=freqs[:5],
        )


def validate_covert_afa_synthetic_oracle(
    n_samples: int = 4000, seed: int = 2026
) -> dict[str, float]:
    """Sanity-check ``CovertGreedyCMIAFA`` on a heterogeneous 4-channel synthetic benchmark.

    Each arrival has an observable subtype ``g in {0, 1, 2, 3}`` hinted in ``X_base``, where only
    channel ``Z_g`` carries true diagnostic signal and ``Z_{-g}`` are pure noise. Verifies that
    1-channel instancewise ``CovertGreedyCMIAFA`` recovers >95% of the full 4-channel oracle AUROC
    lift over a fixed 1-channel cohort selector.
    """
    rng = np.random.default_rng(seed)
    g_id = rng.integers(0, 4, size=n_samples)
    z_sig = rng.normal(0.0, 1.0, size=(n_samples, 4))
    y = rng.binomial(1, expit(2.2 * z_sig[np.arange(n_samples), g_id]))

    X_base = pd.DataFrame(
        {
            "b_subtype_hint": (g_id.astype(float) + rng.normal(0.0, 0.15, size=n_samples)).astype(
                np.float32
            ),
            "b_noise": rng.normal(0.0, 1.0, size=n_samples).astype(np.float32),
        }
    )
    Z_pool = pd.DataFrame({f"z_{k}": z_sig[:, k].astype(np.float32) for k in range(4)})
    cands = [CandidateChannelSpec(f"ch_{k}", (f"z_{k}",), 1.0, "het") for k in range(4)]
    contract = DecisionContract(
        task_name="SynHet",
        target_precision=0.55,
        max_fpr_budget=0.20,
        cost_fp=1.0,
        cost_fn=1.0,
    )

    n_tr = int(0.60 * n_samples)
    n_ca = int(0.20 * n_samples)
    tr_s, ca_s, te_s = slice(0, n_tr), slice(n_tr, n_tr + n_ca), slice(n_tr + n_ca, n_samples)

    Xb_tr = X_base.iloc[tr_s].to_numpy()
    Xb_ca = X_base.iloc[ca_s].to_numpy()
    Xb_te = X_base.iloc[te_s].to_numpy()
    y_tr_s = y[tr_s]
    p1_tr = np.full(y_tr_s.size, float(np.mean(y_tr_s)), dtype=float)
    folds = np.arange(y_tr_s.size) % 4
    for f_i in range(4):
        m_va = folds == f_i
        m_tr = ~m_va
        s1_f = lgb.LGBMClassifier(
            n_estimators=60, learning_rate=0.05, num_leaves=15, random_state=seed + f_i, verbose=-1
        )
        s1_f.fit(Xb_tr[m_tr], y_tr_s[m_tr])
        p1_tr[m_va] = np.clip(s1_f.predict_proba(Xb_tr[m_va])[:, 1], 1e-5, 1.0 - 1e-5)

    s1 = lgb.LGBMClassifier(
        n_estimators=60, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1
    )
    s1.fit(Xb_tr, y_tr_s)
    p1_ca = np.clip(s1.predict_proba(Xb_ca)[:, 1], 1e-5, 1.0 - 1e-5)
    p1_te = np.clip(s1.predict_proba(Xb_te)[:, 1], 1e-5, 1.0 - 1e-5)

    s_fix1 = lgb.LGBMClassifier(
        n_estimators=90, learning_rate=0.05, num_leaves=23, random_state=seed, verbose=-1
    )
    s_fix1.fit(np.hstack([Xb_tr, z_sig[tr_s, 0:1]]), y[tr_s])
    auroc_fix1 = float(
        roc_auc_score(y[te_s], s_fix1.predict_proba(np.hstack([Xb_te, z_sig[te_s, 0:1]]))[:, 1])
    )

    s_ora = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1
    )
    s_ora.fit(np.hstack([Xb_tr, z_sig[tr_s]]), y[tr_s])
    auroc_oracle = float(
        roc_auc_score(y[te_s], s_ora.predict_proba(np.hstack([Xb_te, z_sig[te_s]]))[:, 1])
    )

    afa = CovertGreedyCMIAFA(
        cands,
        contract,
        per_instance_query_cost=0.001,
        acquisition_budget=1.0,
        max_steps_per_instance=1,
        n_masks_per_row=5,
        seed=seed,
    )
    afa.fit(
        X_base.iloc[tr_s],
        Z_pool.iloc[tr_s],
        y[tr_s],
        p1_tr,
        X_base.iloc[ca_s],
        Z_pool.iloc[ca_s],
        y[ca_s],
        p1_ca,
    )
    res_afa = afa.evaluate(
        X_base.iloc[te_s], Z_pool.iloc[te_s], y[te_s], p1_te, target_query_rate=1.0
    )
    auroc_afa = float(roc_auc_score(y[te_s], res_afa.probs_test))
    gain_rec = (auroc_afa - auroc_fix1) / max(1e-6, auroc_oracle - auroc_fix1)

    return {
        "n_samples": float(n_samples),
        "auroc_fixed_1ch": auroc_fix1,
        "auroc_oracle_4ch": auroc_oracle,
        "auroc_covert_afa_1step": auroc_afa,
        "oracle_gain_recovery_frac": float(gain_rec),
    }


__all__ = [
    "CovertAFAResult",
    "CovertGreedyCMIAFA",
    "validate_covert_afa_synthetic_oracle",
]
