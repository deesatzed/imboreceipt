"""Gate 0.5 Model Organism: Multi-mechanism rare-event & selective-receipt simulator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from imboreceipt.contract.decision import DecisionContract
from imboreceipt.core.types import SemanticType
from imboreceipt.discover.compilers import (
    ConservationMismatchCompiler,
    ConservationRule,
    CriticalSlowingDownCompiler,
    MissingLineSilenceCompiler,
)
from imboreceipt.discover.selector import CandidateChannelSpec


@dataclass(slots=True)
class MultiChannelCohort:
    """Standardized multi-channel rare-event cohort for CREDO-Bench evaluation."""

    name: str
    X_base: pd.DataFrame
    Z_pool: pd.DataFrame
    candidates: list[CandidateChannelSpec]
    ground_truth_relevant_channels: set[str]
    y_true: np.ndarray
    y_tilde: np.ndarray
    contract: DecisionContract
    acquisition_budget: float
    base_semantic_types: dict[str, SemanticType] | None = None
    group_ids: np.ndarray | None = None
    time_order: np.ndarray | None = None
    traps_df: pd.DataFrame | None = None
    future_positive_mask: np.ndarray | None = None
    competing_censor_mask: np.ndarray | None = None
    receipt_cols: list[str] | None = None
    mechanism_subtype: np.ndarray | None = None
    raw_aux_df: pd.DataFrame | None = None
    compiler_factories: list[Callable[[], Any]] | None = None
    disjoint_test_blocks: bool = False

    def materialize_z_splits(
        self,
        train_idx: np.ndarray,
        *other_idxs: np.ndarray,
    ) -> list[pd.DataFrame]:
        """Fit stateful compilers strictly on ``train_idx`` and return materialized Z splits."""
        all_idxs = (train_idx, *other_idxs)
        base_parts = [self.Z_pool.iloc[idx].reset_index(drop=True) for idx in all_idxs]
        if self.raw_aux_df is None or not self.compiler_factories:
            return base_parts

        aux_tr = self.raw_aux_df.iloc[train_idx].reset_index(drop=True)
        fitted_compilers = [fac().fit(aux_tr) for fac in self.compiler_factories]
        out: list[pd.DataFrame] = []
        for base_df, idx in zip(base_parts, all_idxs, strict=True):
            aux_sub = self.raw_aux_df.iloc[idx].reset_index(drop=True)
            comp_dfs = [comp.transform(aux_sub) for comp in fitted_compilers]
            out.append(pd.concat([base_df, *comp_dfs], axis=1))
        return out


def generate_gate05_cohort(
    n_samples: int = 12000,
    seed: int = 42,
    selective_receipts: bool = True,
) -> MultiChannelCohort:
    """Generate Gate 0.5 synthetic ground-truth cohort with 4 rare-event mechanisms & receipts.

    Ground-truth rare-event mechanisms (overall true prevalence ~3.5%):
    1. ``transition_csd``: Dynamical regime shift preceded by Critical Slowing Down (rising
       variance + lag-1 autocorrelation in a 6-step trajectory).
    2. ``conservation_silence``: Accounting flow violation + sudden silence in a previously
       regular activity cadence ("the missing line").
    3. ``orthogonal_coincidence``: Ambiguous in ``X_base`` due to a shared false-positive
       confounder ``u_fp`` that contaminates both ``X_base`` and ``Z_confounded_fp``,
       resolvable ONLY by orthogonal channel ``Z_orth_biomarker`` (where ``kappa_0 approx 1``).
    4. ``exogenous_shock``: Memoryless random failure (~0.4% rate) completely independent of
       all covariates, establishing a strict non-zero aleatoric ceiling.
    """
    rng = np.random.default_rng(seed)

    # Shared false-positive confounder among negatives (triggers false alarms in X_base & decoy)
    u_fp = rng.exponential(scale=1.0, size=n_samples) * rng.binomial(1, 0.12, size=n_samples)
    stress_load = rng.normal(0.0, 1.0, size=n_samples)
    age_norm = rng.normal(0.0, 1.0, size=n_samples)
    site_code = rng.integers(101, 108, size=n_samples)  # Nominal integer ID code

    # Mechanism 1: Critical Slowing Down (CSD) regime transition (~1.1% prevalence)
    is_csd_candidate = (stress_load > 0.8) & (rng.uniform(size=n_samples) < 0.18)
    y_csd = (is_csd_candidate & (rng.uniform(size=n_samples) < 0.35)).astype(int)

    traj = np.zeros((n_samples, 6), dtype=float)
    unit_baseline = rng.normal(50.0, 8.0, size=n_samples)
    for t in range(6):
        phi = np.where(y_csd == 1, 0.25 + 0.13 * t, 0.20)
        sigma = np.where(y_csd == 1, 0.8 + 0.55 * t, 1.0)
        eps = rng.normal(0.0, sigma, size=n_samples)
        if t == 0:
            traj[:, t] = unit_baseline + eps
        else:
            dev_prev = traj[:, t - 1] - unit_baseline
            traj[:, t] = unit_baseline + phi * dev_prev + eps
    # Drift on transitioning units in final steps
    traj[:, 4] += y_csd * 2.2
    traj[:, 5] += y_csd * 3.8

    # Mechanism 2: Conservation violation + Missing-Line Silence (~1.0% prevalence)
    y_cons = ((age_norm < -0.4) & (rng.uniform(size=n_samples) < 0.032)).astype(int)
    inflow = rng.lognormal(mean=4.0, sigma=0.5, size=n_samples)
    expected_outflow = inflow * rng.normal(1.0, 0.03, size=n_samples)
    # Violate flow balance on y_cons positives
    observed_outflow = expected_outflow + y_cons * inflow * rng.uniform(0.35, 0.80, size=n_samples)

    cadence = np.zeros((n_samples, 6), dtype=float)
    active_user = rng.binomial(1, 0.80, size=n_samples)
    for t in range(6):
        prob_act = np.where(active_user == 1, 0.90, 0.15)
        if t >= 4:
            # Expected heartbeat/payment suddenly goes silent on y_cons==1!
            prob_act = np.where(y_cons == 1, 0.02, prob_act)
        cadence[:, t] = rng.binomial(1, prob_act) * rng.uniform(10.0, 50.0, size=n_samples)

    # Mechanism 3: High-overlap baseline signal needing orthogonal confirmation (~1.1% prevalence)
    latent_bio = rng.normal(0.0, 1.0, size=n_samples)
    y_orth = ((stress_load + 1.4 * latent_bio > 2.9) & (rng.uniform(size=n_samples) < 0.55)).astype(
        int
    )

    # Mechanism 4: Pure memoryless exogenous shock (~0.35% prevalence)
    y_shock = rng.binomial(1, 0.0035, size=n_samples)

    y_true = np.clip(y_csd + y_cons + y_orth + y_shock, 0, 1)
    subtype = np.full(n_samples, "none", dtype=object)
    subtype[y_shock == 1] = "exogenous_shock"
    subtype[y_cons == 1] = "conservation_silence"
    subtype[y_csd == 1] = "transition_csd"
    subtype[y_orth == 1] = "orthogonal_coincidence"

    # Observed baseline channel X_base (contaminated by u_fp on negatives!)
    x_scout_score = 1.1 * stress_load + 0.8 * y_orth + 1.85 * u_fp + rng.normal(0, 0.8, n_samples)
    # Receipt ordering count (clinician/investigator orders extra checks when u_fp or stress high)
    order_intensity = rng.poisson(lam=np.exp(0.4 + 0.45 * stress_load + 0.5 * u_fp))

    X_base = pd.DataFrame(
        {
            "stress_index": stress_load,
            "scout_signal": x_scout_score,
            "age_z": age_norm,
            "snapshot_latest": traj[:, -1],
            "site_id": site_code,
            "monitoring_orders": order_intensity,
        }
    )
    # Inject realistic informative missingness
    miss_m = rng.uniform(size=n_samples) < expit(-2.2 - 0.8 * stress_load)
    X_base.loc[miss_m, "scout_signal"] = np.nan

    # Raw longitudinal & auxiliary columns; stateful compilers MUST be fit strictly on train!
    raw_aux = pd.DataFrame(
        {
            **{f"traj_t{t}": traj[:, t] for t in range(6)},
            **{f"cad_t{t}": cadence[:, t] for t in range(6)},
            "flow_in": inflow,
            "flow_out": observed_outflow,
        }
    )
    compiler_factories: list[Callable[[], Any]] = [
        lambda: CriticalSlowingDownCompiler({"vital": [f"traj_t{t}" for t in range(6)]}),
        lambda: MissingLineSilenceCompiler({"heartbeat": [f"cad_t{t}" for t in range(6)]}),
        lambda: ConservationMismatchCompiler(
            [ConservationRule("ledger", "flow_balance", ("flow_in",), ("flow_out",))]
        ),
    ]
    csd_cols = (
        "csd__vital__diff_base",
        "csd__vital__log_var_inflation",
        "csd__vital__ac1",
        "csd__vital__slope",
    )
    eesh_inv_cols = (
        "eesh__heartbeat__silence_streak",
        "eesh__heartbeat__silence_hazard",
        "inv__ledger__signed_z",
        "inv__ledger__violation_abs",
    )

    # Orthogonal confirmation channel (independent of u_fp) vs Confounded channel (shares u_fp)
    z_orth_1 = 1.9 * latent_bio + 1.1 * y_orth + rng.normal(0.0, 0.85, size=n_samples)
    z_orth_2 = 1.3 * latent_bio + 0.9 * y_csd + rng.normal(0.0, 1.0, size=n_samples)

    # Confounded channel looks predictive globally (high standalone AUC!) because it correlates with
    # stress_load, PLUS shares u_fp with X_base so its false positives collide (kappa_0 >> 1)!
    z_confounded = 1.1 * stress_load + 2.1 * u_fp + 0.35 * y_orth + rng.normal(0, 0.7, n_samples)

    Z_pool = pd.DataFrame(
        {
            "z_orth_bio1": z_orth_1,
            "z_orth_bio2": z_orth_2,
            "z_confounded_fp": z_confounded,
            "z_global_weak": 0.45 * stress_load + rng.normal(0, 1.0, n_samples),
            "z_decoy_noise1": rng.normal(0, 1.0, n_samples),
            "z_decoy_noise2": rng.laplace(0, 1.0, n_samples),
        }
    )

    candidates = [
        CandidateChannelSpec(
            "ch_orth_biomarker",
            ("z_orth_bio1", "z_orth_bio2"),
            acquisition_cost=1.0,
            mechanism_tag="orthogonal_confirmation",
        ),
        CandidateChannelSpec(
            "ch_csd_dynamics",
            csd_cols,
            acquisition_cost=1.0,
            mechanism_tag="critical_slowing_down",
        ),
        CandidateChannelSpec(
            "ch_conservation_silence",
            eesh_inv_cols,
            acquisition_cost=1.0,
            mechanism_tag="conservation_and_missing_line",
        ),
        CandidateChannelSpec(
            "ch_confounded_fp_trap",
            ("z_confounded_fp",),
            acquisition_cost=1.0,
            mechanism_tag="confounded_fp",
        ),
        CandidateChannelSpec(
            "ch_redundant_global",
            ("z_global_weak",),
            acquisition_cost=1.0,
            mechanism_tag="redundant",
        ),
        CandidateChannelSpec(
            "ch_decoy_1",
            ("z_decoy_noise1",),
            acquisition_cost=1.0,
            mechanism_tag="noise_decoy",
        ),
        CandidateChannelSpec(
            "ch_decoy_2",
            ("z_decoy_noise2",),
            acquisition_cost=1.0,
            mechanism_tag="noise_decoy",
        ),
    ]

    # Selective Receipt Generation (Stage R ground truth)
    if selective_receipts:
        trap1_prob = expit(-0.35 + 0.9 * stress_load + 0.4 * u_fp)
        trap2_prob = expit(-0.20 - 0.5 * age_norm)
        trap_1 = (y_true == 1) & (rng.uniform(size=n_samples) < trap1_prob)
        trap_2 = (y_true == 1) & (rng.uniform(size=n_samples) < trap2_prob)

        # Horizon censoring: delayed positives after the 30-day window are recorded as 0!
        future_pos = (y_true == 1) & (~(trap_1 | trap_2)) & (rng.uniform(size=n_samples) < 0.65)
        # Competing event censoring: 4% of negatives transferred/expired before window end
        comp_censor = (y_true == 0) & (rng.uniform(size=n_samples) < 0.04)

        y_tilde = (trap_1 | trap_2).astype(int)
        traps_df: pd.DataFrame | None = pd.DataFrame(
            {"trap_rules": trap_1.astype(int), "trap_dispute": trap_2.astype(int)}
        )
    else:
        y_tilde = y_true.copy()
        traps_df = None
        future_pos = np.zeros(n_samples, dtype=bool)
        comp_censor = np.zeros(n_samples, dtype=bool)

    sem_types = {
        "stress_index": SemanticType.CONTINUOUS,
        "scout_signal": SemanticType.CONTINUOUS,
        "age_z": SemanticType.CONTINUOUS,
        "snapshot_latest": SemanticType.CONTINUOUS,
        "site_id": SemanticType.NOMINAL,
        "monitoring_orders": SemanticType.RECEIPT_FLAG,
    }

    return MultiChannelCohort(
        name="Gate05_Synthetic_MultiMechanism",
        X_base=X_base,
        Z_pool=Z_pool,
        candidates=candidates,
        ground_truth_relevant_channels={
            "ch_orth_biomarker",
            "ch_csd_dynamics",
            "ch_conservation_silence",
        },
        y_true=y_true,
        y_tilde=y_tilde,
        contract=DecisionContract(
            task_name="Gate05_MultiMechanism",
            target_precision=0.45,
            max_fpr_budget=0.01,
            cost_fp=1.0,
            cost_fn=25.0,
            graded_action_cost_fp=0.20,
        ),
        acquisition_budget=3.0,
        base_semantic_types=sem_types,
        traps_df=traps_df,
        future_positive_mask=future_pos,
        competing_censor_mask=comp_censor,
        receipt_cols=["monitoring_orders"],
        mechanism_subtype=subtype,
        raw_aux_df=raw_aux,
        compiler_factories=compiler_factories,
    )
