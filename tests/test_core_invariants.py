"""Tests for core invariants: leak-free splits, nominal int64 safety, 3-ID dedup, CREDO math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from imboreceipt.contract import (
    DecisionContract,
    achieved_ppv,
    gate_c_audit,
    instance_evidence_deficit,
    required_likelihood_ratio,
    required_positives_for_resolution,
)
from imboreceipt.core import (
    CanonicalNode,
    SemanticType,
    canonical_plan_hash,
    freeze_split,
    infer_semantic_types,
    is_operator_compatible,
    paired_cluster_bootstrap,
    prediction_signature_hash,
)
from imboreceipt.discover import TypedFeatureDAGCompiler
from imboreceipt.frontier import TabularPreprocessor


def test_credo_exact_paper_identities() -> None:
    """Verify exact numerical anchors cited in the CREDO manuscript."""
    # At pi = 0.001, q = 0.30 -> required LR ~= 428.14
    lr_req = required_likelihood_ratio(0.001, 0.30)
    assert lr_req == pytest.approx((0.3 / 0.7) / (0.001 / 0.999), rel=1e-6)
    assert 428.0 < lr_req < 428.5

    # At pi = 0.001, TPR = 0.80, FPR = 0.01 -> PPV ~= 7.41%
    ppv = achieved_ppv(0.001, 0.80, 0.01)
    assert ppv == pytest.approx(0.0008 / (0.0008 + 0.00999), rel=1e-6)

    # Required test positives at r = 0.80, half-width = 0.05 -> 246
    n_req = required_positives_for_resolution(half_width=0.05, reference_recall=0.80)
    assert n_req == 246

    # Deficit D_q(x): if stage 1 reaches p=0.07414, residual LR factor is ~5.35 (log(5.35) ~= 1.677)
    deficits = instance_evidence_deficit(np.array([ppv, 0.40]), 0.001, 0.30)
    assert np.exp(deficits[0]) == pytest.approx(lr_req / 80.0, rel=1e-2)
    assert deficits[1] == 0.0

    audit = gate_c_audit(
        DecisionContract(task_name="test", target_precision=0.30),
        prevalence=0.001,
        n_pos_test=50,
    )
    assert not audit.thermometer_adequate
    assert not audit.one_stage_plausible


def test_grouped_and_chronological_splits_zero_leakage() -> None:
    rng = np.random.default_rng(7)
    patient_ids = np.repeat(np.arange(120), 4)
    y_grp = np.repeat((rng.uniform(size=120) < 0.2).astype(int), 4)
    df = pd.DataFrame(
        {
            "patient_nbr": patient_ids,
            "x": rng.normal(size=480),
            "timestamp": np.arange(480),
            "y": y_grp,
        }
    )

    sp_grp = freeze_split(df, "y", group_col="patient_nbr", seed=11)
    p_tr = set(df.iloc[list(sp_grp.train_idx)]["patient_nbr"])
    p_cal = set(df.iloc[list(sp_grp.cal_idx)]["patient_nbr"])
    p_val = set(df.iloc[list(sp_grp.val_idx)]["patient_nbr"])
    p_tst = set(df.iloc[list(sp_grp.test_idx)]["patient_nbr"])

    assert not (p_tr & p_cal)
    assert not (p_tr & p_val)
    assert not (p_tr & p_tst)
    assert not (p_cal & p_val)
    assert not (p_cal & p_tst)
    assert not (p_val & p_tst)

    sp_time = freeze_split(df, "y", time_col="timestamp", seed=11)
    assert (
        df.iloc[list(sp_time.train_idx)]["timestamp"].max()
        < df.iloc[list(sp_time.cal_idx)]["timestamp"].min()
    )
    assert (
        df.iloc[list(sp_time.cal_idx)]["timestamp"].max()
        < df.iloc[list(sp_time.val_idx)]["timestamp"].min()
    )
    assert (
        df.iloc[list(sp_time.val_idx)]["timestamp"].max()
        < df.iloc[list(sp_time.test_idx)]["timestamp"].min()
    )


def test_nominal_int64_override_no_drift_and_blocks_arithmetic() -> None:
    """Ensure int64 hospital ID codes overridden to NOMINAL never drift at transform or divide."""
    df_tr = pd.DataFrame(
        {
            "discharge_disposition_id": [1, 3, 6, 1, 11, 3],
            "num_inpatient": [0, 2, 1, 0, 4, 1],
        }
    )
    df_te = pd.DataFrame(
        {
            "discharge_disposition_id": [1, 99, 3],  # unseen int64 code 99
            "num_inpatient": [1, 0, 3],
        }
    )
    sem = infer_semantic_types(df_tr, overrides={"discharge_disposition_id": SemanticType.NOMINAL})
    assert sem["discharge_disposition_id"] == SemanticType.NOMINAL
    assert not is_operator_compatible(
        "ratio", (sem["num_inpatient"], sem["discharge_disposition_id"])
    )

    prep = TabularPreprocessor(semantic_types=sem).fit(df_tr)
    mat_te = prep.transform(df_te)
    assert mat_te.shape[0] == 3
    assert np.all(np.isfinite(mat_te))

    sem_ord = infer_semantic_types(
        df_tr,
        overrides={
            "discharge_disposition_id": SemanticType.NOMINAL,
            "num_inpatient": SemanticType.ORDINAL,
        },
    )
    mat_ord = TabularPreprocessor(semantic_types=sem_ord).fit(df_tr).transform(df_te)
    assert mat_ord.shape[0] == 3
    assert np.all(np.isfinite(mat_ord))

    bad_dag = TypedFeatureDAGCompiler(
        {"illegal": [CanonicalNode("n1", "ratio", ("num_inpatient", "discharge_disposition_id"))]},
        semantic_types=sem,
    )
    with pytest.raises(ValueError, match="Illegal semantic operation"):
        bad_dag.fit(df_tr)


def test_three_identity_canonical_and_prediction_deduplication() -> None:
    rng = np.random.default_rng(19)
    df = pd.DataFrame({"a": rng.normal(10, 2, 80), "b": rng.normal(5, 1, 80)})
    sem = {"a": SemanticType.CONTINUOUS, "b": SemanticType.CONTINUOUS}

    n_sum_ab = [CanonicalNode("u", "sum", ("a", "b"))]
    n_add_ba = [CanonicalNode("v", "add", ("b", "a"))]
    assert canonical_plan_hash(n_sum_ab, sem) == canonical_plan_hash(n_add_ba, sem)

    # Affine negated rank-equivalent vector should share prediction_signature_hash
    v1 = df["a"].to_numpy() - df["b"].to_numpy()
    v2 = -3.5 * v1 + 12.0
    assert prediction_signature_hash(v1) == prediction_signature_hash(v2)

    compiler = TypedFeatureDAGCompiler(
        {
            "p1": n_sum_ab,
            "p2_alias": n_add_ba,
            "p3_diff_ab": [CanonicalNode("d1", "difference", ("a", "b"))],
            "p4_diff_ba": [CanonicalNode("d2", "difference", ("b", "a"))],
        },
        semantic_types=sem,
    ).fit(df)
    # p2_alias shares plan_hash with p1; p4_diff_ba is -p3_diff_ab and shares prediction_hash!
    assert list(compiler.accepted_plans_.keys()) == ["p1", "p3_diff_ab"]


def test_paired_cluster_bootstrap_resolves_correlated_lift() -> None:
    rng = np.random.default_rng(23)
    clusters = np.repeat(np.arange(100), 3)
    y = np.repeat((rng.uniform(size=100) < 0.25).astype(int), 3)
    base_noise = np.repeat(rng.normal(0, 1.0, size=100), 3)
    s_a = 0.8 * y + base_noise
    s_b = 1.25 * y + base_noise + rng.normal(0, 0.1, size=300)

    from sklearn.metrics import roc_auc_score

    res = paired_cluster_bootstrap(y, s_a, s_b, roc_auc_score, groups=clusters, n_boot=300, seed=5)
    assert res.delta_point > 0.02
    assert res.ci_low > 0.0
    assert res.statistically_significant
