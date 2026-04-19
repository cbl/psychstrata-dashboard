"""Tests for DeepHitMDD — placeholder training path, predict, adapter, save/load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.deephit_mdd.adapter import (
    CYP_COLS,
    HLA_COLS,
    POPULATION_PC_COLS,
    PRS_COLS,
    ui_to_pipeline_df,
)
from models.deephit_mdd.model import DeepHitMDD


@pytest.fixture(scope="module")
def deephit(tmp_path_factory) -> DeepHitMDD:
    artifacts = tmp_path_factory.mktemp("deephit_artifacts")
    return DeepHitMDD.build(artifacts_dir=artifacts)


def _defaults(model: DeepHitMDD) -> dict:
    return {f.id: f.default for f in model.features}


def test_build_produces_fitted_pipeline_and_network(deephit: DeepHitMDD) -> None:
    assert deephit._pipeline is not None and deephit._model is not None
    assert deephit._model.config.n_causes >= 1
    assert deephit._model.config.n_time_bins == deephit._pipeline.n_time_bins


def test_adapter_adds_missing_genetics_and_outcome_columns() -> None:
    raw = {f.id: f.default for f in DeepHitMDD().features}
    df = ui_to_pipeline_df(raw)
    for col in PRS_COLS + POPULATION_PC_COLS:
        assert col in df.columns
        assert pd.isna(df[col].iloc[0])
    for col in CYP_COLS:
        assert df[col].iloc[0] == "NM"
    for col in HLA_COLS:
        assert df[col].iloc[0] == 0
    assert df["diagnosis"].iloc[0] == "MDD"
    assert df["T_tr"].iloc[0] == 0.0
    assert df["E_tr"].iloc[0] == 0


def test_adapter_encodes_sex_as_binary() -> None:
    raw = {f.id: f.default for f in DeepHitMDD().features}
    assert ui_to_pipeline_df({**raw, "sex": "Female"}).loc[0, "sex"] == 0
    assert ui_to_pipeline_df({**raw, "sex": "Male"}).loc[0, "sex"] == 1


def test_predict_returns_risk_curve_and_valid_probability(deephit: DeepHitMDD) -> None:
    p = deephit.predict(_defaults(deephit))
    assert 0.0 <= p.probability <= 1.0
    assert p.label in {"Resistant", "Responsive"}
    assert p.risk_curve is not None and len(p.risk_curve) > 0
    assert all(0.0 <= v <= 1.0 for v in p.risk_curve.values())


def test_explain_returns_empty_placeholder(deephit: DeepHitMDD) -> None:
    exp = deephit.explain(_defaults(deephit))
    assert exp.shap_values == {}
    assert exp.top_positive == []
    assert exp.top_negative == []


def test_save_and_reload_round_trip(deephit: DeepHitMDD, tmp_path: Path) -> None:
    deephit.save(tmp_path)
    reloaded = DeepHitMDD.from_pretrained(tmp_path)
    raw = _defaults(deephit)
    p1 = deephit.predict(raw)
    p2 = reloaded.predict(raw)
    assert abs(p1.probability - p2.probability) < 1e-6
    assert p1.label == p2.label
