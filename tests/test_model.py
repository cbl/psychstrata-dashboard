"""Tests for RandomForestMDD — train, predict, explain, save/load round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.random_forest_mdd.model import RandomForestMDD


@pytest.fixture(scope="module")
def trained_model() -> RandomForestMDD:
    return RandomForestMDD.train(n_samples=200, random_state=0)


def _defaults(model: RandomForestMDD) -> dict:
    return {f.id: f.default for f in model.features}


def test_train_produces_fitted_classifier_with_finite_auc(trained_model: RandomForestMDD) -> None:
    assert 0.0 <= trained_model.auc <= 1.0
    assert trained_model.tsne_embedding.shape[0] == 200
    assert trained_model.y_labels.shape[0] == 200


def test_predict_on_defaults_returns_valid_prediction(trained_model: RandomForestMDD) -> None:
    p = trained_model.predict(_defaults(trained_model))
    assert p.label in {"Resistant", "Responsive", "Uncertain"}
    assert 0.0 <= p.probability <= 1.0
    assert p.conformal_set is not None
    assert p.cause_curves is None


def test_explain_returns_shap_for_every_model_column(trained_model: RandomForestMDD) -> None:
    exp = trained_model.explain(_defaults(trained_model))
    assert set(exp.shap_values.keys()) == set(trained_model.encoder.model_feature_names)
    assert len(exp.top_positive) <= 3 and len(exp.top_negative) <= 3
    for c in exp.top_positive + exp.top_negative:
        assert c.human_label
        assert isinstance(c.shap_value, float)


def test_save_and_from_pretrained_round_trip(trained_model: RandomForestMDD, tmp_path: Path) -> None:
    trained_model.save(tmp_path)
    reloaded = RandomForestMDD.from_pretrained(tmp_path)
    assert reloaded.auc == trained_model.auc
    assert reloaded.encoder.model_feature_names == trained_model.encoder.model_feature_names
    raw = _defaults(reloaded)
    p1 = trained_model.predict(raw)
    p2 = reloaded.predict(raw)
    assert p1.label == p2.label
    assert abs(p1.probability - p2.probability) < 1e-9
