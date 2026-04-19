"""Tests for the REST API wrappers over TRModel."""

from __future__ import annotations

import json

import pytest
from flask import Flask

from api import register_api
from models.random_forest_mdd.model import RandomForestMDD


@pytest.fixture(scope="module")
def client():
    model = RandomForestMDD.train(n_samples=200, random_state=0)
    server = Flask(__name__)
    register_api(server, model)
    server.testing = True
    with server.test_client() as c:
        yield c, model


def _defaults(model: RandomForestMDD) -> dict:
    return {f.id: f.default for f in model.features}


def _post_predict(c, payload: dict):
    return c.post("/api/predict", data=json.dumps(payload), content_type="application/json")


def test_health_reports_model_and_diagnosis(client) -> None:
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["diagnosis"] == "MDD"


def test_features_returns_full_schema(client) -> None:
    c, model = client
    r = c.get("/api/features")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["features"]) == len(model.features)
    assert body["model_feature_names"] == model.encoder.model_feature_names


def test_predict_with_defaults_returns_label_and_probability(client) -> None:
    c, model = client
    r = _post_predict(c, {"features": _defaults(model)})
    assert r.status_code == 200
    body = r.get_json()
    assert body["prediction"]["label"] in {"Resistant", "Responsive", "Uncertain"}
    assert 0.0 <= body["prediction"]["probability_resistance"] <= 1.0


def test_predict_rejects_missing_feature(client) -> None:
    c, model = client
    payload = _defaults(model)
    del payload["sex"]
    r = _post_predict(c, {"features": payload})
    assert r.status_code == 400
    assert "sex" in r.get_json()["error"]


def test_predict_rejects_unknown_feature(client) -> None:
    c, model = client
    payload = _defaults(model)
    payload["definitely_not_a_feature"] = 1
    r = _post_predict(c, {"features": payload})
    assert r.status_code == 400
    assert "Unknown" in r.get_json()["error"]


def test_predict_rejects_out_of_range_numeric(client) -> None:
    c, model = client
    payload = _defaults(model)
    payload["age_at_first_rx"] = 999
    r = _post_predict(c, {"features": payload})
    assert r.status_code == 400
    assert "age_at_first_rx" in r.get_json()["error"]


def test_predict_rejects_invalid_categorical_value(client) -> None:
    c, model = client
    payload = _defaults(model)
    payload["sex"] = "Robot"
    r = _post_predict(c, {"features": payload})
    assert r.status_code == 400
    assert "sex" in r.get_json()["error"]
