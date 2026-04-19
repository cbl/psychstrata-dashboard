from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, request

from models import FeatureSpec, TRModel


def register_api(server: Flask, model: TRModel) -> None:
    features_by_id = {f.id: f for f in model.features}

    @server.get("/api/health")
    def api_health():
        return jsonify({"status": "ok", "model": model.name, "diagnosis": model.diagnosis})

    @server.get("/api/features")
    def api_features():
        return jsonify(
            {
                "model": model.name,
                "diagnosis": model.diagnosis,
                "features": [_feature_schema(f) for f in model.features],
                "model_feature_names": model.encoder.model_feature_names,
            }
        )

    @server.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        features_payload = payload.get("features", payload)
        try:
            raw = _validate(features_payload, model.features, features_by_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        prediction = model.predict(raw)
        explanation = model.explain(raw)
        return jsonify(
            {
                "features": raw,
                "prediction": {
                    "label": prediction.label,
                    "probability_resistance": _round(prediction.probability),
                    "conformal_set": prediction.conformal_set,
                },
                "shap_values": {k: _round(v) for k, v in explanation.shap_values.items()},
                "top_contributors": {
                    "positive": [_contributor_dict(c) for c in explanation.top_positive],
                    "negative": [_contributor_dict(c) for c in explanation.top_negative],
                },
            }
        )


def _feature_schema(f: FeatureSpec) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "id": f.id, "label": f.label, "category": f.category,
        "kind": f.kind, "default": f.default,
    }
    if f.kind == "numeric":
        schema.update({"min": f.min, "max": f.max, "step": f.step})
    if f.options:
        schema["options"] = [{"label": o.label, "value": o.value} for o in f.options]
    if f.description:
        schema["description"] = f.description
    return schema


def _validate(
    payload: dict[str, Any],
    features: list[FeatureSpec],
    features_by_id: dict[str, FeatureSpec],
) -> dict[str, Any]:
    missing = [f.id for f in features if f.id not in payload]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}.")
    unknown = sorted(set(payload) - set(features_by_id))
    if unknown:
        raise ValueError(f"Unknown features: {', '.join(unknown)}.")
    return {f.id: _coerce(f, payload[f.id]) for f in features}


def _coerce(spec: FeatureSpec, raw: Any) -> Any:
    if spec.kind == "numeric":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Feature '{spec.id}' must be numeric.")
        value = float(raw)
        if spec.min is not None and value < spec.min:
            raise ValueError(f"Feature '{spec.id}' must be ≥ {spec.min}.")
        if spec.max is not None and value > spec.max:
            raise ValueError(f"Feature '{spec.id}' must be ≤ {spec.max}.")
        return value
    valid = {opt.value for opt in (spec.options or ())}
    if raw not in valid:
        raise ValueError(f"Feature '{spec.id}' must be one of {sorted(map(str, valid))}.")
    return raw


def _contributor_dict(c) -> dict[str, Any]:
    return {
        "feature": c.feature_id,
        "label": c.human_label,
        "selected_value": c.selected_value,
        "shap_value": _round(c.shap_value),
    }


def _round(x: float | None, digits: int = 6) -> float | None:
    return None if x is None else round(float(x), digits)
