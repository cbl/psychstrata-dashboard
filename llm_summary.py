from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from models import Contributor, Explanation, Prediction, TRModel

OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


def generate_prediction_summary(
    model: TRModel,
    raw: dict[str, Any],
    prediction: Prediction,
    explanation: Explanation,
) -> str:
    prompt = _build_prompt(model, raw, prediction, explanation)
    return _fetch_summary(prompt)


def _build_prompt(
    model: TRModel,
    raw: dict[str, Any],
    prediction: Prediction,
    explanation: Explanation,
) -> str:
    payload = {
        "task": "Summarise why the model predicted this resistance probability using only the supplied feature list and evidence notes.",
        "diagnosis": model.diagnosis,
        "prediction_label": prediction.label,
        "probability_resistance": round(float(prediction.probability or 0.0), 4),
        "factors_pushing_higher": [_contributor_payload(model, c) for c in explanation.top_positive],
        "factors_pushing_lower": [_contributor_payload(model, c) for c in explanation.top_negative],
        "rules": [
            "Use plain language and keep it concise.",
            "Only mention features in the payload.",
            "Treat SHAP direction as the source of truth for whether a feature pushed the prediction up or down.",
            "Only cite PMIDs that appear in evidence_pmids for the same feature.",
            "Do not invent papers, PMIDs, mechanisms, or clinical facts.",
            "If evidence is weak, mixed, or absent, say so.",
            "Do not give medical advice and do not claim causality.",
            "Return markdown: one opening sentence, 'Factors pushing higher' bullets, 'Factors pushing lower' bullets.",
        ],
    }
    return json.dumps(payload, indent=2)


def _contributor_payload(model: TRModel, c: Contributor) -> dict[str, Any]:
    evidence = model.evidence.get(c.feature_id)
    return {
        "feature": c.human_label,
        "selected_value": c.selected_value,
        "shap_value": round(float(c.shap_value), 4),
        "direction": "raises" if c.shap_value > 0 else "lowers",
        "evidence_association": evidence.association if evidence else "No supporting literature entry was provided for this feature.",
        "evidence_pmids": list(evidence.pmids) if evidence else [],
        "evidence_note": evidence.note if evidence and evidence.note else "",
    }


def _fetch_summary(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    if not api_key:
        return (
            "Prediction explanation unavailable: the language model service is not configured.\n\n"
            "The SHAP chart still shows which features pushed this prediction higher or lower."
        )

    payload = {
        "model": model_name,
        "input": [
            {
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "You explain model predictions for a synthetic treatment-resistance demo. "
                        "Base every statement only on the provided JSON payload. "
                        "Use plain language, cite only supplied PMIDs, and avoid unsupported claims."
                    ),
                }],
            },
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        "temperature": 0.2,
        "max_output_tokens": 350,
    }

    req = request.Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return f"Prediction explanation unavailable: HTTP {exc.code}.\n\n```text\n{detail[:800]}\n```"
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        return f"Prediction explanation unavailable: could not reach service ({reason})."
    except TimeoutError:
        return "Prediction explanation unavailable: service timed out."
    except json.JSONDecodeError:
        return "Prediction explanation unavailable: response could not be parsed."

    text = _extract_text(response_payload)
    return text or "Prediction explanation unavailable: empty response."


def _extract_text(response_payload: dict[str, Any]) -> str:
    if response_payload.get("output_text"):
        return str(response_payload["output_text"]).strip()
    parts: list[str] = []
    for output_item in response_payload.get("output", []):
        for content_item in output_item.get("content", []):
            text = content_item.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()
