from __future__ import annotations

from typing import Any

from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

from config import PILL
from llm_summary import generate_prediction_summary
from models import Prediction, TRModel
from visualization import (
    create_indicator_figure,
    create_shap_bar_figure,
    create_tsne_scatter_figure,
)


BADGE_STYLES: dict[str, dict[str, str]] = {
    "Resistant": {"bg": "#fee2e2", "border": "1px solid #fca5a5", "color": "#991b1b"},
    "Responsive": {"bg": "#dcfce7", "border": "1px solid #86efac", "color": "#14532d"},
    "Uncertain": {"bg": "#fef3c7", "border": "1px solid #fcd34d", "color": "#92400e"},
}


def register_callbacks(app: Dash, model: TRModel) -> None:
    feature_ids = [f.id for f in model.features]
    input_ids = [f"input-{fid}" for fid in feature_ids]

    @app.callback(
        [
            Output("pred-indicator", "figure"),
            Output("prediction-badge", "children"),
            Output("prediction-badge", "style"),
            Output("shap-bar", "figure"),
            Output("tsne-scatter", "figure"),
        ],
        [Input(cid, "value") for cid in input_ids] + [Input("ci-slider", "value")],
    )
    def update_predictions(*args):
        *values, _ci_level = args
        raw = _build_raw(feature_ids, values)
        prediction = model.predict(raw)
        explanation = model.explain(raw)

        indicator = create_indicator_figure(prediction.probability or 0.0)
        badge_label = prediction.label
        badge_style = _badge_style(badge_label)

        shap_fig = create_shap_bar_figure(explanation, model.encoder, top_n=15)
        sel_x, sel_y = model.tsne_position(raw)
        tsne_fig = create_tsne_scatter_figure(model.tsne_embedding, model.y_labels, sel_x, sel_y)

        return indicator, badge_label, badge_style, shap_fig, tsne_fig

    @app.callback(
        [Output("llm-summary", "children"), Output("llm-summary-status", "children")],
        Input("llm-explain-button", "n_clicks"),
        [State(cid, "value") for cid in input_ids],
        prevent_initial_call=True,
        running=[(Output("llm-explain-button", "disabled"), True, False)],
    )
    def update_llm_summary(n_clicks: int, *values):
        if not n_clicks:
            raise PreventUpdate
        raw = _build_raw(feature_ids, values)
        prediction = model.predict(raw)
        explanation = model.explain(raw)
        summary = generate_prediction_summary(model, raw, prediction, explanation)
        return summary, "Explanation generated for the current selection."


def _build_raw(feature_ids: list[str], values) -> dict[str, Any]:
    return {fid: v for fid, v in zip(feature_ids, values)}


def _badge_style(label: str) -> dict[str, str]:
    colors = BADGE_STYLES.get(label, BADGE_STYLES["Uncertain"])
    return {**PILL, "backgroundColor": colors["bg"], "border": colors["border"], "color": colors["color"]}
