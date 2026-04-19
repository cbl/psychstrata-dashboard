from __future__ import annotations

from dash import dcc, html

from config import CARD, INFO_TEXT_STYLE, SECTION_TITLE, SUBHEADING
from models import FeatureSpec


def create_feature_input(spec: FeatureSpec) -> html.Div:
    label = html.Label(
        spec.label,
        htmlFor=f"input-{spec.id}",
        style={"fontWeight": "600", "fontSize": "12px", "color": "#374151"},
    )
    help_text = (
        html.Div(spec.description, style={"fontSize": "11px", "color": "#6b7280", "marginTop": "2px"})
        if spec.description else None
    )
    control = _control_for(spec)
    return html.Div(
        [label, control] + ([help_text] if help_text else []),
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
    )


def _control_for(spec: FeatureSpec):
    input_id = f"input-{spec.id}"
    if spec.kind == "numeric":
        return dcc.Input(
            id=input_id,
            type="number",
            min=spec.min,
            max=spec.max,
            step=spec.step,
            value=spec.default,
            debounce=True,
            style={
                "padding": "6px 10px",
                "border": "1px solid #d1d5db",
                "borderRadius": "6px",
                "fontSize": "14px",
                "width": "100%",
            },
        )
    options = [{"label": opt.label, "value": opt.value} for opt in (spec.options or ())]
    return dcc.Dropdown(
        id=input_id,
        options=options,
        value=spec.default,
        clearable=False,
        style={"fontSize": "14px"},
    )


def create_feature_section(category: str, specs: list[FeatureSpec], open_by_default: bool) -> html.Details:
    grid = html.Div(
        [create_feature_input(s) for s in specs],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "12px",
            "marginTop": "10px",
        },
    )
    summary = html.Summary(
        [
            html.Span(category, style={"fontWeight": "600"}),
            html.Span(f"  ({len(specs)})", style={"color": "#6b7280", "fontSize": "12px"}),
        ],
        style={"cursor": "pointer", "padding": "8px 0", "fontSize": "14px"},
    )
    return html.Details(
        [summary, grid],
        open=open_by_default,
        style={"borderBottom": "1px solid #eee", "padding": "4px 0"},
    )


def create_feature_form(features: list[FeatureSpec], category_order: list[str]) -> html.Div:
    by_cat: dict[str, list[FeatureSpec]] = {}
    for f in features:
        by_cat.setdefault(f.category, []).append(f)
    ordered = [c for c in category_order if c in by_cat] + [
        c for c in by_cat if c not in category_order
    ]
    return html.Div(
        [create_feature_section(c, by_cat[c], open_by_default=(i == 0)) for i, c in enumerate(ordered)],
        style={"display": "flex", "flexDirection": "column"},
    )


def create_info_details(summary: str, content: str, style_override: dict | None = None) -> html.Details:
    base_style = {"marginTop": "8px"}
    if style_override:
        base_style.update(style_override)
    return html.Details(
        [html.Summary(summary), html.Div(content, style=INFO_TEXT_STYLE)],
        open=False,
        style=base_style,
    )


def create_header_card(auc: float, diagnosis: str, model_name: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H2(f"Treatment Resistance Classifier — {diagnosis}", style={"margin": 0}),
                    html.Div(
                        f"Demo • {model_name} • Not medical advice",
                        style={"color": "#6b7280", "fontSize": "12px", "marginTop": "2px"},
                    ),
                    html.Div(
                        "This demo uses fully synthetic data created for illustration. "
                        "It does not reflect actual patient information. "
                        "It is not a medical device and must not be used for clinical decisions.",
                        style={
                            "color": "#6b7280", "fontSize": "12px", "marginTop": "6px",
                            "maxWidth": "820px", "lineHeight": "1.4",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "column"},
            ),
            html.Div(
                [
                    html.Span("Model AUC", style={"marginRight": "8px", "color": "#6b7280", "fontSize": "12px"}),
                    html.Span(
                        f"{auc:.3f}",
                        style={
                            "display": "inline-block", "padding": "4px 10px", "borderRadius": "999px",
                            "backgroundColor": "#eef2ff", "color": "#3730a3",
                            "fontSize": "12px", "fontWeight": "600",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "4px"},
            ),
        ],
        style={**CARD, "display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"},
    )


def create_prediction_card() -> html.Div:
    return html.Div(
        [
            html.H4("Prediction", style=SECTION_TITLE),
            html.Div("Probability of treatment resistance", style=SUBHEADING),
            dcc.Graph(id="pred-indicator", config={"displayModeBar": False},
                      style={"height": "200px", "marginTop": "4px"}),
            html.Div(
                [
                    html.Div("Conformal prediction", style={**SUBHEADING, "marginBottom": "6px"}),
                    html.Div("Confidence level (%)", style={"fontSize": "12px", "color": "#444", "marginBottom": "4px"}),
                    dcc.Slider(id="ci-slider", min=80, max=99, step=1, value=95,
                               marks={x: str(x) for x in [80, 85, 90, 95, 99]}),
                    html.Div([html.Div(id="prediction-badge", children="")],
                             style={"display": "flex", "justifyContent": "center", "marginTop": "10px"}),
                ],
                style={"marginTop": "10px", "padding": "10px", "backgroundColor": "#f9fafb",
                       "borderRadius": "8px", "border": "1px solid #eee"},
            ),
            create_info_details(
                "What's this?",
                "The gauge shows the model's estimated resistance probability. "
                "The badge reports a calibrated-uncertainty label at the chosen confidence level.",
            ),
        ],
        style=CARD,
    )


def create_cif_card() -> html.Div:
    return html.Div(
        [
            html.H4("Cumulative incidence (CIF)", style={**SECTION_TITLE, "textAlign": "center"}),
            dcc.Graph(id="cif-curve", config={"displayModeBar": False},
                      style={"height": "360px", "margin": "6px auto", "width": "95%"}),
            create_info_details(
                "What's this?",
                "For this patient, the model's predicted cumulative probability of each outcome over time. "
                "Treatment resistance is the primary outcome; death and discontinuation are competing risks. "
                "The dashed green line is the remaining event-free probability (= 1 − sum of the others). "
                "The dot marks the 5-year horizon on the TR curve.",
                {"width": "95%", "marginLeft": "auto", "marginRight": "auto"},
            ),
        ],
        style=CARD,
    )


def create_shap_card() -> html.Div:
    return html.Div(
        [
            html.H4("Top feature contributions (SHAP)", style={**SECTION_TITLE, "textAlign": "center"}),
            dcc.Graph(id="shap-bar", config={"displayModeBar": False},
                      style={"height": "440px", "margin": "6px auto", "width": "95%"}),
            create_info_details(
                "What's this?",
                "Each bar shows how a feature pushed this prediction. Red raises resistance risk; green lowers it. "
                "Only the top-contributing features are shown.",
                {"width": "95%", "marginLeft": "auto", "marginRight": "auto"},
            ),
        ],
        style=CARD,
    )


def create_tsne_card() -> html.Div:
    return html.Div(
        [
            html.H4("Population map (t-SNE)", style={**SECTION_TITLE, "textAlign": "center"}),
            dcc.Graph(id="tsne-scatter", config={"displayModeBar": False},
                      style={"height": "380px", "margin": "6px auto", "width": "95%"}),
            create_info_details(
                "What's this?",
                "Similar patients sit near each other. Green = responsive, red = resistant. "
                "The blue dot marks the current selection.",
                {"width": "95%", "marginLeft": "auto", "marginRight": "auto"},
            ),
        ],
        style=CARD,
    )


def create_llm_summary_card() -> html.Div:
    return html.Div(
        [
            html.H4("Prediction explanation", style=SECTION_TITLE),
            html.Div(
                [
                    html.Button(
                        "Explain this to me",
                        id="llm-explain-button",
                        n_clicks=0,
                        style={
                            "padding": "8px 14px", "borderRadius": "8px",
                            "border": "1px solid #2563eb", "backgroundColor": "#2563eb",
                            "color": "white", "fontWeight": "600", "cursor": "pointer",
                        },
                    ),
                    html.Div(
                        "Change any feature, then click to refresh the explanation.",
                        id="llm-summary-status",
                        style={"fontSize": "12px", "color": "#6b7280"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "10px"},
            ),
            dcc.Loading(
                html.Div(
                    dcc.Markdown(
                        id="llm-summary",
                        children="Click **Explain this to me** to generate a plain-language summary.",
                        style={
                            "backgroundColor": "#f9fafb", "border": "1px solid #eee",
                            "borderRadius": "8px", "padding": "12px",
                            "color": "#374151", "fontSize": "14px", "lineHeight": "1.5",
                            "minHeight": "180px",
                        },
                    )
                ),
                type="dot",
            ),
            create_info_details(
                "What's this?",
                "A plain-language walk-through of which features pushed this prediction up or down.",
            ),
        ],
        style={**CARD, "flex": 1},
    )
