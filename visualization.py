from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from models import Explanation, FeatureEncoder

CAUSE_COLORS: dict[str, str] = {
    "Treatment resistance": "#C44E52",
    "Death": "#4C72B0",
    "Discontinuation": "#CCB974",
    "Event-free": "#55A868",
}


def create_indicator_figure(prob: float) -> go.Figure:
    color = "#d62728" if prob >= 0.5 else "#2ca02c"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": "Probability of resistance"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#e6f4ea"},
                    {"range": [50, 100], "color": "#fdecea"},
                ],
                "threshold": {"line": {"color": "#444", "width": 2}, "thickness": 0.75, "value": 50},
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=30))
    return fig


def create_shap_bar_figure(explanation: Explanation, encoder: FeatureEncoder, top_n: int = 15) -> go.Figure:
    if not explanation.shap_values:
        fig = go.Figure()
        fig.update_layout(
            title={"text": "SHAP attributions unavailable for this model", "x": 0.5},
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{
                "text": "No SHAP available", "x": 0.5, "y": 0.5,
                "showarrow": False, "font": {"size": 14, "color": "#6b7280"},
                "xref": "paper", "yref": "paper",
            }],
            plot_bgcolor="white", margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig
    ranked = sorted(explanation.shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    ranked.sort(key=lambda kv: abs(kv[1]))
    labels = [encoder.inverse_feature_name(col) for col, _ in ranked]
    values = [v for _, v in ranked]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    fig = go.Figure(
        data=go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": f"Top {len(ranked)} contributions to P(Resistance)", "x": 0.5},
        xaxis_title="SHAP value (Δ probability)",
        margin=dict(l=180, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(ranked) - 0.5,
                  line=dict(color="#9ca3af", width=1))
    return fig


def create_cif_figure(
    cause_curves: dict[str, dict[float, float]] | None,
    horizon_years: float | None = None,
    primary_cause: str = "Treatment resistance",
) -> go.Figure:
    if not cause_curves:
        fig = go.Figure()
        fig.update_layout(
            title={"text": "Cumulative incidence unavailable for this model", "x": 0.5},
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{
                "text": "No CIF curves available", "x": 0.5, "y": 0.5,
                "showarrow": False, "font": {"size": 14, "color": "#6b7280"},
                "xref": "paper", "yref": "paper",
            }],
            plot_bgcolor="white", margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    fig = go.Figure()
    for cause_name, curve in cause_curves.items():
        times = list(curve.keys())
        values = list(curve.values())
        fig.add_trace(
            go.Scatter(
                x=times, y=values, mode="lines",
                name=cause_name,
                line=dict(color=CAUSE_COLORS.get(cause_name, "#888888"), width=2.5),
                hovertemplate="%{x:.2f}y: %{y:.3f}<extra>" + cause_name + "</extra>",
            )
        )

    event_free = _event_free_curve(cause_curves)
    if event_free is not None:
        times, values = event_free
        fig.add_trace(
            go.Scatter(
                x=times, y=values, mode="lines",
                name="Event-free",
                line=dict(color=CAUSE_COLORS["Event-free"], width=2.5, dash="dash"),
                hovertemplate="%{x:.2f}y: %{y:.3f}<extra>Event-free</extra>",
            )
        )

    if horizon_years is not None and primary_cause in cause_curves:
        curve = cause_curves[primary_cause]
        times = np.array(list(curve.keys()))
        values = np.array(list(curve.values()))
        idx = int(np.argmin(np.abs(times - horizon_years)))
        fig.add_trace(
            go.Scatter(
                x=[times[idx]], y=[values[idx]], mode="markers",
                name=f"P@{horizon_years:g}y",
                marker=dict(color=CAUSE_COLORS.get(primary_cause, "#C44E52"),
                            size=12, line=dict(color="white", width=2)),
                hovertemplate=f"P({primary_cause} at {horizon_years:g}y) = %{{y:.3f}}<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_shape(
            type="line", x0=horizon_years, x1=horizon_years, y0=0, y1=1,
            line=dict(color="#9ca3af", width=1, dash="dot"),
        )

    fig.update_layout(
        xaxis_title="Years since first prescription",
        yaxis_title="Cumulative incidence",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor="white",
    )
    return fig


def _event_free_curve(
    cause_curves: dict[str, dict[float, float]],
) -> tuple[list[float], list[float]] | None:
    if not cause_curves:
        return None
    first_curve = next(iter(cause_curves.values()))
    times = sorted(first_curve)
    if not all(sorted(c) == times for c in cause_curves.values()):
        return None
    values = [
        max(0.0, 1.0 - sum(c[t] for c in cause_curves.values()))
        for t in times
    ]
    return times, values


def create_tsne_scatter_figure(
    embedding: np.ndarray, y_labels: np.ndarray, sel_x: float, sel_y: float,
) -> go.Figure:
    if embedding.size == 0:
        fig = go.Figure()
        fig.update_layout(
            title={"text": "Population map unavailable for this model", "x": 0.5},
            xaxis={"visible": False}, yaxis={"visible": False},
            annotations=[{
                "text": "No embedding available", "x": 0.5, "y": 0.5,
                "showarrow": False, "font": {"size": 14, "color": "#6b7280"},
                "xref": "paper", "yref": "paper",
            }],
            plot_bgcolor="white", margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig
    resistant = y_labels == 1
    responsive = y_labels == 0

    trace_resp = go.Scattergl(
        x=embedding[responsive, 0], y=embedding[responsive, 1],
        mode="markers", name="Responsive",
        marker=dict(color="#2ca02c", size=6, opacity=0.75),
        hovertemplate="Responsive<extra></extra>",
    )
    trace_resi = go.Scattergl(
        x=embedding[resistant, 0], y=embedding[resistant, 1],
        mode="markers", name="Resistant",
        marker=dict(color="#d62728", size=6, opacity=0.75),
        hovertemplate="Resistant<extra></extra>",
    )
    trace_sel = go.Scattergl(
        x=[sel_x], y=[sel_y],
        mode="markers", name="Current selection",
        marker=dict(color="#1f77b4", size=12, line=dict(color="white", width=1.5)),
        hovertemplate="Current selection<extra></extra>",
    )

    fig = go.Figure(data=[trace_resp, trace_resi, trace_sel])
    fig.update_layout(
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="white",
    )
    return fig
