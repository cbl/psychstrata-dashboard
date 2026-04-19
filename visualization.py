from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from models import Explanation, FeatureEncoder


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


def create_tsne_scatter_figure(
    embedding: np.ndarray, y_labels: np.ndarray, sel_x: float, sel_y: float,
) -> go.Figure:
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
