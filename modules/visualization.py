# -*- coding: utf-8 -*-
"""Visualization: grafik Plotly."""
import plotly.graph_objects as go

C_LIN = "#e07b00"
C_CS = "#1f77b4"
C_TRUE = "#2a9d4b"
C_KNOT = "#d62728"


def main_fig(df_full, knots, test, tf, y_lin, y_cs, y_true, t_min, t_max, show_full=True):
    fig = go.Figure()
    if show_full:
        fig.add_trace(go.Scatter(x=df_full["time_ms"], y=df_full["amplitude"], mode="lines+markers",
                                 name="Data asli (101 titik)", line=dict(color="#bbbbbb", width=1),
                                 marker=dict(size=4)))
    if y_true is not None:
        fig.add_trace(go.Scatter(x=tf, y=y_true, mode="lines", name="Sinyal sebenarnya",
                                 line=dict(color=C_TRUE, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=tf, y=y_lin, mode="lines", name="Interpolasi linier",
                             line=dict(color=C_LIN, width=2)))
    fig.add_trace(go.Scatter(x=tf, y=y_cs, mode="lines", name="Cubic spline",
                             line=dict(color=C_CS, width=2)))
    if len(test):
        fig.add_trace(go.Scatter(x=test["time_ms"], y=test["amplitude"], mode="markers",
                                 name="Data uji (held-out)",
                                 marker=dict(color="#7f7f7f", symbol="x", size=8)))
    fig.add_trace(go.Scatter(x=knots["time_ms"], y=knots["amplitude"], mode="markers", name="Knot",
                             marker=dict(color=C_KNOT, size=9, line=dict(color="white", width=1))))
    fig.update_layout(xaxis_title="Waktu (ms)", yaxis_title="Amplitudo",
                      height=480, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.2), hovermode="x unified")
    if not show_full:
        pad = 0.02 * (t_max - t_min)
        fig.update_xaxes(range=[t_min - pad, t_max + pad])
    return fig


def error_fig(tf, err_lin, err_cs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tf, y=err_lin, mode="lines", name="Error linier", line=dict(color=C_LIN)))
    fig.add_trace(go.Scatter(x=tf, y=err_cs, mode="lines", name="Error cubic spline", line=dict(color=C_CS)))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_layout(xaxis_title="Waktu (ms)", yaxis_title="Error (interpolasi − sebenarnya)",
                      height=360, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.25), hovermode="x unified")
    return fig


def metric_bar(df_metrics):
    keys = ["RMSE", "MAE", "Error maks"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=keys, y=[df_metrics.loc[k, "Linier"] for k in keys], name="Linier", marker_color=C_LIN))
    fig.add_trace(go.Bar(x=keys, y=[df_metrics.loc[k, "Cubic Spline"] for k in keys], name="Cubic spline", marker_color=C_CS))
    fig.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Nilai error", yaxis_type="log",
                      legend=dict(orientation="h", y=-0.25))
    return fig
