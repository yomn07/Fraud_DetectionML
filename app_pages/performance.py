import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.artifacts import MODEL_LABELS, require_artifacts
from core.metrics import calibration, gain_curve, precision_recall_curve, roc_curve, summary
from core.ui import ALERT_COLOR, PALETTE, style

art = require_artifacts()
preds = art.preds
y = preds["y_true"].to_numpy()

st.title("Model performance")
st.caption("Everything below is computed from holdout predictions (the most recent 15% of transactions).")

default = [m for m in ("p_baseline", "p_v2") if m in art.models] or art.models[:1]
chosen = st.multiselect("Models to compare", art.models, default=default, format_func=MODEL_LABELS.get)
if not chosen:
    st.stop()
colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(art.models)}

tab_roc, tab_pr, tab_gain, tab_dist, tab_cal, tab_time = st.tabs(
    ["ROC", "Precision-Recall", "Gain (alert budget)", "Score distribution", "Calibration", "Over time"])

with tab_roc:
    fig = go.Figure()
    for m in chosen:
        fpr, tpr, _ = roc_curve(y, preds[m])
        step = max(1, len(fpr) // 600)
        fig.add_scatter(x=fpr[::step], y=tpr[::step], mode="lines", name=MODEL_LABELS[m], line_color=colors[m])
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dot", color="#999"))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(style(fig, 420), width="stretch")

with tab_pr:
    fig = go.Figure()
    for m in chosen:
        prec, rec, _ = precision_recall_curve(y, preds[m])
        step = max(1, len(prec) // 600)
        fig.add_scatter(x=rec[::step], y=prec[::step], mode="lines", name=MODEL_LABELS[m], line_color=colors[m])
    fig.add_hline(y=y.mean(), line_dash="dot", line_color="#999", annotation_text="No-skill = fraud rate")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1.02])
    st.plotly_chart(style(fig, 420), width="stretch")
    st.caption("With a few percent fraud, the PR curve is the more honest picture than ROC.")

with tab_gain:
    fig = go.Figure()
    for m in chosen:
        x, g = gain_curve(y, preds[m].to_numpy())
        fig.add_scatter(x=100 * x, y=100 * g, mode="lines", name=MODEL_LABELS[m], line_color=colors[m])
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random review", line=dict(dash="dot", color="#999"))
    fig.update_layout(xaxis_title="% of transactions flagged (highest score first)", yaxis_title="% of fraud caught",
                      xaxis_range=[0, 30])
    st.plotly_chart(style(fig, 420), width="stretch")
    budget = pd.DataFrame({"Alert budget": ["1%", "2%", "5%", "10%"]})
    for m in chosen:
        order = np.argsort(-preds[m].to_numpy())
        vals = []
        for pct_ in (1, 2, 5, 10):
            k = int(np.ceil(pct_ / 100 * len(y)))
            vals.append(y[order[:k]].sum() / max(1, y.sum()))
        budget[MODEL_LABELS[m]] = vals
    st.dataframe(budget, hide_index=True, width="stretch",
                 column_config={MODEL_LABELS[m]: st.column_config.ProgressColumn(
                     format="percent", min_value=0, max_value=1) for m in chosen})

with tab_dist:
    m = st.selectbox("Model", chosen, format_func=MODEL_LABELS.get, key="dist_model")
    p = preds[m].to_numpy()
    bins = np.linspace(0, 1, 41)
    fig = go.Figure()
    fig.add_histogram(x=p[y == 0], xbins=dict(start=0, end=1, size=0.025), name="Legitimate", marker_color=PALETTE[0], opacity=0.75)
    fig.add_histogram(x=p[y == 1], xbins=dict(start=0, end=1, size=0.025), name="Fraud", marker_color=ALERT_COLOR, opacity=0.75)
    fig.update_layout(barmode="overlay", xaxis_title="Predicted fraud probability", yaxis_title="Transactions (log scale)", yaxis_type="log")
    st.plotly_chart(style(fig, 400), width="stretch")

with tab_cal:
    m = st.selectbox("Model", chosen, format_func=MODEL_LABELS.get, key="cal_model")
    cal = calibration(y, preds[m].to_numpy(), bins=10)
    mx = float(max(cal["pred"].max(), cal["obs"].max())) * 1.1
    fig = go.Figure()
    fig.add_scatter(x=cal["pred"], y=cal["obs"], mode="lines+markers", name="Observed", line_color=PALETTE[0])
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect", line=dict(dash="dot", color="#999"))
    fig.update_layout(xaxis_title="Mean predicted probability (score decile)", yaxis_title="Observed fraud rate")
    st.plotly_chart(style(fig, 400), width="stretch")
    st.caption("Model scores are good for *ranking*; if they drift from the diagonal, rely on thresholds chosen on "
               "validation data rather than treating the score as a literal probability.")

with tab_time:
    m = st.selectbox("Model", chosen, format_func=MODEL_LABELS.get, key="time_model")
    d = preds.assign(score=preds[m]).groupby("day").agg(fraud_rate=("y_true", "mean"), mean_score=("score", "mean"),
                                                         n=("y_true", "size")).reset_index()
    fig = go.Figure()
    fig.add_scatter(x=d["day"], y=d["fraud_rate"], name="Actual fraud rate", line_color=ALERT_COLOR)
    fig.add_scatter(x=d["day"], y=d["mean_score"], name="Mean predicted score", line_color=PALETTE[0])
    fig.update_layout(xaxis_title="Day (relative to dataset start)", yaxis_title="Rate")
    st.plotly_chart(style(fig, 380), width="stretch")
    st.caption("Large gaps would signal drift between what the model learned and recent behaviour.")

st.subheader("Metrics")
rows = [{"Model": MODEL_LABELS[m], **summary(y, preds[m].to_numpy())} for m in chosen]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ("AUC", "AP", "Recall @1%", "Recall @5%")})
