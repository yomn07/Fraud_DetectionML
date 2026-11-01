import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.artifacts import MODEL_LABELS, require_artifacts
from core.metrics import best_threshold, confusion, cost_curve
from core.ui import ALERT_COLOR, PALETTE, pct, style

art = require_artifacts()
preds, meta = art.preds, art.meta
y = preds["y_true"].to_numpy()

st.title("Threshold & cost simulator")
st.caption("Turn scores into decisions. Change the costs and see what each rule means for alerts, missed fraud and money.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", art.models, index=art.models.index("p_v2") if "p_v2" in art.models else 0,
                         format_func=MODEL_LABELS.get)
    st.subheader("Costs")
    c_fp = st.number_input("Cost of a false alarm", min_value=0.0, value=float(meta["cost"]["fp"]), step=1.0,
                           help="Review time or customer friction for one legitimate transaction that gets flagged.")
    c_fn = st.number_input("Cost of a missed fraud", min_value=0.0, value=float(meta["cost"]["fn"]), step=10.0,
                           help="Average loss when a fraudulent transaction goes through.")
    cap = st.slider("Maximum alert rate", 0.5, 20.0, float(100 * meta["cost"]["max_alert_rate"]), 0.5,
                    format="%.1f%%") / 100

p = preds[model].to_numpy()
thr_meta = meta["thresholds"]
valid_ok = model == "p_v2"

modes = []
if valid_ok:
    modes += ["F2-optimal (chosen on validation)", "Cost-optimal (chosen on validation)",
              "Alert-capped cost-optimal (chosen on validation)"]
modes += ["Re-optimise cost on holdout (optimistic)", "Alert budget (top X%)", "Custom threshold"]
mode = st.radio("Decision rule", modes, horizontal=False)

if mode.startswith("F2"):
    thr = thr_meta["f2"]
elif mode.startswith("Cost-optimal"):
    thr = thr_meta["cost"] if thr_meta["cost"] is not None else float("inf")
elif mode.startswith("Alert-capped"):
    thr = thr_meta["capped"]
elif mode.startswith("Re-optimise"):
    thr = best_threshold(y, p, c_fp, c_fn, cap)
elif mode.startswith("Alert budget"):
    budget = st.slider("Flag the top X% of transactions", 0.1, 20.0, 2.0, 0.1, format="%.1f%%")
    thr = float(np.quantile(p, 1 - budget / 100))
else:
    thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.005)

if valid_ok and mode.split(" (")[0] in ("F2-optimal", "Cost-optimal", "Alert-capped cost-optimal"):
    st.caption(f"Threshold {thr:.5f} was chosen on validation data with costs {meta['cost']['fp']}:{meta['cost']['fn']} "
               f"(cap {pct(meta['cost']['max_alert_rate'], 0)}). Costs on the left change the **evaluation**, not this threshold.")

if np.isinf(thr):
    st.warning("At these costs, flagging nothing is cheapest, so no transactions are flagged.")
cm = confusion(y, p, thr)
no_model = c_fn * int(y.sum())
total = c_fp * cm["FP"] + c_fn * cm["FN"]
n = len(y)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Alerts", f"{cm['alerts']:,}", f"{pct(cm['alerts'] / n, 2)} of transactions", delta_color="off")
k2.metric("Fraud caught (recall)", pct(cm["recall"]), f"{cm['TP']:,} of {int(y.sum()):,}", delta_color="off")
k3.metric("Precision of alerts", pct(cm["precision"]), f"1 in {1 / max(cm['precision'], 1e-9):.1f} alerts is fraud", delta_color="off")
k4.metric("Total cost", f"{total:,.0f}", f"{100 * (1 - total / max(no_model, 1)):.1f}% saved vs. flagging nothing")

left, right = st.columns([3, 2])
with left:
    cc = cost_curve(y, p, c_fp, c_fn)
    fig = go.Figure()
    fig.add_scatter(x=100 * cc["alert_rate"], y=cc["cost"], mode="lines", name="Total cost", line_color=PALETTE[0])
    fig.add_vline(x=100 * cm["alerts"] / n, line_dash="dash", line_color=ALERT_COLOR, annotation_text="chosen rule")
    fig.add_hline(y=no_model, line_dash="dot", line_color="#999", annotation_text="flag nothing")
    fig.update_layout(title="Total cost vs. share of transactions flagged",
                      xaxis_title="% flagged", yaxis_title="Total cost", xaxis_range=[0, min(30, max(5, 100 * cap * 2))])
    st.plotly_chart(style(fig, 360, legend=False), width="stretch")
with right:
    z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    fig = go.Figure(go.Heatmap(z=np.log10(np.array(z) + 1), text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}",
                               x=["Not flagged", "Flagged"], y=["Legitimate", "Fraud"], colorscale="Blues", showscale=False))
    fig.update_layout(title="Confusion matrix (holdout)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(style(fig, 360, legend=False), width="stretch")

st.subheader("Scaled to 100,000 transactions")
f = 100_000 / n
st.dataframe(pd.DataFrame([{
    "Alerts to review": round(cm["alerts"] * f), "Fraud caught": round(cm["TP"] * f), "Fraud missed": round(cm["FN"] * f),
    "False alarms": round(cm["FP"] * f), "F2": round(cm["F2"], 3)}]), hide_index=True, width="stretch")

st.subheader("How sensitive is the best threshold to the cost ratio?")
rows = []
for ratio in (5, 10, 50, 100, 300, 1000):
    t = best_threshold(y, p, 1, ratio, cap)
    c = confusion(y, p, t)
    rows.append({"Missed fraud costs ×": ratio, "Threshold": t, "Alert rate": c["alerts"] / n,
                 "Recall": c["recall"], "Precision": c["precision"], "Total cost (per 1 false alarm)": c["FP"] + ratio * c["FN"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
             column_config={"Threshold": st.column_config.NumberColumn(format="%.4f"),
                            "Alert rate": st.column_config.NumberColumn(format="percent"),
                            "Recall": st.column_config.NumberColumn(format="percent"),
                            "Precision": st.column_config.NumberColumn(format="percent"),
                            "Total cost (per 1 false alarm)": st.column_config.NumberColumn(format="%d")})
st.caption("Thresholds here are optimised on the holdout itself, so they are slightly optimistic. "
           "The validation-chosen rules above are the honest estimate.")
