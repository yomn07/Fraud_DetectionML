import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from core.artifacts import MODEL_LABELS, require_artifacts
from core.metrics import (amount_cost, amount_cost_curve, best_threshold, best_threshold_amount, confusion, cost_curve,
                          precision_recall_at_k)
from core.ui import ALERT_COLOR, PALETTE, pct, style

art = require_artifacts()
test, valid, meta = art.preds, art.valid, art.meta

st.title("Threshold, alert budget & business cost")
st.caption("The threshold is chosen on the **validation** period by minimising a business cost, then **confirmed on the test period** "
           "(the most recent, untouched data). Move any control and every number updates live.")

has_amount = "amount" in test.columns and valid is not None and "amount" in valid.columns
usable = [m for m in art.models if valid is not None and m in valid.columns]
if valid is None or not usable:
    st.warning("This artifact set has no validation predictions, so thresholds are chosen on the test period "
               "(optimistic). Re-export with the latest section 18 for the full workflow.")
    usable = art.models
    valid = test

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", usable, index=usable.index("p_v2") if "p_v2" in usable else 0, format_func=MODEL_LABELS.get)
    st.subheader("Business cost")
    if has_amount:
        basis = st.radio("Cost model", ["Amounts", "Fixed ratio"], horizontal=True,
                         help="Amounts: a missed fraud costs its transaction amount. Fixed ratio: every missed fraud costs the same.")
    else:
        basis = "Fixed ratio"
        st.info("No transaction amounts in the artifacts: using a fixed ratio.")
    if basis == "Amounts":
        review = st.number_input("Review cost per alert", min_value=0.0, value=5.0, step=1.0,
                                 help="What it costs an analyst to look at one alert (same currency as TransactionAmt).")
    else:
        c_fp = st.number_input("Cost of a false alarm", min_value=0.0, value=float(meta["cost"]["fp"]), step=1.0)
        c_fn = st.number_input("Cost of a missed fraud", min_value=0.0, value=float(meta["cost"]["fn"]), step=10.0)
    st.subheader("Analyst capacity")
    days = max(1, int(test["day"].nunique())) if "day" in test.columns else 1
    per_day = len(test) / days
    capacity = st.number_input("Alerts an analyst team can review per day", 1, 100000, max(1, int(0.02 * per_day)), 10)
    k_share = min(1.0, capacity / per_day)
    st.caption(f"Test period: about {per_day:,.0f} transactions/day, so the budget is **{pct(k_share, 2)}** of transactions.")

yv, pv = valid["y_true"].to_numpy(), valid[model].to_numpy()
yt, pt = test["y_true"].to_numpy(), test[model].to_numpy()
av = valid["amount"].to_numpy() if has_amount else None
at = test["amount"].to_numpy() if has_amount else None


def opt_threshold(max_rate=1.0):
    if basis == "Amounts":
        return best_threshold_amount(yv, pv, av, review, max_rate)
    return best_threshold(yv, pv, c_fp, c_fn, max_rate)


modes = ["Cost-optimal (chosen on validation)", "Cost-optimal within my alert budget (validation)",
         "Alert budget only (top k)", "Custom threshold"]
mode = st.radio("Decision rule", modes, horizontal=True)
k_test = max(1, int(np.ceil(k_share * len(pt))))
if mode == modes[0]:
    thr = opt_threshold()
elif mode == modes[1]:
    thr = opt_threshold(k_share)
elif mode == modes[2]:
    thr = float(np.sort(pt)[::-1][k_test - 1])
else:
    thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.005)

if np.isinf(thr):
    st.warning("At these costs, flagging nothing is cheapest on validation, so nothing is flagged. Lower the review cost.")


def cost_of(y, p, a, t):
    if basis == "Amounts":
        c = amount_cost(y, p, a, t, review)
        return c["total"], c
    cm_ = confusion(y, p, t)
    return c_fp * cm_["FP"] + c_fn * cm_["FN"], cm_


tot_v, _ = cost_of(yv, pv, av, thr)
tot_t, det_t = cost_of(yt, pt, at, thr)
if basis == "Amounts":
    nothing_t, everything_t = float(at[yt == 1].sum()), review * len(yt)
else:
    nothing_t, everything_t = c_fn * int(yt.sum()), c_fp * int((yt == 0).sum())
cm = confusion(yt, pt, thr)
sym = "" if basis == "Amounts" else ""

st.subheader("Result on the test period")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Alerts", f"{cm['alerts']:,}", f"{pct(cm['alerts'] / len(yt), 2)} of transactions", delta_color="off")
k2.metric("Recall (fraud caught)", pct(cm["recall"]), f"{cm['TP']:,} of {int(yt.sum()):,}", delta_color="off")
k3.metric("Precision", pct(cm["precision"]), f"1 alert in {1 / max(cm['precision'], 1e-9):.1f} is fraud", delta_color="off")
k4.metric("Total cost (test)", f"{tot_t:,.0f}", f"{100 * (1 - tot_t / max(nothing_t, 1)):.1f}% below flagging nothing")

k5, k6, k7, k8 = st.columns(4)
k5.metric("F2", f"{cm['F2']:.3f}")
k6.metric("MCC", f"{cm['MCC']:.3f}")
k7.metric("Cost on validation", f"{tot_v:,.0f}", "same threshold", delta_color="off")
gap = (tot_t / max(tot_v, 1e-9) - 1) * 100 if tot_v else 0
k8.metric("Validation → test", f"{gap:+.0f}%", "cost change; large gaps = unstable threshold", delta_color="off")

if basis == "Amounts":
    d1, d2, d3 = st.columns(3)
    d1.metric("Fraud amount missed", f"{det_t['missed_amount']:,.0f}")
    d2.metric("Review cost", f"{det_t['review_cost']:,.0f}")
    d3.metric("Fraud amount in test period", f"{nothing_t:,.0f}", "= cost of doing nothing", delta_color="off")
    st.caption(f"Reference costs on the test period: flag nothing = {nothing_t:,.0f}; review everything = {everything_t:,.0f}.")

tab_cost, tab_k, tab_pr, tab_cm = st.tabs(["Cost vs. threshold", "Precision@k / Recall@k", "Precision-Recall curves", "Confusion matrix"])

with tab_cost:
    fig = go.Figure()
    for nm, y_, p_, a_, colr in (("Validation", yv, pv, av, PALETTE[0]), ("Test", yt, pt, at, PALETTE[1])):
        if basis == "Amounts":
            cc = amount_cost_curve(y_, p_, a_, review)
            fig.add_scatter(x=100 * cc["alert_rate"], y=cc["total"], mode="lines", name=nm, line_color=colr)
        else:
            cc = cost_curve(y_, p_, c_fp, c_fn)
            fig.add_scatter(x=100 * cc["alert_rate"], y=cc["cost"], mode="lines", name=nm, line_color=colr)
    fig.add_vline(x=100 * cm["alerts"] / len(yt), line_dash="dash", line_color=ALERT_COLOR, annotation_text="chosen rule")
    fig.update_layout(title="Total cost as more transactions are flagged (highest score first)", xaxis_title="% of transactions flagged",
                      yaxis_title="Total cost", xaxis_range=[0, min(30, max(5, 100 * k_share * 3))])
    st.plotly_chart(style(fig, 400), width="stretch")
    st.caption("The two curves should have their minimum at a similar alert rate. If they do not, the threshold chosen on "
               "validation will not transfer well to new data.")

with tab_k:
    ks = sorted({k_test, int(0.005 * len(yt)) or 1, int(0.01 * len(yt)) or 1, int(0.02 * len(yt)) or 1,
                 int(0.05 * len(yt)) or 1, int(0.10 * len(yt)) or 1})
    rows = []
    for k in ks:
        p_, r_ = precision_recall_at_k(yt, pt, k)
        rows.append({"k (alerts)": k, "k as % of transactions": k / len(yt), "Precision@k": p_, "Recall@k": r_,
                     "Your budget": "◀ chosen" if k == k_test else ""})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "k as % of transactions": st.column_config.NumberColumn(format="percent"),
        "Precision@k": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
        "Recall@k": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})
    st.markdown(
        f"**Why this k:** the budget comes from analyst capacity ({capacity:,} alerts/day against about {per_day:,.0f} transactions/day "
        f"= {pct(k_share, 2)}). At that budget the model finds **{pct(precision_recall_at_k(yt, pt, k_test)[1])}** of the fraud "
        f"with a precision of **{pct(precision_recall_at_k(yt, pt, k_test)[0])}**.")

with tab_pr:
    fig = go.Figure()
    for i, m in enumerate(art.models):
        pr, rc, _ = precision_recall_curve(yt, test[m].to_numpy())
        step = max(1, len(pr) // 600)
        fig.add_scatter(x=rc[::step], y=pr[::step], mode="lines", name=f"{MODEL_LABELS[m]} (AP {average_precision_score(yt, test[m]):.3f})",
                        line=dict(color=PALETTE[i % len(PALETTE)], width=3 if m == model else 1.5))
    fig.add_hline(y=float(yt.mean()), line_dash="dot", line_color="#999", annotation_text="no-skill")
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1.02])
    st.plotly_chart(style(fig, 420), width="stretch")

with tab_cm:
    z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    fig = go.Figure(go.Heatmap(z=np.log10(np.array(z) + 1), text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}",
                               x=["Not flagged", "Flagged"], y=["Legitimate", "Fraud"], colorscale="Blues", showscale=False))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(style(fig, 360, legend=False), width="stretch")
    st.write(f"PR-AUC {average_precision_score(yt, pt):.4f} · ROC-AUC {roc_auc_score(yt, pt):.4f} · "
             f"F2 {cm['F2']:.3f} · MCC {cm['MCC']:.3f} at threshold {thr:.4f}")
