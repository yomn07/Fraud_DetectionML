import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import rankdata

from core.artifacts import MODEL_LABELS, require_artifacts
from core.metrics import gain_curve, precision_recall_curve, roc_curve, summary
from core.ui import ALERT_COLOR, PALETTE, style

art = require_artifacts()
comp = art.comparison

st.title("Supervised vs. unsupervised")
if comp is None:
    st.info("This comparison needs `comparison_sample.csv.gz` from the notebook export (section 18).")
    st.stop()

# LOF is shown in its version without PCA: drop the PCA-based LOF and give the other one the plain name.
if "u:LOF (no PCA)" in comp.columns:
    comp = comp.drop(columns=["u:LOF"], errors="ignore").rename(columns={"u:LOF (no PCA)": "u:LOF"})

y = comp["y_true"].to_numpy()
fraud_rate = float(y.mean())
sup_cols = [c for c in MODEL_LABELS if c in comp.columns]
uns_cols = [c for c in comp.columns if c.startswith("u:") and "pca" not in c.lower()]  # PCA variants are not shown
labels = {**{c: MODEL_LABELS[c] for c in sup_cols}, **{c: c[2:] for c in uns_cols}}
family = {**{c: "Supervised" for c in sup_cols}, **{c: "Unsupervised" for c in uns_cols}}

st.caption(f"Every model below is scored on the **same {len(comp):,} holdout transactions** ({fraud_rate:.2%} fraud), "
           "the most recent part of the data. Unsupervised detectors never saw a label; supervised models learned from them.")


@st.cache_data(show_spinner=False)
def table(_comp, cols):
    rows = []
    for c in cols:
        m = summary(_comp["y_true"].to_numpy(), _comp[c].to_numpy())
        rows.append({"Family": family[c], "Model": labels[c], "key": c, **m, "Lift in AP": m["AP"] / _comp["y_true"].mean()})
    return pd.DataFrame(rows).sort_values("AP", ascending=False).reset_index(drop=True)


tbl = table(comp, tuple(sup_cols + uns_cols))
best_sup = tbl[tbl["Family"] == "Supervised"].iloc[0] if len(sup_cols) else None
best_uns = tbl[tbl["Family"] == "Unsupervised"].iloc[0] if len(uns_cols) else None

c1, c2, c3, c4 = st.columns(4)
if best_sup is not None:
    c1.metric("Best supervised (AP)", f"{best_sup['AP']:.3f}", best_sup["Model"], delta_color="off")
if best_uns is not None:
    c2.metric("Best unsupervised (AP)", f"{best_uns['AP']:.3f}", best_uns["Model"], delta_color="off")
c3.metric("Random ranking (AP)", f"{fraud_rate:.3f}", "= fraud rate", delta_color="off")
if best_sup is not None and best_uns is not None:
    c4.metric("Supervised advantage", f"{best_sup['AP'] / max(best_uns['AP'], 1e-9):.1f}x", "in average precision", delta_color="off")

tab_tbl, tab_curves, tab_agree, tab_exp = st.tabs(["Ranking table", "Curves", "Do they find the same fraud?", "Unsupervised experiments"])

with tab_tbl:
    show = tbl.drop(columns="key")
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "AUC": st.column_config.NumberColumn(format="%.4f"), "AP": st.column_config.NumberColumn("Avg. precision", format="%.4f"),
        "Recall @1%": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
        "Recall @5%": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
        "Lift in AP": st.column_config.NumberColumn(format="%.1fx", help="Average precision divided by the fraud rate (what random ranking achieves)."),
    })
    fig = go.Figure()
    for fam, colr in (("Supervised", PALETTE[0]), ("Unsupervised", PALETTE[1])):
        d = tbl[tbl["Family"] == fam]
        fig.add_bar(x=d["Model"], y=d["AP"], name=fam, marker_color=colr)
    fig.add_hline(y=fraud_rate, line_dash="dot", line_color="#999", annotation_text="random ranking")
    fig.update_layout(title="Average precision (higher is better)", yaxis_title="Average precision", xaxis_tickangle=-30)
    st.plotly_chart(style(fig, 420), width="stretch")
    st.markdown(
        "**Reading this:** AUC says how well a model orders fraud above legitimate transactions overall; average precision "
        "says how clean the *top of the list* is, which is what an analyst with a limited review budget cares about.")

with tab_curves:
    default = [c for c in (best_sup["key"] if best_sup is not None else None, best_uns["key"] if best_uns is not None else None) if c]
    chosen = st.multiselect("Models", sup_cols + uns_cols, default=default, format_func=labels.get)
    kind = st.radio("Curve", ["Gain (alert budget)", "Precision-Recall", "ROC"], horizontal=True)
    fig = go.Figure()
    for i, c in enumerate(chosen):
        s = comp[c].to_numpy()
        if kind.startswith("Gain"):
            x, g = gain_curve(y, s)
            fig.add_scatter(x=100 * x, y=100 * g, mode="lines", name=labels[c], line_color=PALETTE[i % len(PALETTE)])
        elif kind.startswith("Precision"):
            p, r, _ = precision_recall_curve(y, s)
            st_ = max(1, len(p) // 500)
            fig.add_scatter(x=r[::st_], y=p[::st_], mode="lines", name=labels[c], line_color=PALETTE[i % len(PALETTE)])
        else:
            f, t, _ = roc_curve(y, s)
            st_ = max(1, len(f) // 500)
            fig.add_scatter(x=f[::st_], y=t[::st_], mode="lines", name=labels[c], line_color=PALETTE[i % len(PALETTE)])
    if kind.startswith("Gain"):
        fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random", line=dict(dash="dot", color="#999"))
        fig.update_layout(xaxis_title="% flagged", yaxis_title="% of fraud caught", xaxis_range=[0, 30])
    elif kind.startswith("Precision"):
        fig.add_hline(y=fraud_rate, line_dash="dot", line_color="#999")
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    else:
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dot", color="#999"))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(style(fig, 440), width="stretch")

with tab_agree:
    if best_sup is None or not uns_cols:
        st.info("Needs at least one supervised and one unsupervised model.")
    else:
        a = st.selectbox("Supervised model", sup_cols, index=sup_cols.index(best_sup["key"]), format_func=labels.get)
        b = st.selectbox("Unsupervised detector", uns_cols, index=uns_cols.index(best_uns["key"]), format_func=labels.get)
        pa = rankdata(comp[a]) / len(comp)
        pb = rankdata(comp[b]) / len(comp)
        top = st.slider("What counts as 'flagged' (top X% of each list)", 1.0, 20.0, 5.0, 0.5, format="%.1f%%") / 100
        fa, fb = pa >= 1 - top, pb >= 1 - top
        fr = y == 1
        caught_a, caught_b = (fa & fr).sum(), (fb & fr).sum()
        only_b = (fb & ~fa & fr).sum()
        both = (fa & fb & fr).sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Fraud caught by {labels[a]}", f"{caught_a}", f"of {fr.sum()}", delta_color="off")
        k2.metric(f"Fraud caught by {labels[b]}", f"{caught_b}", f"of {fr.sum()}", delta_color="off")
        k3.metric("Caught by both", f"{both}")
        k4.metric("Only the unsupervised one", f"{only_b}", "missed by the supervised model", delta_color="off")
        rng = np.random.RandomState(0)
        idx = np.concatenate([rng.choice(np.flatnonzero(~fr), min(4000, (~fr).sum()), replace=False), np.flatnonzero(fr)])
        fig = go.Figure()
        fig.add_scatter(x=pa[idx][~fr[idx]], y=pb[idx][~fr[idx]], mode="markers", name="Legitimate",
                        marker=dict(size=4, color="rgba(100,116,139,0.35)"))
        fig.add_scatter(x=pa[idx][fr[idx]], y=pb[idx][fr[idx]], mode="markers", name="Fraud", marker=dict(size=6, color=ALERT_COLOR))
        fig.update_layout(xaxis_title=f"{labels[a]}: rank percentile", yaxis_title=f"{labels[b]}: rank percentile")
        st.plotly_chart(style(fig, 460), width="stretch")
        corr = pd.DataFrame({labels[c]: comp[c] for c in sup_cols + uns_cols}).corr(method="spearman").round(2)
        with st.expander("Rank correlation between all models"):
            st.dataframe(corr, width="stretch")
        st.caption("Fraud in the top-right corner is found by both. Fraud that sits high on the vertical axis but low on the "
                   "horizontal axis is the case for keeping an unsupervised detector as a second opinion.")

with tab_exp:
    def _no_pca(rows):
        return [r for r in rows if not any("pca" in str(v).lower() for v in r.values())]

    ut = {k: _no_pca(v) for k, v in (art.meta.get("unsup_tables") or {}).items() if v}
    if not any(ut.values()):
        # Older exports have no tables in meta.json: rebuild them from the scores in comparison_sample.csv.gz.
        final = {"Isolation Forest", "LOF", "One-Class SVM (approx.)", "Autoencoder", "Rank-average ensemble"}
        cols = ["Model", "AUC", "AP", "Recall @1%", "Recall @5%"]
        uns = tbl[tbl["Family"] == "Unsupervised"]
        ut = {"unsup_results": uns[uns["Model"].isin(final)][cols].to_dict("records"),
              "unsup_tests": uns[~uns["Model"].isin(final)][cols].to_dict("records")}
    if not any(ut.values()):
        st.info("No unsupervised experiment results are available.")
    names = {"unsup_results": "Detectors (final settings)", "unsup_tests": "Extra tests: longer autoencoder, ensembles"}
    for k, v in ut.items():
        if v:
            st.markdown(f"**{names.get(k, k)}**")
            st.dataframe(pd.DataFrame(v), hide_index=True, width="stretch")
    s = dict(art.meta.get("unsup_settings") or {})
    parts = (art.unsup or {}).get("parts") or {}
    if not s and parts:  # fall back to the columns stored with the unsupervised models
        s = {"num_cols": list(parts.get("num_cols", [])), "freq_cols": list(parts.get("freq_cols", []))}
    if s:
        text = (f"Unsupervised settings: {len(s.get('num_cols', []))} numeric and "
                f"{len(s.get('freq_cols', []))} frequency-encoded columns")
        if s.get("feature_mode"):
            text += f", feature mode **{s['feature_mode']}**"
        if "fit_on_legit_only" in s:
            text += f", fit on {'legitimate rows only' if s['fit_on_legit_only'] else 'all training rows (labels not used)'}"
        st.markdown(text + ".")
        with st.expander("Columns used"):
            st.write("Numeric: " + ", ".join(f"`{c}`" for c in s.get("num_cols", [])))
            st.write("Frequency-encoded: " + ", ".join(f"`{c}`" for c in s.get("freq_cols", [])))
    st.markdown(
        "**Why the unsupervised models trail:** they learn what 'unusual' looks like, not what 'fraudulent' looks like. "
        "Plenty of unusual transactions are legitimate, and plenty of fraud is made to look ordinary. They need no labels, so "
        "they are useful for new fraud patterns and for cold-start, not as a replacement for the supervised model.")
