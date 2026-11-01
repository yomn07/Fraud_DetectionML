import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from core.artifacts import require_artifacts
from core.batch import score_stream
from core.scoring import decision_thresholds, explain_row, reason_codes, score_frame, waterfall_data
from core.unsup import available_detectors, score_unsup, to_percentile
from core.ui import ALERT_COLOR, NEW_COLOR, PALETTE, pct, style

art = require_artifacts()
meta = art.meta
rules = decision_thresholds(meta)

st.title("Score transactions")
st.caption("Rank transactions by risk, flag the most suspicious, and see why. Works on files of any size "
           "(processed in chunks).")

with st.sidebar:
    st.header("Settings")
    rule = st.selectbox("Flag a transaction when it matches", list(rules) + ["Top X% of this batch", "Custom threshold"])
    top_pct = st.slider("Top X%", 0.1, 20.0, 2.0, 0.1, format="%.1f%%") if rule == "Top X% of this batch" else None
    custom_thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.005) if rule == "Custom threshold" else None
    ref_mode = st.radio(
        "Where do these transactions come from?", ["New transactions", "Rows from the competition files"],
        help="Client features use reference statistics from the training data. For *new* transactions they are added to the "
             "file's own statistics. For rows that are already part of the reference data (e.g. train_transaction.csv or "
             "test_transaction.csv) only the stored statistics are used, so nothing is counted twice.")
    mode = "new" if ref_mode == "New transactions" else "reference"
    with_unsup = st.toggle("Also compute anomaly scores (unsupervised)", value=art.unsup is not None, disabled=art.unsup is None)
    if art.unsup_error:
        st.warning(art.unsup_error)


def threshold_for(scores):
    if rule in rules:
        return float(rules[rule])
    if rule == "Top X% of this batch":
        return float(np.quantile(scores, 1 - top_pct / 100))
    return float(custom_thr)


def show_explanation(X_row, score, key):
    table, base, logit = explain_row(art, X_row)
    codes = reason_codes(table)
    st.markdown("**Reason codes** (what pushed this score up the most)")
    cols = st.columns(len(codes))
    for c, r in zip(cols, codes):
        with c.container(border=True):
            st.markdown(f"**{r['code']}**")
            st.markdown(f"`{r['feature']}`")
            st.caption(f"value: {r['value']}")
            st.markdown(f":red[+{r['effect']:.2f}]")
    labels, values, measure = waterfall_data(table, base)
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=measure, y=labels, x=values,
        increasing=dict(marker_color=ALERT_COLOR), decreasing=dict(marker_color=NEW_COLOR), totals=dict(marker_color="#64748b"),
        connector=dict(line=dict(color="rgba(128,128,128,0.4)"))))
    fig.update_layout(title="SHAP waterfall: from the average to this transaction (log-odds)", xaxis_title="log-odds of fraud",
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(style(fig, 460, legend=False), width="stretch", key=f"wf_{key}")
    st.caption(f"Starting from the average log-odds ({base:.2f}), contributions add up to {logit:.2f}: a fraud probability of {score:.4f}.")
    with st.expander("All feature contributions"):
        st.dataframe(table, hide_index=True, width="stretch", column_config={"contribution": st.column_config.NumberColumn(format="%.4f")})


tab_batch, tab_single = st.tabs(["Score a file", "Single transaction"])

with tab_batch:
    sample_path = art.path / "sample_transactions.csv"
    sources = ["Upload a CSV", "File on this computer (for very large files)"] + (["Bundled sample"] if sample_path.exists() else [])
    src_kind = st.radio("Data source", sources, horizontal=True, index=len(sources) - 1 if (art.is_demo and sample_path.exists()) else 0)
    tx_src, id_src, sig_key = None, None, None
    if src_kind == "Upload a CSV":
        c1, c2 = st.columns(2)
        with c1:
            up = st.file_uploader("Transactions CSV", type="csv",
                                  help="Needs TransactionDT and TransactionAmt, plus the same columns as train_transaction.csv / test_transaction.csv.")
        with c2:
            up_id = st.file_uploader("Identity CSV (optional)", type="csv", help="train_identity.csv / test_identity.csv, joined on TransactionID.")
        st.caption("Upload limit is 1 GB. A hosted app has little memory, so for the full 0.6 GB training file prefer the next option on your own machine.")
        if up is not None:
            tx_src, id_src, sig_key = up, up_id, ("up", up.name, up.size)
    elif src_kind.startswith("File on this computer"):
        p1 = st.text_input("Path to the transactions CSV", placeholder="/path/to/train_transaction.csv")
        p2 = st.text_input("Path to the identity CSV (optional)", placeholder="/path/to/train_identity.csv")
        if p1:
            if os.path.isfile(p1):
                tx_src, id_src = p1, (p2 if p2 and os.path.isfile(p2) else None)
                sig_key = ("path", p1, os.path.getsize(p1), p2)
            else:
                st.error("File not found on the machine running the app. (On a hosted app, this is the server, not your computer.)")
    else:
        tx_src, sig_key = sample_path, ("sample",)
    st.caption("Files are processed in memory for this session only and are not stored.")

    if tx_src is not None:
        sig = (sig_key, mode, with_unsup)
        if st.session_state.get("score_sig") != sig:
            bar = st.progress(0.0, text="Scoring...")
            try:
                res = score_stream(art, tx_src, id_src, mode=mode, with_unsup=with_unsup, progress=lambda f: bar.progress(f, text=f"Scoring... {f:.0%}"))
            except Exception as e:
                bar.empty()
                st.error(f"Scoring failed: {e}")
                st.stop()
            bar.empty()
            st.session_state.update(score_sig=sig, res=res)
        res = st.session_state["res"]
        scores, disp, info, top_X = res["scores"], res["display"], res["info"], res["top_X"]
        thr = threshold_for(scores)
        flag = scores >= thr

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transactions", f"{len(scores):,}")
        k2.metric("Flagged", f"{int(flag.sum()):,}", pct(flag.mean(), 2) + " of file", delta_color="off")
        k3.metric("Highest score", f"{scores.max():.3f}")
        k4.metric("Threshold used", f"{thr:.4f}")

        if "isFraud" in disp.columns:
            yy = pd.to_numeric(disp["isFraud"], errors="coerce")
            if yy.notna().all() and yy.nunique() == 2:
                with st.container(border=True):
                    st.markdown("**Labels found in your file** (note: if this is the training data, the model has seen part of it).")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("ROC-AUC", f"{roc_auc_score(yy, scores):.3f}")
                    m2.metric("PR-AUC", f"{average_precision_score(yy, scores):.3f}")
                    m3.metric("Fraud caught", f"{int(((yy == 1) & flag).sum())} of {int(yy.sum())}")

        with st.expander("Data quality checks", expanded=bool(info["missing_columns"] or info["unseen_category_share"])):
            st.write(f"- Rows with missing `D1` (weakens the client ID): **{pct(info['d1_missing_share'])}**")
            st.write(f"- Rows whose client has earlier transactions in the reference data or file: **{pct(info['clients_seen_in_reference_share'])}**")
            if info["missing_columns"]:
                st.write(f"- {len(info['missing_columns'])} model columns are absent from your file and were treated as missing: "
                         + ", ".join(f"`{c}`" for c in info["missing_columns"][:15]) + (" ..." if len(info["missing_columns"]) > 15 else ""))
            if info["unseen_category_share"]:
                st.write("- Categorical values not seen in training (treated as missing): "
                         + ", ".join(f"`{k}` {pct(v)}" for k, v in info["unseen_category_share"].items()))

        fig = go.Figure(go.Histogram(x=scores, nbinsx=50, marker_color=PALETTE[0]))
        fig.add_vline(x=thr, line_dash="dash", line_color=ALERT_COLOR, annotation_text="threshold")
        fig.update_layout(title="Score distribution", xaxis_title="Fraud probability", yaxis_title="Transactions", yaxis_type="log")
        st.plotly_chart(style(fig, 280, legend=False), width="stretch")

        out = pd.DataFrame({"row": np.arange(len(scores))})
        for c in ("TransactionID",):
            if c in disp.columns:
                out[c] = disp[c].to_numpy()
        out["score"] = scores
        out["percentile"] = pd.Series(scores).rank(pct=True).to_numpy()
        out["flagged"] = flag
        for name, raw_s in res["unsup"].items():
            out[f"anomaly: {name}"] = to_percentile(art.unsup, name, raw_s).round(1)
        if res["unsup"]:
            first = next(iter(res["unsup"]))
            out["second opinion"] = (~flag) & (out[f"anomaly: {first}"] >= 95)
        for c in ("TransactionAmt", "ProductCD", "P_emaildomain"):
            if c in disp.columns:
                out[c] = disp[c].to_numpy()
        ranked = out.sort_values("score", ascending=False)

        st.subheader("Highest-risk transactions")
        colcfg = {"score": st.column_config.ProgressColumn("score", format="%.4f", min_value=0, max_value=1),
                  "percentile": st.column_config.NumberColumn(format="percent")}
        for name in res["unsup"]:
            colcfg[f"anomaly: {name}"] = st.column_config.NumberColumn(f"anomaly: {name}", format="%.0f", help="Percentile among holdout transactions; 99 = more unusual than 99% of them.")
        st.dataframe(ranked.head(200), hide_index=True, width="stretch", column_config=colcfg)
        if res["unsup"]:
            st.caption("**Second opinion** = not flagged by the supervised model, but among the 5% most unusual for the first anomaly detector.")
        d1, d2 = st.columns(2)
        d1.download_button("Download all scores (CSV)", out.to_csv(index=False).encode(), "scores.csv", "text/csv")
        d2.download_button("Download flagged only (CSV)", out[out["flagged"]].to_csv(index=False).encode(), "flagged.csv", "text/csv")

        st.subheader("Explain an alert")
        cand = [r for r in ranked.head(200)["row"].tolist() if r in top_X.index]
        pick = st.selectbox(
            "Choose from the highest-risk rows", cand,
            format_func=lambda r: (f"{disp.iloc[r]['TransactionID']} " if "TransactionID" in disp.columns else f"row {r} ")
                                  + f"· score {scores[r]:.4f}" + (" · FLAGGED" if flag[r] else ""))
        if pick is not None:
            show_explanation(top_X.loc[[pick]], scores[pick], f"batch_{pick}")
    else:
        st.info("Choose a data source above to see results.")

with tab_single:
    st.markdown("Enter the main fields of one transaction. Everything not listed is treated as **missing**, "
                "which the model handles natively.")
    names = set(meta["feature_names"])
    levels = meta["category_levels"]
    with st.form("single"):
        a, b, c = st.columns(3)
        amount = a.number_input("TransactionAmt", 0.0, 1_000_000.0, 59.0, 1.0)
        day = b.number_input("Day (since dataset start)", 0, 1000, 200, 1, help="Used with D1 to derive the client's first-seen day.")
        hour = c.slider("Hour of day", 0, 23, 14)
        vals = {}
        cat_inputs = [c_ for c_ in ("ProductCD", "card4", "card6", "P_emaildomain", "DeviceType") if c_ in levels]
        cols = st.columns(max(1, len(cat_inputs)))
        for col, name in zip(cols, cat_inputs):
            vals[name] = col.selectbox(name, ["(missing)"] + list(levels[name]))
        num_inputs = [c_ for c_ in ("card1", "card2", "card3", "card5", "addr1", "D1", "C1", "C2") if c_ in names]
        cols = st.columns(4)
        for i, name in enumerate(num_inputs):
            default = {"card1": 9500.0, "card2": 400.0, "card3": 150.0, "card5": 226.0, "addr1": 300.0, "D1": 20.0, "C1": 1.0, "C2": 1.0}.get(name, 0.0)
            use = cols[i % 4].text_input(name, value=str(default), help="Leave empty for missing")
            vals[name] = pd.to_numeric(use, errors="coerce") if use.strip() else np.nan
        go_ = st.form_submit_button("Score this transaction", type="primary")
    if go_:
        row = {"TransactionDT": day * 86400 + hour * 3600, "TransactionAmt": amount}
        for k, v in vals.items():
            row[k] = np.nan if v == "(missing)" else v
        s, X1, info1 = score_frame(art, pd.DataFrame([row]), "new")
        thr1 = float(next(iter(rules.values()))) if rule == "Top X% of this batch" else threshold_for(np.array([s[0]]))
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability", f"{s[0]:.4f}")
        m2.metric("Decision", "FLAG" if s[0] >= thr1 else "Pass", f"threshold {thr1:.4f}", delta_color="off")
        m3.metric("Compared with test-period scores", f"top {100 * float((art.preds['p_v2'] >= s[0]).mean()):.2f}%" if "p_v2" in art.preds else "n/a")
        if art.unsup is not None:
            ps = {n: float(to_percentile(art.unsup, n, v)[0]) for n, v in score_unsup(art.unsup, X1).items()}
            st.write("Anomaly percentiles (unsupervised): " + ", ".join(f"**{n}** {v:.0f}" for n, v in ps.items()))
        show_explanation(X1, s[0], "single")
        st.caption("A hand-entered transaction has no client history unless card1/2/3/5, addr1 and D1 match a known client, "
                   "so the client features mostly reflect a first-time client.")
