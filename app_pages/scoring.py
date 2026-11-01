import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from core.artifacts import require_artifacts
from core.scoring import decision_thresholds, explain_row, score_frame
from core.ui import ALERT_COLOR, NEW_COLOR, OLD_COLOR, PALETTE, pct, style

MAX_ROWS = 300_000

art = require_artifacts()
meta = art.meta
rules = decision_thresholds(meta)

st.title("Score transactions")
st.caption("Upload raw IEEE-CIS style transactions, or enter one by hand, and get fraud scores with explanations.")

with st.sidebar:
    st.header("Settings")
    rule = st.selectbox("Flag a transaction when it matches", list(rules) + ["Top X% of this batch", "Custom threshold"])
    if rule == "Top X% of this batch":
        top_pct = st.slider("Top X%", 0.1, 20.0, 2.0, 0.1, format="%.1f%%")
    elif rule == "Custom threshold":
        custom_thr = st.slider("Threshold", 0.0, 1.0, 0.5, 0.005)
    ref_mode = st.radio(
        "Where do these transactions come from?",
        ["New transactions", "Rows from the competition files"],
        help="Client features use reference statistics from the training data. For *new* transactions they are "
             "added to the batch's own statistics. For rows that are already part of the reference data (e.g. "
             "test_transaction.csv), only the stored statistics are used so nothing is counted twice.")
    mode = "new" if ref_mode == "New transactions" else "reference"


def threshold_for(scores):
    if rule in rules:
        return float(rules[rule])
    if rule == "Top X% of this batch":
        return float(np.quantile(scores, 1 - top_pct / 100))
    return float(custom_thr)


def read_csv(file):
    return pd.read_csv(file, low_memory=False)


def show_explanation(X_row, score, key):
    table, base, logit = explain_row(art, X_row)
    top = table.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["contribution"], y=[f"{f}  =  {v}" for f, v in zip(top["feature"], top["value"])], orientation="h",
        marker_color=[ALERT_COLOR if c > 0 else NEW_COLOR for c in top["contribution"]]))
    fig.update_layout(title="What pushed this score up (red) or down (blue)", xaxis_title="Contribution to log-odds of fraud")
    st.plotly_chart(style(fig, 440, legend=False), width="stretch", key=f"expl_{key}")
    st.caption(f"Starting from the average log-odds ({base:.2f}), the contributions add up to {logit:.2f}, "
               f"which is a fraud probability of {score:.4f}.")
    with st.expander("All feature contributions"):
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"contribution": st.column_config.NumberColumn(format="%.4f")})


tab_batch, tab_single = st.tabs(["Upload a CSV", "Single transaction"])

with tab_batch:
    sample_path = art.path / "sample_transactions.csv"
    c1, c2 = st.columns(2)
    with c1:
        tx_file = st.file_uploader("Transactions CSV", type="csv",
                                   help="Needs at least TransactionDT and TransactionAmt. Use the same columns as "
                                        "train_transaction.csv / test_transaction.csv. Identity columns can be in the same file.")
    with c2:
        id_file = st.file_uploader("Identity CSV (optional)", type="csv", help="train_identity.csv / test_identity.csv, joined on TransactionID.")
    use_sample = False
    if sample_path.exists():
        use_sample = st.toggle("Use the bundled sample file instead", value=tx_file is None and art.is_demo)
    st.caption("Files are processed in memory for this session only and are not stored.")

    raw = None
    if use_sample and sample_path.exists():
        raw = pd.read_csv(sample_path)
        st.info(f"Using the bundled sample ({len(raw)} rows).")
    elif tx_file is not None:
        try:
            raw = read_csv(tx_file)
            if id_file is not None:
                ident = read_csv(id_file).rename(columns=lambda c: c.replace("id-", "id_"))
                raw = raw.merge(ident, on="TransactionID", how="left")
        except Exception as e:
            st.error(f"Could not read the file: {e}")

    if raw is not None:
        if len(raw) > MAX_ROWS:
            st.warning(f"Scoring the first {MAX_ROWS:,} of {len(raw):,} rows to stay within memory limits.")
            raw = raw.iloc[:MAX_ROWS]
        sig = (len(raw), tuple(raw.columns[:8]), mode, float(pd.to_numeric(raw.get("TransactionAmt"), errors="coerce").sum()))
        if st.session_state.get("score_sig") != sig:
            try:
                with st.spinner(f"Scoring {len(raw):,} transactions..."):
                    scores, X, info = score_frame(art, raw, mode)
                st.session_state.update(score_sig=sig, scores=scores, X=X, info=info, raw=raw)
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                st.stop()
        scores, X, info, raw = (st.session_state[k] for k in ("scores", "X", "info", "raw"))
        thr = threshold_for(scores)
        flag = scores >= thr

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Transactions", f"{len(scores):,}")
        k2.metric("Flagged", f"{int(flag.sum()):,}", pct(flag.mean(), 2) + " of batch", delta_color="off")
        k3.metric("Highest score", f"{scores.max():.3f}")
        k4.metric("Threshold used", f"{thr:.4f}")

        if "isFraud" in raw.columns:
            yy = pd.to_numeric(raw["isFraud"], errors="coerce")
            if yy.notna().all() and yy.nunique() == 2:
                with st.container(border=True):
                    st.markdown("**Labels found in your file.** Quality on this batch:")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("ROC-AUC", f"{roc_auc_score(yy, scores):.3f}")
                    m2.metric("Average precision", f"{average_precision_score(yy, scores):.3f}")
                    caught = int(((yy == 1) & flag).sum())
                    m3.metric("Fraud caught", f"{caught} of {int(yy.sum())}")

        with st.expander("Data quality checks", expanded=bool(info["missing_columns"] or info["unseen_category_share"])):
            st.write(f"- Rows with missing `D1` (weakens the client ID): **{pct(info['d1_missing_share'])}**")
            st.write(f"- Rows whose client has earlier transactions in the reference data or batch: **{pct(info['clients_seen_in_reference_share'])}**")
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

        id_col = "TransactionID" if "TransactionID" in raw.columns else None
        out = pd.DataFrame({"row": np.arange(len(scores))})
        if id_col:
            out["TransactionID"] = raw[id_col].to_numpy()
        out["score"] = scores
        out["percentile"] = pd.Series(scores).rank(pct=True).to_numpy()
        out["flagged"] = flag
        for c in ("TransactionAmt", "ProductCD", "P_emaildomain"):
            if c in raw.columns:
                out[c] = raw[c].to_numpy()
        ranked = out.sort_values("score", ascending=False)

        st.subheader("Highest-risk transactions")
        st.dataframe(ranked.head(200), hide_index=True, width="stretch",
                     column_config={"score": st.column_config.ProgressColumn("score", format="%.4f", min_value=0, max_value=1),
                                    "percentile": st.column_config.NumberColumn(format="percent")})
        d1, d2 = st.columns(2)
        d1.download_button("Download all scores (CSV)", out.to_csv(index=False).encode(), "scores.csv", "text/csv")
        d2.download_button("Download flagged only (CSV)", out[out["flagged"]].to_csv(index=False).encode(), "flagged.csv", "text/csv")

        st.subheader("Explain a transaction")
        labels = ranked.head(200)
        pick = st.selectbox(
            "Choose from the 200 highest-risk rows", labels["row"].tolist(),
            format_func=lambda r: (f"{raw.iloc[r][id_col]} " if id_col else f"row {r} ") + f"· score {scores[r]:.4f}"
                                  + (" · FLAGGED" if flag[r] else ""))
        show_explanation(X.iloc[[pick]], scores[pick], f"batch_{pick}")
    else:
        st.info("Upload a file, or switch on the bundled sample, to see results.")

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
            default = {"card1": 9500.0, "card2": 400.0, "card3": 150.0, "card5": 226.0, "addr1": 300.0, "D1": 20.0,
                       "C1": 1.0, "C2": 1.0}.get(name, 0.0)
            use = cols[i % 4].text_input(name, value=str(default), help="Leave empty for missing")
            vals[name] = pd.to_numeric(use, errors="coerce") if use.strip() else np.nan
        go_ = st.form_submit_button("Score this transaction", type="primary")
    if go_:
        row = {"TransactionDT": day * 86400 + hour * 3600, "TransactionAmt": amount}
        for k, v in vals.items():
            row[k] = np.nan if v == "(missing)" else v
        df_one = pd.DataFrame([row])
        s, X1, info1 = score_frame(art, df_one, "new")
        thr1 = threshold_for(np.array([s[0]])) if rule != "Top X% of this batch" else float(rules[list(rules)[0]])
        verdict = "FLAG" if s[0] >= thr1 else "Pass"
        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud probability", f"{s[0]:.4f}")
        m2.metric("Decision", verdict, f"threshold {thr1:.4f}", delta_color="off")
        m3.metric("Compared with holdout scores", f"top {100 * float((art.preds['p_v2'] >= s[0]).mean()):.2f}%"
                  if "p_v2" in art.preds else "n/a")
        show_explanation(X1, s[0], "single")
        st.caption("A hand-entered transaction has no client history in the reference data unless card1/2/3/5, addr1 "
                   "and D1 match a known client, so the client features mostly reflect a first-time client.")
