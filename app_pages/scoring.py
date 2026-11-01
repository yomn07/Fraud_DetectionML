
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

from core.artifacts import require_artifacts
from core.scoring import decision_thresholds, explain_row, score_frame
from core.ui import ALERT_COLOR, NEW_COLOR, PALETTE, pct, style


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MAX_ROWS = 300

art = require_artifacts()
meta = art.meta
rules = decision_thresholds(meta)

st.title("🛡️ Fraud Detection Studio")
st.caption(
    "IEEE-CIS fraud detection · Local dataset loading · "
    "LightGBM fraud scoring and transaction explanations"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(transaction_path, identity_path, max_rows):

    raw = pd.read_csv(
        transaction_path,
        low_memory=False,
        nrows=max_rows
    )

    if identity_path is not None:
        ident = pd.read_csv(
            identity_path,
            low_memory=False
        )

        ident = ident.rename(
            columns=lambda c: c.replace("id-", "id_")
        )

        # Keep only identity rows belonging to loaded transactions
        ident = ident[
            ident["TransactionID"].isin(raw["TransactionID"])
        ]

        raw = raw.merge(
            ident,
            on="TransactionID",
            how="left",
            validate="one_to_one"
        )

    return raw


def available_datasets():

    datasets = {}

    for split in ("train", "test"):

        transaction = DATA_DIR / f"{split}_transaction.csv"
        identity = DATA_DIR / f"{split}_identity.csv"

        if transaction.exists():

            datasets[split] = {
                "transaction": transaction,
                "identity": identity if identity.exists() else None
            }

    return datasets


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.header("Settings")

    st.caption(f"Data directory: {DATA_DIR}")

    rule = st.selectbox(
        "Flag a transaction when it matches",
        list(rules) + [
            "Top X% of this batch",
            "Custom threshold"
        ]
    )

    if rule == "Top X% of this batch":

        top_pct = st.slider(
            "Top X%",
            0.1,
            20.0,
            2.0,
            0.1,
            format="%.1f%%"
        )

    elif rule == "Custom threshold":

        custom_thr = st.slider(
            "Threshold",
            0.0,
            1.0,
            0.5,
            0.005
        )

    ref_mode = st.radio(
        "Transaction source",
        [
            "New transactions",
            "Rows from competition files"
        ],
        help=(
            "Competition rows use stored reference statistics. "
            "New transactions use the batch's own statistics."
        )
    )

    mode = (
        "new"
        if ref_mode == "New transactions"
        else "reference"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def threshold_for(scores):

    if rule in rules:
        return float(rules[rule])

    if rule == "Top X% of this batch":
        return float(
            np.quantile(scores, 1 - top_pct / 100)
        )

    return float(custom_thr)


def show_explanation(X_row, score, key):

    table, base, logit = explain_row(art, X_row)

    top = table.head(12).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=top["contribution"],
            y=[
                f"{f} = {v}"
                for f, v in zip(
                    top["feature"],
                    top["value"]
                )
            ],
            orientation="h",
            marker_color=[
                ALERT_COLOR if c > 0 else NEW_COLOR
                for c in top["contribution"]
            ]
        )
    )

    fig.update_layout(
        title="What pushed this score up or down",
        xaxis_title="Contribution to fraud log-odds"
    )

    st.plotly_chart(
        style(fig, 440, legend=False),
        width="stretch",
        key=f"expl_{key}"
    )

    st.caption(
        f"Starting from average log-odds ({base:.2f}), "
        f"the contributions add up to {logit:.2f}, "
        f"giving a fraud probability of {score:.4f}."
    )

    with st.expander("All feature contributions"):

        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "contribution": st.column_config.NumberColumn(
                    format="%.4f"
                )
            }
        )


# ============================================================
# DATASET SELECTION
# ============================================================

tab_batch, tab_single = st.tabs(
    [
        "Dataset scoring",
        "Single transaction"
    ]
)


with tab_batch:

    st.subheader("Load transactions from your data folder")

    datasets = available_datasets()

    if not datasets:

        st.error("No datasets found.")

        st.code(
            """
fraud_detectionml/
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
            """,
            language="text"
        )

        st.stop()

    dataset_options = list(datasets.keys())

    default_index = (
        dataset_options.index("test")
        if "test" in dataset_options
        else 0
    )

    selected = st.selectbox(
        "Choose your dataset",
        dataset_options,
        index=default_index,
        format_func=lambda x: (
            "Training dataset (contains fraud labels)"
            if x == "train"
            else "Testing dataset (unlabelled)"
        )
    )

    st.caption(
        f"Transaction file: "
        f"`{datasets[selected]['transaction'].name}`"
    )

    if datasets[selected]["identity"]:

        st.caption(
            f"Identity file: "
            f"`{datasets[selected]['identity'].name}`"
        )

    else:

        st.caption("No matching identity CSV found. Identity features will be missing.")

    if selected == "train":

        st.info(
            "Training data contains isFraud labels. "
            "Metrics on these rows are not independent holdout evaluation."
        )

    else:

        st.info(
            "Test data does not contain fraud labels. "
            "The dashboard will display predictions without accuracy metrics."
        )

    load_clicked = st.button(
        "Load dataset",
        type="primary",
        use_container_width=True
    )

    if load_clicked:

        st.session_state.pop("score_sig", None)
        st.session_state.pop("scores", None)

        try:

            with st.spinner("Reading CSV files from data folder..."):

                selected_files = datasets[selected]

                raw_loaded = load_dataset(
                    str(selected_files["transaction"]),
                    (
                        str(selected_files["identity"])
                        if selected_files["identity"]
                        else None
                    ),
                    MAX_ROWS
                )

                st.session_state["local_raw"] = raw_loaded
                st.session_state["local_dataset"] = selected

        except Exception as e:

            st.error(f"Dataset loading failed: {e}")

    raw = st.session_state.get("local_raw")

    # Do not accidentally use a previously selected dataset
    if st.session_state.get("local_dataset") != selected:
        raw = None

    # ========================================================
    # SCORE DATASET
    # ========================================================

    if raw is not None:

        st.success(
            f"Successfully loaded {len(raw):,} transactions."
        )

        if len(raw) >= MAX_ROWS:

            st.warning(
                f"Limited to the first {MAX_ROWS:,} rows "
                "to control memory usage."
            )

        signature = (
            selected,
            len(raw),
            tuple(raw.columns),
            mode,
            float(
                pd.to_numeric(
                    raw.get("TransactionAmt"),
                    errors="coerce"
                ).sum()
            )
        )

        if st.session_state.get("score_sig") != signature:

            try:

                with st.spinner(
                    f"Scoring {len(raw):,} transactions..."
                ):

                    scores, X, info = score_frame(
                        art,
                        raw,
                        mode
                    )

                st.session_state.update(
                    score_sig=signature,
                    scores=scores,
                    X=X,
                    info=info
                )

            except Exception as e:

                st.error(f"Scoring failed: {e}")

                st.exception(e)

                st.stop()

        scores = st.session_state["scores"]
        X = st.session_state["X"]
        info = st.session_state["info"]

        thr = threshold_for(scores)
        flag = scores >= thr

        # ====================================================
        # SUMMARY METRICS
        # ====================================================

        st.subheader("Fraud detection results")

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Transactions",
            f"{len(scores):,}"
        )

        k2.metric(
            "Flagged",
            f"{int(flag.sum()):,}",
            pct(flag.mean(), 2) + " of batch",
            delta_color="off"
        )

        k3.metric(
            "Highest score",
            f"{scores.max():.3f}"
        )

        k4.metric(
            "Threshold",
            f"{thr:.4f}"
        )

        # ====================================================
        # EVALUATION
        # ====================================================

        if "isFraud" in raw.columns:

            yy = pd.to_numeric(
                raw["isFraud"],
                errors="coerce"
            )

            valid = yy.notna()

            if valid.sum() > 0 and yy[valid].nunique() == 2:

                with st.container(border=True):

                    st.markdown(
                        "**Fraud detection evaluation**"
                    )

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "ROC-AUC",
                        f"{roc_auc_score(yy[valid], scores[valid]):.3f}"
                    )

                    m2.metric(
                        "Average precision",
                        f"{average_precision_score(yy[valid], scores[valid]):.3f}"
                    )

                    caught = int(
                        ((yy == 1) & flag).sum()
                    )

                    m3.metric(
                        "Fraud caught",
                        f"{caught} of {int(yy.sum())}"
                    )

        # ====================================================
        # DATA QUALITY
        # ====================================================

        with st.expander(
            "Data quality checks",
            expanded=bool(
                info["missing_columns"]
                or info["unseen_category_share"]
            )
        ):

            st.write(
                f"- Rows with missing D1: "
                f"**{pct(info['d1_missing_share'])}**"
            )

            st.write(
                f"- Clients seen in reference data or batch: "
                f"**{pct(info['clients_seen_in_reference_share'])}**"
            )

            if info["missing_columns"]:

                st.warning(
                    "Missing model columns: "
                    + ", ".join(
                        info["missing_columns"][:20]
                    )
                )

            if info["unseen_category_share"]:

                st.write(
                    "Unseen categorical values:"
                )

                st.json(
                    info["unseen_category_share"]
                )

        # ====================================================
        # SCORE DISTRIBUTION
        # ====================================================

        st.subheader("Fraud probability distribution")

        fig = go.Figure(
            go.Histogram(
                x=scores,
                nbinsx=50
            )
        )

        fig.add_vline(
            x=thr,
            line_dash="dash",
            line_color=ALERT_COLOR,
            annotation_text="Threshold"
        )

        fig.update_layout(
            xaxis_title="Fraud probability",
            yaxis_title="Transactions",
            yaxis_type="log"
        )

        st.plotly_chart(
            style(fig, 300, legend=False),
            width="stretch"
        )

        # ====================================================
        # RANKED TRANSACTIONS
        # ====================================================

        id_col = (
            "TransactionID"
            if "TransactionID" in raw.columns
            else None
        )

        out = pd.DataFrame({
            "row": np.arange(len(scores))
        })

        if id_col:

            out["TransactionID"] = raw[id_col].to_numpy()

        out["score"] = scores

        out["percentile"] = (
            pd.Series(scores).rank(pct=True).to_numpy()
        )

        out["flagged"] = flag

        if "isFraud" in raw.columns:

            out["actual_fraud"] = raw["isFraud"].to_numpy()

        for c in (
            "TransactionAmt",
            "ProductCD",
            "P_emaildomain"
        ):

            if c in raw.columns:

                out[c] = raw[c].to_numpy()

        ranked = out.sort_values(
            "score",
            ascending=False
        )

        st.subheader("Highest-risk transactions")

        st.dataframe(
            ranked.head(200),
            hide_index=True,
            width="stretch",
            column_config={
                "score": st.column_config.ProgressColumn(
                    "Fraud score",
                    format="%.4f",
                    min_value=0,
                    max_value=1
                ),
                "percentile": st.column_config.NumberColumn(
                    format="percent"
                )
            }
        )

        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.subheader("Export predictions")

        d1, d2 = st.columns(2)

        d1.download_button(
            "Download all scores",
            out.to_csv(index=False).encode("utf-8"),
            "fraud_scores.csv",
            "text/csv",
            use_container_width=True
        )

        d2.download_button(
            "Download flagged transactions",
            out[out["flagged"]].to_csv(
                index=False
            ).encode("utf-8"),
            "flagged_transactions.csv",
            "text/csv",
            use_container_width=True
        )

        # ====================================================
        # EXPLANATIONS
        # ====================================================

        st.subheader("Explain a transaction")

        labels = ranked.head(200)

        pick = st.selectbox(
            "Choose one of the 200 highest-risk transactions",
            labels["row"].tolist(),
            format_func=lambda r: (
                f"{raw.iloc[r][id_col]} "
                if id_col
                else f"Row {r} "
            )
            + f"· Score {scores[r]:.4f}"
            + (" · FLAGGED" if flag[r] else "")
        )

        show_explanation(
            X.iloc[[pick]],
            scores[pick],
            f"batch_{pick}"
        )

    else:

        st.info(
            "Choose a dataset and click Load dataset to begin."
        )


# ============================================================
# SINGLE TRANSACTION SCORING
# ============================================================

with tab_single:

    st.subheader("Score an individual transaction")

    st.markdown(
        "Enter transaction details. Fields not provided "
        "will be treated as missing."
    )

    names = set(meta["feature_names"])
    levels = meta["category_levels"]

    with st.form("single"):

        a, b, c = st.columns(3)

        amount = a.number_input(
            "TransactionAmt",
            0.0,
            1_000_000.0,
            59.0,
            1.0
        )

        day = b.number_input(
            "Day since dataset start",
            0,
            1000,
            200,
            1
        )

        hour = c.slider(
            "Hour of day",
            0,
            23,
            14
        )

        vals = {}

        cat_inputs = [
            name
            for name in (
                "ProductCD",
                "card4",
                "card6",
                "P_emaildomain",
                "DeviceType"
            )
            if name in levels
        ]

        cols = st.columns(
            max(1, len(cat_inputs))
        )

        for col, name in zip(cols, cat_inputs):

            vals[name] = col.selectbox(
                name,
                ["(missing)"] + list(levels[name])
            )

        num_inputs = [
            name
            for name in (
                "card1",
                "card2",
                "card3",
                "card5",
                "addr1",
                "D1",
                "C1",
                "C2"
            )
            if name in names
        ]

        cols = st.columns(4)

        for i, name in enumerate(num_inputs):

            default = {
                "card1": 9500.0,
                "card2": 400.0,
                "card3": 150.0,
                "card5": 226.0,
                "addr1": 300.0,
                "D1": 20.0,
                "C1": 1.0,
                "C2": 1.0
            }.get(name, 0.0)

            use = cols[i % 4].text_input(
                name,
                value=str(default),
                help="Leave empty for missing"
            )

            vals[name] = (
                pd.to_numeric(use, errors="coerce")
                if use.strip()
                else np.nan
            )

        go_ = st.form_submit_button(
            "Score transaction",
            type="primary"
        )

    if go_:

        row = {
            "TransactionDT": day * 86400 + hour * 3600,
            "TransactionAmt": amount
        }

        for k, v in vals.items():

            row[k] = (
                np.nan
                if v == "(missing)"
                else v
            )

        df_one = pd.DataFrame([row])

        try:

            s, X1, info1 = score_frame(
                art,
                df_one,
                "new"
            )

            thr1 = (
                threshold_for(s)
                if rule != "Top X% of this batch"
                else float(rules[list(rules)[0]])
            )

            verdict = (
                "FLAG"
                if s[0] >= thr1
                else "Pass"
            )

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Fraud probability",
                f"{s[0]:.4f}"
            )

            m2.metric(
                "Decision",
                verdict,
                f"Threshold {thr1:.4f}",
                delta_color="off"
            )

            comparison_col = "lightgbm_probability"

            if comparison_col in art.preds.columns:

                percentile = (
                    art.preds[comparison_col] >= s[0]
                ).mean()

                comparison = f"Top {100 * percentile:.2f}%"

            else:

                comparison = "Unavailable"

            m3.metric(
                "Compared with holdout",
                comparison
            )

            show_explanation(
                X1,
                s[0],
                "single"
            )

        except Exception as e:

            st.error(
                f"Transaction scoring failed: {e}"
            )

            st.exception(e)

    st.caption(
        "A manually entered transaction has no client history "
        "unless its client identifiers match stored reference data."
    )
