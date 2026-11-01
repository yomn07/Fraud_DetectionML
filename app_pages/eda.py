
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.artifacts import require_artifacts
from core.ui import (
    ALERT_COLOR,
    OLD_COLOR,
    PALETTE,
    feature_group,
    pct,
    style,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.title("Exploratory data analysis")

art = require_artifacts()

DEFAULT_COLS = (
    [
        "isFraud",
        "TransactionDT",
        "TransactionAmt",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
        "addr1",
        "dist1",
    ]
    + [f"M{i}" for i in range(1, 10)]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 6)]
)


# --------------------------------------------------
# EDA COMPUTATION
# --------------------------------------------------

@st.cache_data(
    show_spinner="Computing EDA from your file...",
    max_entries=2
)
def eda_from_source(src_key, _src, cols):

    from core.eda import build_eda

    if hasattr(_src, "seek"):
        _src.seek(0)

    df = pd.read_csv(
        _src,
        usecols=lambda c: c in cols
    )

    required = [
        "isFraud",
        "TransactionDT",
        "TransactionAmt"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    return build_eda(df)


# --------------------------------------------------
# DATA SOURCE SELECTION
# --------------------------------------------------

source = st.radio(
    "Data",
    [
        "Dataset used for the model (exported summaries)",
        "Analyse my own file"
    ],
    horizontal=True
)

eda = art.eda


# --------------------------------------------------
# ANALYSE A CUSTOM DATASET
# --------------------------------------------------

if source.startswith("Analyse"):

    st.markdown(
        """
        Load a **labelled transactions CSV** to perform
        exploratory data analysis.

        Required columns:
        - `isFraud`
        - `TransactionDT`
        - `TransactionAmt`
        """
    )

    # Extra columns
    extra = st.text_input(
        "Extra columns to include (comma-separated)",
        ""
    )

    cols = set(DEFAULT_COLS) | {
        c.strip()
        for c in extra.split(",")
        if c.strip()
    }

    # --------------------------------------------------
    # FIND DATA FOLDER
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    data_folder = project_root / "data"

    # Create data folder if it does not exist
    data_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find CSV files
    csv_files = sorted(
        data_folder.glob("*.csv")
    )

    st.markdown("### Select your dataset")

    if csv_files:

        data_options = [
            "Choose a CSV file..."
        ] + [
            file.name for file in csv_files
        ]

        selected_file = st.selectbox(
            "CSV files found in data/",
            data_options
        )

    else:

        selected_file = "Choose a CSV file..."

        st.info(
            "No CSV files found in the data/ folder. "
            "Add your CSV files there or upload one below."
        )

    # --------------------------------------------------
    # OTHER INPUT METHODS
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        up = st.file_uploader(
            "Or upload a CSV file",
            type=["csv"]
        )

    with col2:

        path = st.text_input(
            "Or enter a file path",
            placeholder="/path/to/train_transaction.csv"
        )

    # --------------------------------------------------
    # SELECT THE DATA SOURCE
    # --------------------------------------------------

    src = None
    key = None

    # Priority 1: uploaded file
    if up is not None:

        src = up

        key = (
            "upload",
            up.name,
            up.size
        )

    # Priority 2: custom path
    elif path.strip():

        custom_path = Path(path.strip())

        if custom_path.is_file():

            src = str(custom_path.resolve())

            key = (
                "path",
                str(custom_path.resolve()),
                custom_path.stat().st_size
            )

        else:

            st.error(
                "File not found on the machine running the app."
            )

    # Priority 3: CSV from data folder
    elif selected_file != "Choose a CSV file...":

        selected_path = data_folder / selected_file

        if selected_path.is_file():

            src = str(selected_path.resolve())

            key = (
                "data",
                selected_file,
                selected_path.stat().st_size
            )

    # --------------------------------------------------
    # COMPUTE EDA
    # --------------------------------------------------

    if src is None:

        st.info(
            "Choose a CSV from data/, upload a file, "
            "or enter a file path."
        )

        st.stop()

    try:

        eda = eda_from_source(
            key + (tuple(sorted(cols)),),
            src,
            frozenset(cols)
        )

    except Exception as e:

        st.error(
            f"Could not analyse the file: {e}"
        )

        st.stop()


# --------------------------------------------------
# EXPORTED MODEL DATASET SUMMARIES
# --------------------------------------------------

elif eda is None:

    st.info(
        "No EDA summaries were exported. "
        "Re-run the notebook export (section 18), "
        "or analyse your own file above."
    )

    st.stop()


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.caption(
    f"Computed from {eda['n_rows']:,} labelled transactions "
    f"and {eda['n_columns']} columns. "
    "Only aggregates are kept, never individual rows."
)


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

tabs = st.tabs(
    [
        "Overview",
        "Time",
        "Categories",
        "Numeric features",
        "Missing values",
        "Correlations",
        "Clients",
    ]
)


# ==================================================
# 1. OVERVIEW
# ==================================================

with tabs[0]:

    a, b, c = st.columns(3)

    a.metric(
        "Transactions",
        f"{eda['n_rows']:,}"
    )

    b.metric(
        "Fraud cases",
        f"{eda['n_fraud']:,}"
    )

    c.metric(
        "Fraud rate",
        pct(eda["fraud_rate"], 2),
        "heavily imbalanced",
        delta_color="off"
    )

    # Transaction amount distribution
    amt = eda["amount"]

    edges = np.array(amt["edges"])

    mid = np.sqrt(
        edges[:-1] * edges[1:]
    )

    fig = go.Figure()

    for cls, name, colr in (
        ("0", "Legitimate", PALETTE[0]),
        ("1", "Fraud", ALERT_COLOR),
    ):

        h = np.array(
            amt["hist"][cls],
            dtype=float
        )

        fig.add_scatter(
            x=mid,
            y=h / max(h.sum(), 1),
            mode="lines",
            fill="tozeroy",
            name=name,
            line_color=colr,
            opacity=0.6
        )

    fig.update_layout(
        title=(
            "Transaction amount by class "
            "(share of class, log x-axis)"
        ),
        xaxis_type="log",
        xaxis_title="TransactionAmt",
        yaxis_title="Share of class"
    )

    st.plotly_chart(
        style(fig, 380),
        width="stretch"
    )

    # Amount summary
    st.dataframe(
        pd.DataFrame(
            amt["summary"]
        ).rename(
            columns={"class": "isFraud"}
        ),
        hide_index=True,
        width="stretch"
    )

    # Amount percentiles
    q = amt["quantiles"]

    qdf = pd.DataFrame(
        {
            "percentile": [
                "1%",
                "5%",
                "25%",
                "50%",
                "75%",
                "95%",
                "99%",
            ],
            "Legitimate": q["0"],
            "Fraud": q["1"],
        }
    )

    with st.expander("Amount percentiles by class"):

        st.dataframe(
            qdf,
            hide_index=True,
            width="stretch"
        )

    st.caption(
        "Fraud and legitimate amounts overlap heavily: "
        "amount alone is a weak signal, which is why "
        "client behaviour matters."
    )


# ==================================================
# 2. TIME ANALYSIS
# ==================================================

with tabs[1]:

    d = pd.DataFrame(
        eda["by_day"]
    )

    fig = go.Figure()

    fig.add_bar(
        x=d["day"],
        y=d["transactions"],
        name="Transactions",
        marker_color=OLD_COLOR,
        opacity=0.5,
        yaxis="y"
    )

    fig.add_scatter(
        x=d["day"],
        y=d["fraud_rate"],
        name="Fraud rate",
        line_color=ALERT_COLOR,
        yaxis="y2"
    )

    fig.update_layout(
        title="Volume and fraud rate by day",
        xaxis_title="Day since dataset start",
        yaxis=dict(
            title="Transactions"
        ),
        yaxis2=dict(
            title="Fraud rate",
            overlaying="y",
            side="right",
            tickformat=".1%"
        )
    )

    st.plotly_chart(
        style(fig, 380),
        width="stretch"
    )

    # Hour analysis
    h = pd.DataFrame(
        eda["by_hour"]
    )

    fig = go.Figure()

    fig.add_bar(
        x=h["hour"],
        y=h["transactions"],
        name="Transactions",
        marker_color=OLD_COLOR,
        opacity=0.5,
        yaxis="y"
    )

    fig.add_scatter(
        x=h["hour"],
        y=h["fraud_rate"],
        name="Fraud rate",
        line_color=ALERT_COLOR,
        yaxis="y2"
    )

    fig.update_layout(
        title="By hour of day",
        xaxis_title=(
            "Hour relative to dataset time origin"
        ),
        yaxis=dict(
            title="Transactions"
        ),
        yaxis2=dict(
            title="Fraud rate",
            overlaying="y",
            side="right",
            tickformat=".1%"
        )
    )

    st.plotly_chart(
        style(fig, 340),
        width="stretch"
    )


# ==================================================
# 3. CATEGORICAL FEATURES
# ==================================================

with tabs[2]:

    cats = eda["categorical"]

    names = sorted(
        cats,
        key=lambda c: (feature_group(c), c)
    )

    default = (
        names.index("ProductCD")
        if "ProductCD" in names
        else 0
    )

    col = st.selectbox(
        "Column",
        names,
        index=default,
        format_func=lambda c: (
            f"{c} ({cats[c]['levels']} levels)"
        )
    )

    t = pd.DataFrame(
        cats[col]["table"]
    )

    fig = go.Figure()

    fig.add_bar(
        x=t["level"],
        y=t["count"],
        name="Transactions",
        marker_color=OLD_COLOR,
        opacity=0.5,
        yaxis="y"
    )

    fig.add_scatter(
        x=t["level"],
        y=t["fraud_rate"],
        name="Fraud rate",
        mode="lines+markers",
        line_color=ALERT_COLOR,
        yaxis="y2"
    )

    fig.add_hline(
        y=eda["fraud_rate"],
        line_dash="dot",
        line_color="#999",
        yref="y2"
    )

    fig.update_layout(
        title=f"{col}: volume and fraud rate by level",
        yaxis=dict(
            title="Transactions",
            type="log"
        ),
        yaxis2=dict(
            title="Fraud rate",
            overlaying="y",
            side="right",
            tickformat=".1%"
        ),
        xaxis_tickangle=-30
    )

    st.plotly_chart(
        style(fig, 420),
        width="stretch"
    )

    st.dataframe(
        t,
        hide_index=True,
        width="stretch",
        column_config={
            "fraud_rate": st.column_config.NumberColumn(
                format="percent"
            )
        }
    )

    st.caption(
        "Dotted line = overall fraud rate. "
        "Small levels have noisy rates: check the "
        "transaction count before reading a spike."
    )


# ==================================================
# 4. NUMERIC FEATURES
# ==================================================

with tabs[3]:

    nums = eda["numeric"]

    names = sorted(
        nums,
        key=lambda c: (feature_group(c), c)
    )

    default = (
        names.index("TransactionAmt")
        if "TransactionAmt" in names
        else 0
    )

    col = st.selectbox(
        "Feature",
        names,
        index=default,
        key="numcol"
    )

    info = nums[col]

    bins = pd.DataFrame(
        info["bins"]
    )

    bins["label"] = [
        f"{l:.3g} to {r:.3g}"
        for l, r in zip(
            bins["left"],
            bins["right"]
        )
    ]

    fig = go.Figure()

    fig.add_bar(
        x=bins["label"],
        y=bins["fraud_rate"],
        marker_color=PALETTE[0],
        name="Fraud rate"
    )

    fig.add_hline(
        y=eda["fraud_rate"],
        line_dash="dot",
        line_color="#999",
        annotation_text="overall"
    )

    fig.update_layout(
        title=(
            f"{col}: fraud rate by quantile bin "
            "(about equal numbers of transactions per bin)"
        ),
        yaxis_tickformat=".1%",
        xaxis_tickangle=-40,
        yaxis_title="Fraud rate"
    )

    st.plotly_chart(
        style(fig, 420, legend=False),
        width="stretch"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Missing",
        pct(info["missing_share"])
    )

    m2.metric(
        "Median (legitimate)",
        f"{info['median_legit']:.4g}"
    )

    m3.metric(
        "Median (fraud)",
        (
            "n/a"
            if info["median_fraud"] is None
            else f"{info['median_fraud']:.4g}"
        )
    )

    m4.metric(
        "Mean fraud / legit",
        (
            "n/a"
            if not info["mean_fraud"]
            or not info["mean_legit"]
            else f"{info['mean_fraud'] / info['mean_legit']:.2f}x"
        )
    )


# ==================================================
# 5. MISSING VALUES
# ==================================================

with tabs[4]:

    m = pd.DataFrame(
        eda["missing"]
    )

    top = st.slider(
        "Columns to show",
        10,
        80,
        30
    )

    mm = m.head(top).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=mm["missing_share"],
            y=mm["feature"],
            orientation="h",
            marker_color=PALETTE[0]
        )
    )

    fig.update_layout(
        title="Share of missing values",
        xaxis_tickformat=".0%"
    )

    st.plotly_chart(
        style(fig, 90 + 18 * top, legend=False),
        width="stretch"
    )

    both = m.dropna(
        subset=[
            "fraud_rate_if_missing",
            "fraud_rate_if_present"
        ]
    ).copy()

    if len(both):

        both["ratio"] = (
            both["fraud_rate_if_missing"]
            / both["fraud_rate_if_present"].replace(
                0,
                np.nan
            )
        )

        st.markdown(
            "**Is being missing itself informative?** "
            "Columns where the fraud rate differs most "
            "between missing and present:"
        )

        both = both.reindex(
            both["ratio"]
            .sub(1)
            .abs()
            .sort_values(ascending=False)
            .index
        ).head(15)

        st.dataframe(
            both[
                [
                    "feature",
                    "missing_share",
                    "fraud_rate_if_missing",
                    "fraud_rate_if_present",
                    "ratio",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "missing_share": st.column_config.NumberColumn(
                    format="percent"
                ),
                "fraud_rate_if_missing": st.column_config.NumberColumn(
                    format="percent"
                ),
                "fraud_rate_if_present": st.column_config.NumberColumn(
                    format="percent"
                ),
                "ratio": st.column_config.NumberColumn(
                    "missing / present",
                    format="%.2fx"
                ),
            }
        )


# ==================================================
# 6. CORRELATIONS
# ==================================================

with tabs[5]:

    tc = pd.DataFrame(
        eda.get("target_correlation", [])
    )

    if len(tc):

        t = tc.head(25).iloc[::-1]

        fig = go.Figure(
            go.Bar(
                x=t["corr_with_fraud"],
                y=t["feature"],
                orientation="h",
                marker_color=[
                    ALERT_COLOR if v > 0 else PALETTE[0]
                    for v in t["corr_with_fraud"]
                ]
            )
        )

        fig.update_layout(
            title="Linear correlation with isFraud (top 25)"
        )

        st.plotly_chart(
            style(fig, 640, legend=False),
            width="stretch"
        )

    cm = eda.get("correlation_matrix")

    if cm:

        fig = go.Figure(
            go.Heatmap(
                z=cm["values"],
                x=cm["features"],
                y=cm["features"],
                zmin=-1,
                zmax=1,
                colorscale="RdBu_r"
            )
        )

        fig.update_layout(
            title=(
                "Correlation among the 25 features "
                "most correlated with fraud"
            ),
            yaxis=dict(
                autorange="reversed"
            )
        )

        st.plotly_chart(
            style(fig, 620, legend=False),
            width="stretch"
        )

    st.caption(
        "Linear correlation with fraud is small for almost "
        "every column: signal comes from interactions, "
        "which is why tree models win here."
    )


# ==================================================
# 7. CLIENT ANALYSIS
# ==================================================

with tabs[6]:

    cl = pd.DataFrame(
        eda.get("client", [])
    )

    if not len(cl):

        st.info(
            "No client-structure summaries were exported."
        )

    else:

        st.markdown(
            "Why the client-ID features were added: "
            "fraud rate by how a transaction relates "
            "to its pseudo-client."
        )

        for feat in cl["feature"].unique():

            d = cl[
                cl["feature"] == feat
            ]

            fig = go.Figure()

            fig.add_bar(
                x=d["bucket"],
                y=d["transactions"],
                marker_color=OLD_COLOR,
                opacity=0.5,
                name="Transactions",
                yaxis="y"
            )

            fig.add_scatter(
                x=d["bucket"],
                y=d["fraud_rate"],
                mode="lines+markers",
                line_color=ALERT_COLOR,
                name="Fraud rate",
                yaxis="y2"
            )

            fig.update_layout(
                title=feat,
                yaxis=dict(
                    title="Transactions"
                ),
                yaxis2=dict(
                    title="Fraud rate",
                    overlaying="y",
                    side="right",
                    tickformat=".1%"
                )
            )

            st.plotly_chart(
                style(fig, 320),
                width="stretch",
                key=f"client_{feat}"
            )
