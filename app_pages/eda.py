
from pathlib import Path
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Explorer | Fraud Detection",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Data Explorer")
st.caption(
    "Explore IEEE-CIS transaction and identity data, "
    "including fraud patterns, missing values and transaction behavior."
)

MAX_ROWS = 300_000

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


# ============================================================
# KAGGLE DATASET DOWNLOAD
# ============================================================

def is_valid_csv(path):
    """Check whether a file exists and is not a Git LFS pointer."""

    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with open(path, "rb") as file:
            first_line = file.readline(300).decode(
                "utf-8", errors="ignore"
            )

        return "git-lfs.github.com/spec/v1" not in first_line

    except OSError:
        return False


def ensure_kaggle_dataset(dataset_type):
    """
    Download missing IEEE-CIS CSV files from Kaggle.

    Existing valid CSV files are kept.
    Git LFS pointer files are removed and replaced.
    """

    if dataset_type not in ("train", "test"):
        raise ValueError("Dataset type must be train or test.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    filenames = [
        f"{dataset_type}_transaction.csv",
        f"{dataset_type}_identity.csv",
    ]

    # Download only files that are missing or invalid.
    missing_files = []

    for filename in filenames:
        csv_path = DATA_DIR / filename

        if is_valid_csv(csv_path):
            continue

        if csv_path.exists():
            csv_path.unlink()

        missing_files.append(filename)

    if not missing_files:
        return

    # Read Kaggle credentials from Streamlit Secrets.
    try:
        username = st.secrets["kaggle"]["username"]
        key = st.secrets["kaggle"]["key"]

    except Exception:
        raise RuntimeError(
            "Kaggle credentials are missing. "
            "Add [kaggle], username and key in "
            "Streamlit Cloud Settings > Secrets."
        )

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = str(username)
    env["KAGGLE_KEY"] = str(key)

    for filename in missing_files:

        st.info(f"Downloading {filename} from Kaggle...")

        zip_path = DATA_DIR / f"{filename}.zip"
        csv_path = DATA_DIR / filename

        command = [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            "ieee-fraud-detection",
            "-f",
            filename,
            "-p",
            str(DATA_DIR),
        ]

        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Kaggle download failed for {filename}.\n"
                f"{result.stderr[-3000:]}"
            )

        if not zip_path.exists():
            raise RuntimeError(
                f"Kaggle did not create the expected file: "
                f"{zip_path.name}. "
                f"Command output: {result.stdout[-1000:]}"
            )

        # Extract the CSV.
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:

                expected_name = Path(filename).name

                matching_files = [
                    name for name in archive.namelist()
                    if Path(name).name == expected_name
                ]

                if not matching_files:
                    raise RuntimeError(
                        f"{expected_name} was not found "
                        "inside the downloaded ZIP archive."
                    )

                with archive.open(matching_files[0]) as source:
                    with open(csv_path, "wb") as destination:
                        while True:
                            chunk = source.read(1024 * 1024)

                            if not chunk:
                                break

                            destination.write(chunk)

        finally:
            if zip_path.exists():
                zip_path.unlink()

        if not is_valid_csv(csv_path):
            raise RuntimeError(
                f"The downloaded file {filename} is invalid."
            )

        st.success(f"{filename} downloaded successfully.")


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def normalize_identity_columns(df):
    """Normalize identity column names: id-01 -> id_01."""

    df = df.copy()

    df.columns = [
        col.replace("id-", "id_") if isinstance(col, str) else col
        for col in df.columns
    ]

    return df


def read_csv_file(file):
    """Read a CSV file and normalize its column names."""

    df = pd.read_csv(
        file,
        low_memory=False,
        encoding="utf-8-sig",
    )

    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    for col in df.columns:
        if col.lower().replace("_", "").replace(" ", "") == "transactionid":
            df = df.rename(columns={col: "TransactionID"})
            break

    return df


def merge_transaction_identity(transaction_df, identity_df):
    """Merge transaction and identity data using TransactionID."""

    transaction_df = transaction_df.copy()
    identity_df = normalize_identity_columns(identity_df.copy())

    if "TransactionID" not in transaction_df.columns:
        raise ValueError(
            "TransactionID is missing from the transaction file."
        )

    if "TransactionID" not in identity_df.columns:
        raise ValueError(
            "TransactionID is missing from the identity file."
        )

    if identity_df["TransactionID"].duplicated().any():
        raise ValueError(
            "Duplicate TransactionID values were found in identity data."
        )

    duplicated_columns = (
        set(transaction_df.columns)
        & set(identity_df.columns)
    ) - {"TransactionID"}

    if duplicated_columns:
        raise ValueError(
            "Duplicate columns found: "
            f"{sorted(duplicated_columns)}"
        )

    return transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )


def load_local_dataset(dataset_type):
    """Load and merge a local or Kaggle-downloaded dataset."""

    ensure_kaggle_dataset(dataset_type)

    transaction_path = DATA_DIR / f"{dataset_type}_transaction.csv"
    identity_path = DATA_DIR / f"{dataset_type}_identity.csv"

    if not is_valid_csv(transaction_path):
        raise FileNotFoundError(
            f"Invalid transaction file: {transaction_path}"
        )

    if not is_valid_csv(identity_path):
        raise FileNotFoundError(
            f"Invalid identity file: {identity_path}"
        )

    identity_df = read_csv_file(identity_path)

    if "TransactionID" not in transaction_df.columns:
        raise ValueError(
            "TransactionID missing from transaction data. "
            f"Columns found: {transaction_df.columns.tolist()[:20]}"
        )

    if "TransactionID" not in identity_df.columns:
        raise ValueError(
            "TransactionID missing from identity data. "
            f"Columns found: {identity_df.columns.tolist()[:20]}"
        )

    return merge_transaction_identity(
        transaction_df,
        identity_df,
    )


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def prepare_dataframe(df):
    """Prepare a copy of the dataset for analysis."""

    df = df.copy()

    for col in ["TransactionAmt", "TransactionDT", "isFraud"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


def calculate_summary(df):

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": missing_cells,
        "missing_percentage": (
            100 * missing_cells / total_cells
            if total_cells > 0 else 0
        ),
    }

    if "isFraud" in df.columns:

        labels = pd.to_numeric(
            df["isFraud"],
            errors="coerce",
        ).dropna()

        if not labels.empty:
            summary["fraud_count"] = int((labels == 1).sum())
            summary["normal_count"] = int((labels == 0).sum())
            summary["fraud_rate"] = float((labels == 1).mean())

    return summary


def numeric_columns(df):

    excluded = {"TransactionID", "isFraud"}

    return [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col not in excluded
    ]


def categorical_columns(df):

    return [
        col
        for col in df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns
        if df[col].nunique(dropna=True) <= 100
    ]


# ============================================================
# PLOTS
# ============================================================

def plot_fraud_distribution(df):

    if "isFraud" not in df.columns:
        st.info("Fraud labels are not available in this dataset.")
        return

    labels = pd.to_numeric(
        df["isFraud"],
        errors="coerce",
    ).dropna()

    if labels.empty:
        st.warning("No valid fraud labels were found.")
        return

    counts = labels.value_counts().sort_index()

    chart_df = pd.DataFrame({
        "Transaction type": [
            "Normal" if x == 0 else "Fraud"
            for x in counts.index
        ],
        "Count": counts.values,
    })

    fig = px.pie(
        chart_df,
        names="Transaction type",
        values="Count",
        title="Normal vs fraudulent transactions",
        hole=0.45,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_transaction_amount(df):

    if "TransactionAmt" not in df.columns:
        st.info("TransactionAmt is not available.")
        return

    amounts = pd.to_numeric(
        df["TransactionAmt"],
        errors="coerce",
    ).dropna()

    if amounts.empty:
        st.warning("No valid transaction amounts.")
        return

    fig = px.histogram(
        x=amounts.clip(upper=amounts.quantile(0.99)),
        nbins=60,
        title="Transaction amount distribution",
        labels={
            "x": "Transaction amount",
            "y": "Number of transactions",
        },
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_amount_by_fraud(df):

    required = {"TransactionAmt", "isFraud"}

    if not required.issubset(df.columns):
        st.info("TransactionAmt or isFraud is missing.")
        return

    plot_df = df[
        ["TransactionAmt", "isFraud"]
    ].copy()

    plot_df["TransactionAmt"] = pd.to_numeric(
        plot_df["TransactionAmt"],
        errors="coerce",
    )

    plot_df = plot_df.dropna()

    if plot_df.empty:
        return

    plot_df["Transaction type"] = plot_df["isFraud"].map({
        0: "Normal",
        1: "Fraud",
    })

    plot_df = plot_df[
        plot_df["Transaction type"].notna()
    ]

    plot_df = plot_df[
        plot_df["TransactionAmt"]
        <= plot_df["TransactionAmt"].quantile(0.99)
    ]

    fig = px.box(
        plot_df,
        x="Transaction type",
        y="TransactionAmt",
        color="Transaction type",
        title="Transaction amounts by class",
        points=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_time_distribution(df):

    if "TransactionDT" not in df.columns:
        st.info("TransactionDT is not available.")
        return

    values = pd.to_numeric(
        df["TransactionDT"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return

    time_df = pd.DataFrame({
        "TransactionDT": values
    })

    time_df["Relative day"] = (
        time_df["TransactionDT"] // 86400
    ).astype(int)

    daily_counts = (
        time_df.groupby("Relative day")
        .size()
        .reset_index(name="Transactions")
    )

    fig = px.line(
        daily_counts,
        x="Relative day",
        y="Transactions",
        markers=True,
        title="Transactions over time",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_categorical_distribution(df, column):

    if column not in df.columns:
        return

    counts = (
        df[column]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(20)
        .reset_index()
    )

    counts.columns = [column, "Count"]

    fig = px.bar(
        counts,
        x="Count",
        y=column,
        orientation="h",
        title=f"Top values of {column}",
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_missing_values(df):

    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    missing = missing[missing > 0].head(30)

    if missing.empty:
        st.success("No missing values were detected.")
        return

    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing percentage"]

    fig = px.bar(
        missing_df,
        x="Missing percentage",
        y="Column",
        orientation="h",
        title="Columns with the most missing values",
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_correlations(df):

    cols = numeric_columns(df)

    if len(cols) < 2:
        st.info("Not enough numeric columns for correlation analysis.")
        return

    selected = st.multiselect(
        "Select numeric features",
        options=cols,
        default=cols[:min(len(cols), 25)],
        key="correlation_features",
    )

    if len(selected) < 2:
        st.warning("Select at least two features.")
        return

    corr = df[selected].corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Numeric feature correlation matrix",
        aspect="auto",
    )

    st.plotly_chart(fig, use_container_width=True)


def show_feature_statistics(df):

    cols = numeric_columns(df)

    if not cols:
        st.info("No numeric columns found.")
        return

    stats = df[cols].describe().T
    stats["missing"] = df[cols].isna().sum()
    stats["missing_pct"] = df[cols].isna().mean() * 100

    st.dataframe(stats, use_container_width=True)


# ============================================================
# SIDEBAR: DATASET SELECTION
# ============================================================

st.sidebar.header("Dataset selection")

dataset_source = st.sidebar.radio(
    "Choose data source",
    [
        "Kaggle training dataset",
        "Kaggle test dataset",
        "Upload my own files",
    ],
)

df = None


# ============================================================
# LOAD KAGGLE TRAINING DATASET
# ============================================================

if dataset_source == "Kaggle training dataset":

    with st.spinner(
        "Checking and loading Kaggle training data..."
    ):

        try:
            df = load_local_dataset("train")

            st.sidebar.success(
                "Training dataset loaded successfully."
            )

        except Exception as error:
            st.error(f"Could not load training data: {error}")
            st.stop()


# ============================================================
# LOAD KAGGLE TEST DATASET
# ============================================================

elif dataset_source == "Kaggle test dataset":

    with st.spinner(
        "Checking and loading Kaggle test data..."
    ):

        try:
            df = load_local_dataset("test")

            st.sidebar.success(
                "Test dataset loaded successfully."
            )

        except Exception as error:
            st.error(f"Could not load test data: {error}")
            st.stop()


# ============================================================
# UPLOAD CUSTOM DATASET
# ============================================================

elif dataset_source == "Upload my own files":

    st.subheader("Upload IEEE-CIS CSV files")

    st.write(
        "Upload the transaction file and optionally "
        "its corresponding identity file."
    )

    transaction_file = st.file_uploader(
        "Transaction CSV",
        type=["csv"],
        key="transaction_upload",
    )

    identity_file = st.file_uploader(
        "Identity CSV (optional)",
        type=["csv"],
        key="identity_upload",
    )

    if transaction_file is not None:

        try:
            transaction_df = read_csv_file(transaction_file)

            if identity_file is not None:

                identity_df = read_csv_file(identity_file)

                df = merge_transaction_identity(
                    transaction_df,
                    identity_df,
                )

                st.success(
                    "Transaction and identity files merged successfully."
                )

            else:
                df = transaction_df

                st.info(
                    "No identity file supplied. "
                    "Only transaction features will be available."
                )

        except Exception as error:
            st.error(f"Could not load your CSV files: {error}")
            st.stop()

    else:
        st.info("Upload a transaction CSV file to start exploring.")


# ============================================================
# DATA VALIDATION AND PREPARATION
# ============================================================

if df is None:
    st.stop()

if len(df) == 0:
    st.error("The selected dataset is empty.")
    st.stop()

if len(df) > MAX_ROWS:

    st.warning(
        f"The dataset contains {len(df):,} rows. "
        f"For interactive analysis, the first {MAX_ROWS:,} rows "
        "will be used."
    )

    df = df.iloc[:MAX_ROWS].copy()

df = prepare_dataframe(df)

st.divider()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("Dataset overview")

summary = calculate_summary(df)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Transactions", f"{summary['rows']:,}")
c2.metric("Features", f"{summary['columns']:,}")
c3.metric("Duplicate rows", f"{summary['duplicates']:,}")
c4.metric("Missing cells", f"{summary['missing_percentage']:.2f}%")


if "fraud_rate" in summary:

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Fraudulent transactions",
        f"{summary['fraud_count']:,}",
    )

    c2.metric(
        "Normal transactions",
        f"{summary['normal_count']:,}",
    )

    c3.metric(
        "Fraud rate",
        f"{summary['fraud_rate'] * 100:.3f}%",
    )

else:

    st.info(
        "This dataset has no usable isFraud labels. "
        "Fraud rates and supervised fraud comparisons are unavailable."
    )


# ============================================================
# RAW DATA PREVIEW
# ============================================================

with st.expander("Preview dataset", expanded=False):

    st.write("First rows")

    st.dataframe(
        df.head(100),
        use_container_width=True,
    )

    st.write("Dataset information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data type": [
            str(df[col].dtype) for col in df.columns
        ],
        "Missing values": [
            int(df[col].isna().sum()) for col in df.columns
        ],
        "Missing percentage": [
            float(df[col].isna().mean() * 100)
            for col in df.columns
        ],
        "Unique values": [
            int(df[col].nunique(dropna=True))
            for col in df.columns
        ],
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download merged dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="merged_ieee_dataset.csv",
        mime="text/csv",
    )


# ============================================================
# INTERACTIVE EXPLORATION TABS
# ============================================================

(
    tab_overview,
    tab_time,
    tab_categories,
    tab_numeric,
    tab_missing,
    tab_correlations,
    tab_identity,
) = st.tabs([
    "Overview",
    "Time analysis",
    "Categories",
    "Numeric features",
    "Missing values",
    "Correlations",
    "Identity features",
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab_overview:

    st.subheader("Fraud distribution")
    plot_fraud_distribution(df)

    st.subheader("Transaction amounts")
    plot_transaction_amount(df)

    st.subheader("Amount comparison by fraud class")
    plot_amount_by_fraud(df)


# ============================================================
# TAB 2: TIME ANALYSIS
# ============================================================

with tab_time:

    st.subheader("Transaction activity over time")
    plot_time_distribution(df)

    if "TransactionDT" in df.columns:

        st.subheader("Transaction time statistics")

        time_stats = pd.to_numeric(
            df["TransactionDT"],
            errors="coerce",
        ).describe()

        st.dataframe(
            time_stats.to_frame("TransactionDT"),
            use_container_width=True,
        )

    if {"TransactionDT", "isFraud"}.issubset(df.columns):

        st.subheader("Fraud distribution over time")

        time_df = df[
            ["TransactionDT", "isFraud"]
        ].copy()

        time_df["TransactionDT"] = pd.to_numeric(
            time_df["TransactionDT"],
            errors="coerce",
        )

        time_df["isFraud"] = pd.to_numeric(
            time_df["isFraud"],
            errors="coerce",
        )

        time_df = time_df.dropna()

        time_df["Relative day"] = (
            time_df["TransactionDT"] // 86400
        ).astype(int)

        daily_fraud = (
            time_df.groupby("Relative day")["isFraud"]
            .mean()
            .reset_index(name="Fraud rate")
        )

        fig = px.line(
            daily_fraud,
            x="Relative day",
            y="Fraud rate",
            markers=True,
            title="Fraud rate over time",
        )

        fig.update_yaxes(tickformat=".1%")

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3: CATEGORICAL FEATURES
# ============================================================

with tab_categories:

    st.subheader("Categorical feature exploration")

    cat_cols = categorical_columns(df)

    if not cat_cols:

        st.info("No suitable categorical features were found.")

    else:

        selected_category = st.selectbox(
            "Choose a categorical feature",
            cat_cols,
        )

        plot_categorical_distribution(
            df,
            selected_category,
        )

        if "isFraud" in df.columns:

            st.subheader(
                f"Fraud rate by {selected_category}"
            )

            category_df = df[
                [selected_category, "isFraud"]
            ].copy()

            category_df["isFraud"] = pd.to_numeric(
                category_df["isFraud"],
                errors="coerce",
            )

            category_df[selected_category] = (
                category_df[selected_category]
                .fillna("Missing")
                .astype(str)
            )

            fraud_by_category = (
                category_df.groupby(selected_category)["isFraud"]
                .agg(["mean", "count"])
                .reset_index()
            )

            fraud_by_category.columns = [
                selected_category,
                "Fraud rate",
                "Transactions",
            ]

            fraud_by_category = fraud_by_category.sort_values(
                "Transactions",
                ascending=False,
            ).head(20)

            fig = px.bar(
                fraud_by_category,
                x=selected_category,
                y="Fraud rate",
                hover_data=["Transactions"],
                title="Fraud rate by category",
            )

            fig.update_yaxes(tickformat=".1%")

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# TAB 4: NUMERIC FEATURES
# ============================================================

with tab_numeric:

    st.subheader("Numeric feature statistics")

    show_feature_statistics(df)

    cols = numeric_columns(df)

    if cols:

        selected_numeric = st.selectbox(
            "Select a numeric feature",
            cols,
        )

        values = pd.to_numeric(
            df[selected_numeric],
            errors="coerce",
        ).dropna()

        if not values.empty:

            fig = px.histogram(
                x=values.clip(
                    upper=values.quantile(0.99)
                ),
                nbins=50,
                title=f"Distribution of {selected_numeric}",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.write(
                values.describe().to_frame(
                    selected_numeric
                )
            )


# ============================================================
# TAB 5: MISSING VALUES
# ============================================================

with tab_missing:

    st.subheader("Missing value analysis")

    plot_missing_values(df)

    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing count": [
            int(df[col].isna().sum())
            for col in df.columns
        ],
        "Missing percentage": [
            float(df[col].isna().mean() * 100)
            for col in df.columns
        ],
    })

    missing_table = missing_table.sort_values(
        "Missing percentage",
        ascending=False,
    )

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download missing values report",
        data=missing_table.to_csv(index=False).encode("utf-8"),
        file_name="missing_values_report.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 6: CORRELATIONS
# ============================================================

with tab_correlations:

    st.subheader("Feature correlation analysis")
    plot_correlations(df)


# ============================================================
# TAB 7: IDENTITY FEATURES
# ============================================================

with tab_identity:

    st.subheader("Identity information")

    identity_features = [
        col for col in df.columns
        if (
            col.startswith("id_")
            or col in ["DeviceType", "DeviceInfo"]
        )
    ]

    if not identity_features:

        st.warning(
            "No identity features found. "
            "Upload the identity CSV file and join it "
            "with the transaction file."
        )

    else:

        st.success(
            f"{len(identity_features)} identity features detected."
        )

        identity_summary = pd.DataFrame({
            "Feature": identity_features,
            "Missing count": [
                int(df[col].isna().sum())
                for col in identity_features
            ],
            "Missing percentage": [
                float(df[col].isna().mean() * 100)
                for col in identity_features
            ],
            "Unique values": [
                int(df[col].nunique(dropna=True))
                for col in identity_features
            ],
        })

        st.dataframe(
            identity_summary.sort_values(
                "Missing percentage",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        selected_identity = st.selectbox(
            "Select an identity feature",
            identity_features,
        )

        plot_categorical_distribution(
            df,
            selected_identity,
        )

        if (
            "isFraud" in df.columns
            and df[selected_identity].nunique(dropna=True) <= 100
        ):

            st.subheader(
                f"Fraud rate by {selected_identity}"
            )

            identity_df = df[
                [selected_identity, "isFraud"]
            ].copy()

            identity_df["isFraud"] = pd.to_numeric(
                identity_df["isFraud"],
                errors="coerce",
            )

            identity_df[selected_identity] = (
                identity_df[selected_identity]
                .fillna("Missing")
                .astype(str)
            )

            fraud_identity = (
                identity_df.groupby(selected_identity)["isFraud"]
                .agg(["mean", "count"])
                .reset_index()
            )

            fraud_identity.columns = [
                selected_identity,
                "Fraud rate",
                "Transactions",
            ]

            fraud_identity = fraud_identity.sort_values(
                "Transactions",
                ascending=False,
            ).head(20)

            fig = px.bar(
                fraud_identity,
                x=selected_identity,
                y="Fraud rate",
                hover_data=["Transactions"],
                title="Fraud rate by identity feature",
            )

            fig.update_yaxes(tickformat=".1%")

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "IEEE-CIS Fraud Detection | ENSI Challenge 2 | "
    "Interactive Data Explorer"
)
