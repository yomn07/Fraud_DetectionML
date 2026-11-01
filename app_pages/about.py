import streamlit as st

from core.artifacts import require_artifacts

art = require_artifacts()
meta = art.meta

st.title("Method & limits")

st.subheader("Data and validation")
st.markdown(
    f"""
- **Data:** IEEE-CIS Fraud Detection (Kaggle), transactions joined with identity data.
- **Split:** chronological 70% train / 15% validation / 15% holdout. Early stopping and thresholds use validation only;
  every number on the dashboard pages comes from the untouched holdout ({meta['n_holdout']:,} transactions).
- **Model:** LightGBM with native categorical handling. The model used for scoring is the one trained on the training
  split ({meta.get('best_iteration') or 'n/a'} trees), the same model whose holdout predictions are shown.
- **Created:** {meta.get('created', 'n/a')}
"""
)

st.subheader("Client-ID features")
st.markdown(
    """
A pseudo-client is built from `card1`, `card2`, `card3`, `card5`, `addr1` and the day the card was first seen
(`TransactionDT` in days minus `D1`). Counts and average amounts are then computed per pseudo-client, per card and per
e-mail domain. In training they were computed over all labelled **and** unlabelled rows; the app stores those statistics
as hashed lookup tables (no raw card values are kept) and rebuilds the features at scoring time.
"""
)

st.subheader("Known limits")
st.markdown(
    """
- **Holdout numbers are slightly optimistic.** The count and mean features use label-free statistics from holdout and
  test rows too. Only `uid_prior_count` is strictly causal.
- **Scoring new transactions is an approximation.** For new rows, reference statistics are added to the batch's own
  statistics. That matches training only if new transactions arrive after the reference data.
- **The client ID is approximate.** Several real people can share one pseudo-client, and a missing `D1` weakens it.
  Averaging scores per client helped only marginally in testing.
- **Scores rank risk; they are not calibrated probabilities.** Prefer the validation-chosen thresholds.
- **Thresholds depend on your costs.** The default 1 : 300 cost ratio is an assumption, not a measurement.
- **Not production software.** No authentication, audit trail or drift monitoring. Do not make automated decisions about
  real customers from this demo.
"""
)

st.subheader("Experiments that did not make the cut")
st.markdown(
    """
SMOTE-NC oversampling, per-client score averaging, dropping time-inconsistent features and a LightGBM + XGBoost + CatBoost
blend were all tested against the single LightGBM. See the **Overview** page for their tables when exported with the model.
"""
)
