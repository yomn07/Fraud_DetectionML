import pandas as pd
import streamlit as st

from core.artifacts import require_artifacts

art = require_artifacts()
meta = art.meta

st.title("Method, limits & project checklist")

st.subheader("Project requirements: where each one is")
checklist = [
    ("1 Data", "Load a dataset (upload, local path, or the notebook's automatic Kaggle download)", "Done", "EDA and Score pages; notebook section 2"),
    ("1 Data", "Fraud rate, amount distribution by class, fraud rate over time, by hour and by category", "Done", "EDA"),
    ("2 Features", "Cyclic hour encoding", "Done", "hour sin/cos in the model"),
    ("2 Features", "Day of week", "Not included", "TransactionDT has an anonymised origin, so a weekday would be a guess"),
    ("2 Features", "Behavioural features per client (count, amount vs client average, previous transactions)", "Partly",
     "Pseudo-client counts, amount vs client mean, and a causal count of earlier transactions. No 1 h / 24 h windows, no 30-day average, no time-since-previous"),
    ("2 Features", "Chronological split 60 / 20 / 20", "Deviation", "Split is 70 / 15 / 15 (train / validation / test), still strictly chronological"),
    ("2 Features", "Imbalance strategy on the training set only, justified on validation", "Partly",
     "SMOTE-NC vs. an undersampling control vs. baseline were compared in the notebook, judged mainly on the test-period metrics (validation AUC was shown too); the final model uses no resampling"),
    ("3 Models", "At least one supervised model", "Done", "LightGBM (final), XGBoost, CatBoost, blend"),
    ("3 Models", "Isolation Forest + at least one more unsupervised method, with a comparison table", "Done", "Isolation Forest, LOF, One-Class SVM (approx.), autoencoder; Supervised vs. unsupervised"),
    ("4 Evaluation", "PR-AUC, ROC-AUC, precision@k, recall@k, F2, MCC", "Done", "Threshold & cost, Model performance"),
    ("4 Evaluation", "Overlaid precision-recall curves", "Done", "Threshold & cost, Model performance, Supervised vs. unsupervised"),
    ("4 Evaluation", "Threshold chosen on validation by a business cost (missed fraud amounts + review cost), confirmed on test", "Done", "Threshold & cost (needs the section 18 export)"),
    ("4 Evaluation", "Live sliders for threshold and budget", "Done", "Threshold & cost"),
    ("5 Explainability", "Global SHAP beeswarm", "Done", "Feature insights"),
    ("5 Explainability", "SHAP waterfall and 3-5 reason codes per alert", "Done", "Score transactions"),
    ("Objective", "Expose the model through an API", "Done", "api.py (FastAPI), see README"),
]
cl = pd.DataFrame(checklist, columns=["Module", "Requirement", "Status", "Where / note"])
st.dataframe(cl, hide_index=True, width="stretch", column_config={
    "Status": st.column_config.TextColumn(width="small"), "Requirement": st.column_config.TextColumn(width="large"),
    "Where / note": st.column_config.TextColumn(width="large")})
st.caption("The deviations are stated here on purpose: they come from the notebook that was already run, and are the first things to change if the specification must be followed to the letter.")

st.subheader("Data and validation")
st.markdown(
    f"""
- **Data:** IEEE-CIS Fraud Detection (Kaggle), transactions joined with identity data.
- **Split:** chronological 70% train / 15% validation / 15% test (called *holdout* in some pages). Early stopping and thresholds use validation only;
  every number on the dashboard pages comes from the untouched test period ({meta['n_holdout']:,} transactions).
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
- **Scores rank risk; they are not calibrated probabilities.** Prefer the validation-chosen thresholds.
- **Thresholds depend on your costs.** The default 1 : 300 cost ratio is an assumption, not a measurement.
- **Not production software.** No authentication, audit trail or drift monitoring. Do not make automated decisions about
  real customers from this demo.
"""
)

st.subheader("Experiments that did not make the cut")
st.markdown(
    """
SMOTE-NC oversampling, dropping time-inconsistent features and a LightGBM + XGBoost + CatBoost blend were all tested against the single LightGBM. See the **Overview** page for their tables when exported with the model.
"""
)
