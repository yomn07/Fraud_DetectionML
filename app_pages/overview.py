import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.artifacts import MODEL_LABELS, require_artifacts
from core.metrics import summary
from core.ui import ALERT_COLOR, NEW_COLOR, OLD_COLOR, pct, style

art = require_artifacts()
meta, preds = art.meta, art.preds
y = preds["y_true"].to_numpy()

st.title("🛡️ Fraud Detection Studio")
st.caption("IEEE-CIS credit-card fraud · chronological holdout evaluation · LightGBM with client-ID features")


@st.cache_data(show_spinner=False)
def model_summary(_art, key):
    p = _art.preds[key].to_numpy()
    return summary(_art.preds["y_true"].to_numpy(), p)


rows = []
for key in art.models:
    rows.append({"Model": MODEL_LABELS[key], "key": key, **model_summary(art, key)})
table = pd.DataFrame(rows)

best = table.set_index("key").loc["p_v2"] if "p_v2" in table["key"].values else table.iloc[0]
base = table.set_index("key").loc["p_baseline"] if "p_baseline" in table["key"].values else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Holdout average precision", f"{best['AP']:.3f}",
          None if base is None else f"{best['AP'] - base['AP']:+.3f} vs baseline")
c2.metric("Holdout ROC-AUC", f"{best['AUC']:.3f}", None if base is None else f"{best['AUC'] - base['AUC']:+.3f}")
c3.metric("Fraud found with 5% alerts", pct(best["Recall @5%"]),
          None if base is None else f"{100 * (best['Recall @5%'] - base['Recall @5%']):+.1f} pts")
c4.metric("Holdout transactions", f"{meta['n_holdout']:,}", f"{pct(meta['holdout_fraud_rate'], 2)} fraud", delta_color="off")

st.subheader("How the models compare")
left, right = st.columns([3, 2])
with left:
    show = table.drop(columns="key").copy()
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "AUC": st.column_config.NumberColumn(format="%.4f"),
            "AP": st.column_config.NumberColumn("Avg. precision", format="%.4f"),
            "Recall @1%": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
            "Recall @5%": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
        },
    )
    st.caption("All numbers are computed live from the holdout predictions: the last 15% of transactions in time, "
               "never used for training, early stopping or threshold selection.")
with right:
    fig = go.Figure(go.Bar(
        x=table["AP"], y=table["Model"], orientation="h",
        marker_color=[NEW_COLOR if k == "p_v2" else OLD_COLOR for k in table["key"]],
        text=[f"{v:.3f}" for v in table["AP"]], textposition="outside"))
    fig.update_layout(title="Holdout average precision", yaxis=dict(autorange="reversed"), xaxis_range=[0, max(table["AP"]) * 1.2])
    st.plotly_chart(style(fig, height=300, legend=False), width="stretch")

st.subheader("What was tried")
exp = meta.get("experiments") or {}
if exp and any(v for v in exp.values()):
    labels = {"smote": "SMOTE-NC oversampling", "client_avg": "Per-client score averaging",
              "time_consistency": "Dropping time-inconsistent features", "ensemble": "Three-model ensemble"}
    verdict = {"smote": "", "client_avg": "", "time_consistency": "", "ensemble": ""}
    for k, v in exp.items():
        if v:
            with st.expander(labels.get(k, k), expanded=False):
                st.dataframe(pd.DataFrame(v), hide_index=True, width="stretch")
else:
    st.info("No experiment tables were exported with these artifacts.")

st.subheader("How a transaction is scored")
st.markdown(
    """
1. **Raw transaction** (+ optional identity data) is aligned to the training columns; categoricals use the training levels.
2. **Client-ID features** are rebuilt from stored reference statistics: how many transactions share this pseudo-client
   (`card1/2/3/5 + addr1 + first-seen day`), their average amount, and how unusual this amount is for them.
3. **LightGBM** returns a fraud probability; **thresholds** (F2, cost-optimal, alert-capped) turn it into a flag.
4. **Per-transaction explanation** shows which features pushed the score up or down (native LightGBM contributions).
"""
)
try:
    st.page_link("app_pages/scoring.py", label="Try it on your own transactions", icon=":material/policy:")
except Exception:   # page not registered (e.g. page file run on its own)
    pass
