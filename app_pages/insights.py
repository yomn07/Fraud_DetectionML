import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.artifacts import require_artifacts
from core.ui import NEW_COLOR, OLD_COLOR, PALETTE, feature_group, pct, style

art = require_artifacts()
meta = art.meta
new_feats = set(meta["new_features"])

st.title("Feature insights")
imp = pd.DataFrame(meta["importance"])
imp["group"] = imp["feature"].map(lambda f: feature_group(f, new_feats))
imp["is_new"] = imp["feature"].isin(new_feats)

st.caption("Importance is the total split gain of each feature across all LightGBM trees. "
           "It shows what the model relies on, not what causes fraud.")

k1, k2, k3 = st.columns(3)
k1.metric("Features in the model", f"{len(imp):,}")
k2.metric("Share of gain from client-ID features", pct(imp.loc[imp["is_new"], "share"].sum()))
k3.metric("Client-ID features in top 20", f"{int(imp.head(20)['is_new'].sum())} of {len(new_feats)}")

left, right = st.columns([3, 2])
with left:
    n = st.slider("Features to show", 10, 50, 25)
    top = imp.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["share"], y=top["feature"], orientation="h",
                           marker_color=[NEW_COLOR if v else OLD_COLOR for v in top["is_new"]]))
    fig.update_layout(title="Top features by gain share (blue = client-ID features)", xaxis_tickformat=".1%")
    st.plotly_chart(style(fig, 80 + 22 * n, legend=False), width="stretch")
with right:
    g = imp.groupby("group")["share"].sum().sort_values(ascending=False).reset_index()
    fig = px.pie(g, names="group", values="share", hole=0.5, color_discrete_sequence=PALETTE + ["#a3a3a3", "#0ea5e9", "#d946ef", "#22c55e"])
    fig.update_traces(textposition="inside", textinfo="percent")
    fig.update_layout(title="Gain by feature group")
    st.plotly_chart(style(fig, 420), width="stretch")

st.subheader("The client-ID features")
desc = {
    "uid_count": "Transactions sharing this pseudo-client (card fields + address + first-seen day).",
    "uid_amt_mean": "Average amount of that pseudo-client.",
    "uid_amt_std": "Spread of that client's amounts.",
    "amt_to_uid_mean": "This amount divided by the client's average: how unusual is it for them?",
    "uid_prior_count": "How many *earlier* transactions the client had (the only strictly causal one).",
    "card_count": "Transactions on the same card + address.",
    "card_amt_mean": "Average amount on that card.",
    "email_count": "Transactions with the same purchaser e-mail domain.",
    "email_amt_mean": "Average amount for that domain.",
}
nf = imp[imp["is_new"]].copy()
nf["what it measures"] = nf["feature"].map(desc)
nf["rank"] = nf["feature"].map({f: i + 1 for i, f in enumerate(imp["feature"])})
st.dataframe(nf[["rank", "feature", "share", "what it measures"]], hide_index=True, width="stretch",
             column_config={"share": st.column_config.NumberColumn("gain share", format="percent")})

bad = meta.get("time_inconsistent_features") or []
st.subheader("Time-consistency check")
if bad:
    st.markdown(
        "A single-feature model trained on early data and scored on later data revealed these columns find patterns that "
        "**reverse over time** (late AUC below 0.5). They are kept in the model because dropping them did not clearly "
        "improve holdout results, but they are the first candidates to remove if performance drifts.")
    st.write(", ".join(f"`{b}`" for b in bad))
else:
    st.info("No time-consistency results were exported.")
