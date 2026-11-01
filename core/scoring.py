"""Scoring and explanation on top of the exported LightGBM booster (no sklearn wrapper needed)."""
import numpy as np
import pandas as pd

from core.features import prepare_features

CHUNK = 50_000


def score_frame(art, raw, mode="new"):
    X, info = prepare_features(raw, art.meta, art.ref, mode)
    out = np.empty(len(X), dtype="float64")
    for i in range(0, len(X), CHUNK):
        out[i:i + CHUNK] = art.booster.predict(X.iloc[i:i + CHUNK])
    return out, X, info


def explain_row(art, X_row):
    """Native LightGBM contributions (log-odds). Returns (table sorted by |contribution|, base_value, logit)."""
    contrib = art.booster.predict(X_row, pred_contrib=True)[0]
    names = list(X_row.columns)
    table = pd.DataFrame({
        "feature": names,
        "value": [X_row.iloc[0][c] for c in names],
        "contribution": contrib[:-1],
    })
    table["value"] = table["value"].map(lambda v: "missing" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)))
    table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    base = float(contrib[-1])
    return table, base, base + float(contrib[:-1].sum())


def decision_thresholds(meta):
    t = meta["thresholds"]
    out = {"F2-optimal (validation)": t["f2"]}
    if t.get("cost") is not None:
        out["Cost-optimal (validation)"] = t["cost"]
    out["Alert-capped cost-optimal (validation)"] = t["capped"]
    return out


def reason_codes(table, k=5, min_k=3):
    """The 3-5 features that pushed the score up most ("reason codes" for an analyst)."""
    pos = table[table["contribution"] > 0].head(k)
    if len(pos) < min_k:
        pos = table.head(min_k)
    return [{"code": f"R{i + 1}", "feature": r.feature, "value": r.value, "effect": float(r.contribution)}
            for i, r in enumerate(pos.itertuples())]


def waterfall_data(table, base, n=9):
    """Rows for a SHAP waterfall: base value, the n largest contributions, everything else, final log-odds."""
    top = table.head(n)
    rest = float(table["contribution"].iloc[n:].sum())
    labels = ["Average (base value)"] + [f"{f} = {v}" for f, v in zip(top["feature"], top["value"])]
    values = [base] + top["contribution"].tolist()
    measure = ["absolute"] + ["relative"] * len(top)
    if len(table) > n:
        labels.append(f"{len(table) - n} other features")
        values.append(rest)
        measure.append("relative")
    labels.append("Final log-odds")
    values.append(base + float(table["contribution"].sum()))
    measure.append("total")
    return labels, values, measure
