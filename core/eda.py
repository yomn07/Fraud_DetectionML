"""EDA summaries (aggregates only: no raw rows are stored). Used by the notebook export and by the app's live EDA.

build_eda(df) works on the labelled training frame (the notebook's `train`, which already has TransactionDT and
isFraud). Everything is plain python / pandas so it can be pasted into the notebook.
"""
import gzip
import json
import os

import numpy as np
import pandas as pd

TARGET = "isFraud"
NEVER = {TARGET, "TransactionID", "TransactionDT"}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _recs(df):
    return _clean(df.to_dict("records"))


def build_eda(df, client=None, max_cat_levels=15, n_bins=20):
    """df: labelled frame with isFraud, TransactionDT, TransactionAmt.
    client: optional DataFrame aligned to df with uid_count, uid_prior_count, amt_to_uid_mean."""
    y = df[TARGET].astype(int)
    n = len(df)
    out = {"n_rows": int(n), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean()), "n_columns": int(df.shape[1])}

    # --- target and amount
    amt = pd.to_numeric(df["TransactionAmt"], errors="coerce")
    edges = np.logspace(np.log10(max(amt.min(), 0.1)), np.log10(amt.max()), 41)
    hist = {}
    for cls in (0, 1):
        counts, _ = np.histogram(amt[y == cls].dropna(), bins=edges)
        hist[str(cls)] = counts.tolist()
    out["amount"] = {
        "edges": edges.tolist(), "hist": hist,
        "summary": _recs(amt.groupby(y).agg(["count", "mean", "median", "std", "min", "max"]).reset_index().rename(columns={TARGET: "class"})),
        "quantiles": {str(c): amt[y == c].quantile([.01, .05, .25, .5, .75, .95, .99]).tolist() for c in (0, 1)},
    }

    # --- time
    day = (pd.to_numeric(df["TransactionDT"], errors="coerce") // 86400).astype("Int64")
    hour = ((pd.to_numeric(df["TransactionDT"], errors="coerce") // 3600) % 24).astype("Int64")
    out["by_day"] = _recs(pd.DataFrame({"day": day, "y": y}).groupby("day")["y"].agg(transactions="size", fraud_rate="mean").reset_index())
    out["by_hour"] = _recs(pd.DataFrame({"hour": hour, "y": y}).groupby("hour")["y"].agg(transactions="size", fraud_rate="mean").reset_index())

    # --- missingness
    miss = df.drop(columns=[c for c in NEVER if c in df.columns]).isna().mean().sort_values(ascending=False)
    rows = []
    for col, share in miss.items():
        r = {"feature": col, "missing_share": float(share)}
        if 0.001 < share < 0.999:
            m = df[col].isna()
            r["fraud_rate_if_missing"] = float(y[m].mean())
            r["fraud_rate_if_present"] = float(y[~m].mean())
        rows.append(r)
    out["missing"] = _clean(rows)

    # --- categorical-like columns
    cat_like = [c for c in df.columns if c not in NEVER and (df[c].dtype == object or str(df[c].dtype) in ("category", "string", "str")
                                                           or (pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) <= 12))]
    cats = {}
    for c in cat_like:
        s = df[c].astype("object").where(df[c].notna(), "(missing)").astype(str)
        g = pd.DataFrame({"level": s, "y": y}).groupby("level")["y"].agg(count="size", fraud_cases="sum", fraud_rate="mean")
        g = g.sort_values("count", ascending=False)
        top = g.head(max_cat_levels).reset_index()
        rest = g.iloc[max_cat_levels:]
        if len(rest):
            top = pd.concat([top, pd.DataFrame([{"level": f"other ({len(rest)} levels)", "count": int(rest["count"].sum()),
                                                 "fraud_cases": int(rest["fraud_cases"].sum()),
                                                 "fraud_rate": float(rest["fraud_cases"].sum() / rest["count"].sum())}])])
        cats[c] = {"levels": int(len(g)), "table": _recs(top)}
    out["categorical"] = cats

    # --- numeric columns: quantile bins with fraud rate
    num_cols = [c for c in df.columns if c not in NEVER and c not in cat_like and pd.api.types.is_numeric_dtype(df[c])]
    nums, corr_rows = {}, []
    ys = y.to_numpy()
    for c in num_cols:
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ok = v.notna().to_numpy()
        if ok.sum() < 50:
            continue
        vv = v.to_numpy()[ok]
        qs = np.unique(np.quantile(vv, np.linspace(0, 1, n_bins + 1)))
        if len(qs) < 3:
            continue
        b = np.clip(np.searchsorted(qs[1:-1], vv, side="right"), 0, len(qs) - 2)
        cnt = np.bincount(b, minlength=len(qs) - 1)
        fr = np.bincount(b, weights=ys[ok], minlength=len(qs) - 1)
        bins = [{"left": float(qs[i]), "right": float(qs[i + 1]), "count": int(cnt[i]),
                 "fraud_rate": float(fr[i] / cnt[i]) if cnt[i] else None} for i in range(len(qs) - 1)]
        nums[c] = {"bins": bins, "missing_share": float(1 - ok.mean()),
                   "mean_legit": float(np.mean(vv[ys[ok] == 0])), "mean_fraud": float(np.mean(vv[ys[ok] == 1])) if (ys[ok] == 1).any() else None,
                   "median_legit": float(np.median(vv[ys[ok] == 0])), "median_fraud": float(np.median(vv[ys[ok] == 1])) if (ys[ok] == 1).any() else None}
        if vv.std() > 0:
            corr_rows.append({"feature": c, "corr_with_fraud": float(np.corrcoef(vv, ys[ok])[0, 1])})
    out["numeric"] = _clean(nums)
    corr = pd.DataFrame(corr_rows)
    if len(corr):
        corr["abs"] = corr["corr_with_fraud"].abs()
        corr = corr.sort_values("abs", ascending=False)
        out["target_correlation"] = _recs(corr.head(50)[["feature", "corr_with_fraud"]])
        top = corr.head(25)["feature"].tolist()
        sample = df[top].apply(pd.to_numeric, errors="coerce").sample(min(100_000, n), random_state=0)
        out["correlation_matrix"] = {"features": top, "values": _clean(sample.corr().fillna(0).round(3).to_numpy().tolist())}

    # --- client structure
    if client is not None:
        c = client.reset_index(drop=True)
        rows = []
        def bucket(series, bins, labels, name):
            b = pd.cut(series, bins=bins, labels=labels, right=True)
            g = pd.DataFrame({"b": b, "y": y.to_numpy()}).groupby("b", observed=True)["y"].agg(transactions="size", fraud_rate="mean").reset_index()
            g["feature"], g["bucket"] = name, g["b"].astype(str)
            return g[["feature", "bucket", "transactions", "fraud_rate"]]
        parts = [bucket(c["uid_count"], [0, 1, 2, 5, 10, 30, np.inf], ["1", "2", "3-5", "6-10", "11-30", "31+"], "uid_count"),
                 bucket(c["uid_prior_count"], [-1, 0, 1, 4, 10, np.inf], ["0 (first seen)", "1", "2-4", "5-10", "11+"], "uid_prior_count")]
        if "amt_to_uid_mean" in c:
            parts.append(bucket(c["amt_to_uid_mean"], [-np.inf, 0.25, 0.75, 1.25, 2, 4, np.inf],
                                ["<0.25x", "0.25-0.75x", "0.75-1.25x", "1.25-2x", "2-4x", ">4x"], "amount vs client average"))
        out["client"] = _recs(pd.concat(parts))
    return _clean(out)


def write_eda(out_dir, eda):
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, "eda.json.gz"), "wt") as f:
        json.dump(eda, f)
    print("EDA written:", round(os.path.getsize(os.path.join(out_dir, "eda.json.gz")) / 1e6, 2), "MB,",
          len(eda.get("numeric", {})), "numeric and", len(eda.get("categorical", {})), "categorical columns")
