from pathlib import Path

# ---- client-key helpers (identical to the app's core/keys.py) ----
import numpy as np
import pandas as pd

KEY_KINDS = ("uid", "card", "email")


def _num_str(s):
    v = pd.to_numeric(s, errors="coerce")
    return v.round().astype("Int64").astype("string").fillna("nan").astype(object)


def make_keys(df):
    """df needs TransactionDT, D1, card1, card2, card3, card5, addr1, P_emaildomain."""
    day = np.floor(pd.to_numeric(df["TransactionDT"], errors="coerce") / 86400)
    start_day = _num_str(day - pd.to_numeric(df["D1"], errors="coerce"))
    card = _num_str(df["card1"])
    for c in ("card2", "card3", "card5", "addr1"):
        card = card + "_" + _num_str(df[c])
    uid = card + "_" + start_day
    email = df["P_emaildomain"].astype("string").fillna("nan").astype(object)
    return {"uid": uid, "card": card, "email": email}


def hash_keys(keys):
    return pd.util.hash_pandas_object(pd.Series(np.asarray(keys, dtype=object)), index=False).to_numpy()


def key_stats(h, amount):
    """Per-key count, sum and sum of squares of the transaction amount (float64)."""
    d = pd.DataFrame({"h": h, "a": np.asarray(amount, dtype="float64")})
    d["a2"] = d["a"] ** 2
    g = d.groupby("h")
    return pd.DataFrame({"n": g["a"].size().astype("int64"), "s": g["a"].sum(), "ss": g["a2"].sum()})


def stats_to_features(n, s, ss):
    n = np.asarray(n, dtype="float64")
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    var = np.where(n > 1, (ss - n * mean ** 2) / np.maximum(n - 1, 1), np.nan)
    return mean, np.sqrt(np.maximum(var, 0.0))


# ---- EDA summaries (core/eda.py) ----
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


# ---- exporter ----
import gzip
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from core.keys import KEY_KINDS, hash_keys, key_stats, make_keys, stats_to_features  # noqa: F401
except ImportError:  # when pasted into the notebook, core/keys.py is defined above this code
    pass


def _best_cost_threshold(y, p, c_fp, c_fn):
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    cost = c_fp * fp + c_fn * (ys.sum() - tp)
    i = int(np.argmin(cost))
    return None if c_fn * ys.sum() <= cost[i] else float(ps[i])


def _best_capped_threshold(y, p, c_fp, c_fn, max_rate):
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    k = max(1, int(max_rate * len(ys)))
    tp = np.cumsum(ys)[:k]
    fp = np.arange(1, k + 1) - tp
    cost = c_fp * fp + c_fn * (ys.sum() - tp)
    return float(ps[int(np.argmin(cost))])


def _best_f2_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, p)
    f2 = 5 * prec[:-1] * rec[:-1] / (4 * prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.argmax(f2))])


def _records(df):
    if df is None:
        return None
    return json.loads(df.round(6).to_json(orient="records"))


def export_artifacts(out_dir, *, booster, A, new_feats, feature_names, cat_cols, category_levels,
                     y_valid, p_valid, y_hold, p_hold, day_hold, cost_fp=1, cost_fn=300, max_alert_rate=0.05,
                     tables=None, time_inconsistent=None, is_demo=False, split_note="", best_iteration=None,
                     amount_hold=None, amount_valid=None, p_valid_all=None):
    """A: DataFrame over ALL reference rows (train+valid+holdout+test) with columns
    card1, card2, card3, card5, addr1, D1, TransactionAmt, P_emaildomain and TransactionDT.
    amount_hold / amount_valid: TransactionAmt of those rows (needed for the amount-based cost).
    p_valid_all: optional dict name -> validation probabilities of the other models (same keys as p_hold).
    p_hold: dict name -> holdout probabilities; must contain 'p_v2' (the model in `booster`, trained on train only).
    p_valid: validation probabilities of that same model (used to choose thresholds)."""
    os.makedirs(out_dir, exist_ok=True)
    tables = tables or {}

    # 1) model
    with gzip.open(os.path.join(out_dir, "model_lgbm.txt.gz"), "wt") as f:
        f.write(booster.model_to_string())

    # 2) reference statistics for the client features (same keys / same statistics as in training)
    keys = make_keys(A)
    amt = pd.to_numeric(A["TransactionAmt"], errors="coerce").to_numpy(dtype="float64")
    for kind in KEY_KINDS:
        h = hash_keys(keys[kind])
        st = key_stats(h, amt)
        st.index.name = "h"
        st.reset_index().to_parquet(os.path.join(out_dir, f"ref_{kind}.parquet"), index=False)
        if kind == "uid" and "uid_count" in A.columns:      # parity check against what the model was trained on
            n_lookup = st["n"].reindex(pd.Index(h)).to_numpy()
            mean_lookup = (st["s"] / st["n"]).reindex(pd.Index(h)).to_numpy()
            assert np.array_equal(n_lookup, A["uid_count"].to_numpy()), "uid_count mismatch with training features"
            assert np.allclose(mean_lookup, A["uid_amt_mean"].to_numpy(), rtol=1e-4), "uid_amt_mean mismatch"
            print("Parity check passed: reference tables reproduce uid_count / uid_amt_mean.")

    # 3) holdout predictions
    preds = pd.DataFrame({"y_true": np.asarray(y_hold).astype("int8"), "day": np.asarray(day_hold).astype("int32")})
    if amount_hold is not None:
        preds["amount"] = np.asarray(amount_hold, dtype="float32")
    for name, p in p_hold.items():
        preds[name] = np.asarray(p, dtype="float32")
    preds.to_csv(os.path.join(out_dir, "holdout_predictions.csv.gz"), index=False, compression="gzip")

    # validation predictions: the app chooses thresholds here and confirms them on the holdout
    vpred = pd.DataFrame({"y_true": np.asarray(y_valid).astype("int8")})
    if amount_valid is not None:
        vpred["amount"] = np.asarray(amount_valid, dtype="float32")
    vpred["p_v2"] = np.asarray(p_valid, dtype="float32")
    for name, p in (p_valid_all or {}).items():
        if name != "p_v2":
            vpred[name] = np.asarray(p, dtype="float32")
    vpred.to_csv(os.path.join(out_dir, "valid_predictions.csv.gz"), index=False, compression="gzip")

    # 4) thresholds chosen on VALID (same logic as notebook sections 5 and 5b)
    yv = np.asarray(y_valid).astype(int)
    pv = np.asarray(p_valid)
    thresholds = {
        "f2": _best_f2_threshold(yv, pv),
        "cost": _best_cost_threshold(yv, pv, cost_fp, cost_fn),
        "capped": _best_capped_threshold(yv, pv, cost_fp, cost_fn, max_alert_rate),
    }

    # 5) importance
    gain = booster.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": booster.feature_name(), "gain": gain})
    imp["share"] = imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False)

    meta = {
        "is_demo": bool(is_demo),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_names": list(feature_names),
        "new_features": list(new_feats),
        "categorical_columns": list(cat_cols),
        "category_levels": {c: [str(x) for x in category_levels[c]] for c in cat_cols},
        "thresholds": thresholds,
        "cost": {"fp": cost_fp, "fn": cost_fn, "max_alert_rate": max_alert_rate},
        "best_iteration": best_iteration,
        "split": split_note or "chronological 70/15/15 (train/valid/holdout)",
        "n_holdout": int(len(preds)),
        "holdout_fraud_rate": float(preds["y_true"].mean()),
        "n_reference_rows": int(len(A)),
        "model_table": _records(tables.get("model_table")),
        "experiments": {k: _records(v) for k, v in tables.items() if k != "model_table"},
        "time_inconsistent_features": list(time_inconsistent or []),
        "importance": _records(imp),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    sizes = {fn: round(os.path.getsize(os.path.join(out_dir, fn)) / 1e6, 2) for fn in sorted(os.listdir(out_dir))}
    print("Artifacts written to", out_dir, "(MB):", sizes)
    return meta


# ---- extras ----
import json
import os
import platform

import numpy as np
import pandas as pd

try:
    from core.eda import write_eda
except ImportError:  # pasted into the notebook: write_eda is defined above
    pass


def _records(df):
    return None if df is None else json.loads(pd.DataFrame(df).round(6).to_json(orient="records"))


def write_shap_sample(out_dir, booster, X, top=25):
    """Global-explanation data: SHAP values (log-odds) for the `top` features on a sample of rows, plus each feature's
    value as a percentile within the sample (0-1) for colouring. Raw feature values are not stored."""
    contrib = booster.predict(X, pred_contrib=True)
    names = list(X.columns)
    sv = pd.DataFrame(contrib[:, :-1], columns=names)
    order = sv.abs().mean().sort_values(ascending=False).index[:top]
    out = {}
    for c in order:
        v = X[c]
        if str(v.dtype) == "category":
            pct = pd.Series(v.cat.codes.replace(-1, np.nan)).rank(pct=True)
        else:
            pct = pd.Series(pd.to_numeric(v, errors="coerce").to_numpy()).rank(pct=True)
        out[f"shap:{c}"] = sv[c].to_numpy().astype("float32")
        out[f"pct:{c}"] = pct.fillna(-1).to_numpy().astype("float32")      # -1 = missing
    out["_base"] = contrib[:, -1].astype("float32")
    pd.DataFrame(out).to_csv(os.path.join(out_dir, "shap_sample.csv.gz"), index=False, compression="gzip")


def export_extras(out_dir, *, y_eval, day_eval, sup_on_eval, all_scores, unsup_parts=None, unsup_models=None,
                  unsup_tables=None, unsup_settings=None, eda=None, booster=None, X_shap=None, shap_top=25):
    """sup_on_eval: dict p_xxx -> supervised probabilities on exactly the rows of the unsupervised evaluation sample.
    all_scores: dict detector name -> anomaly scores on the same rows.
    unsup_parts: dict(num_cols, freq_cols, freq_maps, imputer, scaler, pca); unsup_models: dict(if_model, svm_model, ae_model)."""
    import joblib
    import sklearn

    comp = pd.DataFrame({"y_true": np.asarray(y_eval).astype("int8"), "day": np.asarray(day_eval).astype("int32")})
    for k, v in sup_on_eval.items():
        comp[k] = np.asarray(v, dtype="float32")
    for k, v in all_scores.items():
        comp["u:" + k] = np.asarray(v, dtype="float32")
    comp.to_csv(os.path.join(out_dir, "comparison_sample.csv.gz"), index=False, compression="gzip")

    if unsup_parts is not None and unsup_models is not None:
        quant = {k: np.quantile(np.asarray(v, dtype="float64"), np.linspace(0, 1, 201)).tolist() for k, v in all_scores.items()}
        joblib.dump({"parts": unsup_parts, "models": unsup_models, "quantiles": quant,
                     "sklearn_version": sklearn.__version__}, os.path.join(out_dir, "unsup_models.joblib"), compress=3)

    if eda is not None:
        write_eda(out_dir, eda)

    if booster is not None and X_shap is not None:
        write_shap_sample(out_dir, booster, X_shap, shap_top)

    mp = os.path.join(out_dir, "meta.json")
    meta = json.load(open(mp))
    meta["comparison"] = {"n_sample": int(len(comp)), "fraud_rate": float(comp["y_true"].mean()),
                          "supervised": list(sup_on_eval), "unsupervised": list(all_scores)}
    meta["unsup_tables"] = {k: _records(v) for k, v in (unsup_tables or {}).items()}
    meta["unsup_settings"] = unsup_settings or {}
    meta["versions"] = {"sklearn": sklearn.__version__, "python": platform.python_version()}
    try:
        import lightgbm
        meta["versions"]["lightgbm"] = lightgbm.__version__
    except Exception:
        pass
    json.dump(meta, open(mp, "w"), indent=1)
    print("Extras written:", sorted(f for f in os.listdir(out_dir)))


# ------------------------------------------------------------------ run the export
import shutil
from scipy.stats import rankdata

EXPORT_DIR = "/kaggle/working/artifacts" if ON_KAGGLE else "artifacts"

def _rk(p):
    return rankdata(p) / len(p)

p_hold = {"p_baseline": lgb_hold_proba, "p_v2": lgb2_hold}
for name, var in [("p_xgb", "xgb_hold"), ("p_cat", "cb_hold")]:
    if var in globals():
        p_hold[name] = globals()[var]
if "xgb_hold" in globals() and "cb_hold" in globals():
    p_hold["p_blend"] = (_rk(lgb2_hold) + _rk(xgb_hold) + _rk(cb_hold)) / 3

tables = {}
for key, var in [("model_table", "cmp"), ("smote", "smote_cmp"), ("client_avg", "pp"),
                 ("time_consistency", "tc_cmp"), ("ensemble", "ens_cmp")]:
    if var in globals():
        tables[key] = globals()[var]

A_ref = A.copy()
A_ref["TransactionDT"] = A_ref["_dt"]

amount_all = A["TransactionAmt"].to_numpy()
p_valid_all = {"p_baseline": lgb_valid_proba}
for name, var in [("p_xgb", "xgb_valid"), ("p_cat", "cb_valid")]:
    if var in globals():
        p_valid_all[name] = globals()[var]
if "xgb_valid" in globals() and "cb_valid" in globals():
    p_valid_all["p_blend"] = (_rk(lgb2_valid) + _rk(xgb_valid) + _rk(cb_valid)) / 3

meta_app = export_artifacts(
    EXPORT_DIR,
    booster=model2.booster_, A=A_ref, new_feats=new_feats, feature_names=list(X_train2.columns),
    cat_cols=cat_cols, category_levels={c: X_train2[c].cat.categories for c in cat_cols},
    y_valid=y_valid.to_numpy(), p_valid=lgb2_valid, y_hold=y_hold.to_numpy(), p_hold=p_hold,
    day_hold=(A["_dt"].iloc[i_valid:n] // 86400).to_numpy(),
    cost_fp=COST_FP, cost_fn=COST_FN, max_alert_rate=MAX_ALERT_RATE,
    tables=tables, time_inconsistent=globals().get("bad_feats", []),
    best_iteration=int(model2.best_iteration_),
    amount_hold=amount_all[i_valid:n], amount_valid=amount_all[i_train:i_valid], p_valid_all=p_valid_all,
)

# ---- supervised-vs-unsupervised comparison, unsupervised models, EDA and SHAP sample
pos_eval = X_hold.index.get_indexer(eval_idx)              # rows of the unsupervised evaluation sample, inside the holdout
sup_on_eval = {k: np.asarray(v)[pos_eval] for k, v in p_hold.items()}
eda = build_eda(train, client=A.iloc[:n][["uid_count", "uid_prior_count", "amt_to_uid_mean"]])
export_extras(
    EXPORT_DIR,
    y_eval=y_eval, day_eval=(A["_dt"].iloc[i_valid:n].to_numpy()[pos_eval] // 86400),
    sup_on_eval=sup_on_eval, all_scores=all_scores,
    unsup_parts={"num_cols": feat.num_cols, "freq_cols": feat.freq_cols, "freq_maps": feat.freq_maps,
                 "imputer": feat.imputer, "scaler": feat.scaler, "pca": pca},
    unsup_models={"if_model": if_model, "svm_model": svm_model, "ae_model": ae_model},
    unsup_tables={"unsup_results": unsup_results, "unsup_tests": globals().get("res")},
    unsup_settings={"feature_mode": FEATURE_MODE, "num_cols": num_cols, "freq_cols": freq_cols,
                    "fit_on_legit_only": FIT_ON_LEGIT_ONLY},
    eda=eda, booster=model2.booster_, X_shap=X_hold2.sample(n=min(2000, len(X_hold2)), random_state=RANDOM_STATE),
)
zip_base = "/kaggle/working/app_artifacts" if ON_KAGGLE else "app_artifacts"
shutil.make_archive(zip_base, "zip", EXPORT_DIR)
big = [f.name for f in Path(EXPORT_DIR).iterdir() if f.stat().st_size > 90e6]
print("ZIP:", zip_base + ".zip")
if big:
    print("WARNING: files over 90 MB cannot be committed to GitHub; host the zip and use the ARTIFACT_URL secret:", big)
