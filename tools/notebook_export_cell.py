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
                     tables=None, time_inconsistent=None, is_demo=False, split_note="", best_iteration=None):
    """A: DataFrame over ALL reference rows (train+valid+holdout+test) with columns
    card1, card2, card3, card5, addr1, D1, TransactionAmt, P_emaildomain and TransactionDT.
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
    for name, p in p_hold.items():
        preds[name] = np.asarray(p, dtype="float32")
    preds.to_csv(os.path.join(out_dir, "holdout_predictions.csv.gz"), index=False, compression="gzip")

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

meta_app = export_artifacts(
    EXPORT_DIR,
    booster=model2.booster_, A=A_ref, new_feats=new_feats, feature_names=list(X_train2.columns),
    cat_cols=cat_cols, category_levels={c: X_train2[c].cat.categories for c in cat_cols},
    y_valid=y_valid.to_numpy(), p_valid=lgb2_valid, y_hold=y_hold.to_numpy(), p_hold=p_hold,
    day_hold=(A["_dt"].iloc[i_valid:n] // 86400).to_numpy(),
    cost_fp=COST_FP, cost_fn=COST_FN, max_alert_rate=MAX_ALERT_RATE,
    tables=tables, time_inconsistent=globals().get("bad_feats", []),
    best_iteration=int(model2.best_iteration_),
)
zip_base = "/kaggle/working/app_artifacts" if ON_KAGGLE else "app_artifacts"
shutil.make_archive(zip_base, "zip", EXPORT_DIR)
big = [f.name for f in Path(EXPORT_DIR).iterdir() if f.stat().st_size > 90e6]
print("ZIP:", zip_base + ".zip")
if big:
    print("WARNING: files over 90 MB cannot be committed to GitHub; host the zip and use the ARTIFACT_URL secret:", big)
