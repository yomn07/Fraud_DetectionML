"""Write everything the Streamlit app needs into one folder.

Used in two places with identical code: the notebook export cell (pasted together with core/keys.py) and
tools/make_demo_artifacts.py (synthetic demo data).
"""
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
