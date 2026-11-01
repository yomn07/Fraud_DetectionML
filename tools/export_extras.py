"""Second half of the export: supervised-vs-unsupervised comparison sample, unsupervised models and EDA."""
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
