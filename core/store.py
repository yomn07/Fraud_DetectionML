"""Streamlit-free artifact loading (used by the app, the API and the tests)."""
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd

REQUIRED = ["meta.json", "model_lgbm.txt.gz", "holdout_predictions.csv.gz",
            "ref_uid.parquet", "ref_card.parquet", "ref_email.parquet"]

MODEL_LABELS = {
    "p_baseline": "LightGBM baseline",
    "p_v2": "LightGBM + client features",
    "p_xgb": "XGBoost + client features",
    "p_cat": "CatBoost + client features",
    "p_blend": "Equal-weight rank blend",
}


@dataclass
class Artifacts:
    meta: dict
    booster: lgb.Booster
    preds: pd.DataFrame                  # holdout ("test") predictions, labels and amounts
    ref: dict
    path: Path
    valid: pd.DataFrame = None           # validation predictions, labels and amounts (thresholds are chosen here)
    comparison: pd.DataFrame = None      # supervised + unsupervised scores on the same holdout sample
    unsup: dict = None
    eda: dict = None
    shap_sample: pd.DataFrame = None     # SHAP values (+ percentile of feature values) for a holdout sample
    unsup_error: str = ""

    @property
    def is_demo(self):
        return bool(self.meta.get("is_demo"))

    @property
    def models(self):
        return [c for c in MODEL_LABELS if c in self.preds.columns]


def valid_dir(p):
    p = Path(p)
    return p.is_dir() and all((p / f).exists() for f in REQUIRED)


def _opt_csv(path, name):
    f = Path(path) / name
    return pd.read_csv(f) if f.exists() else None


def load_from_dir(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    with gzip.open(path / "model_lgbm.txt.gz", "rt") as f:
        booster = lgb.Booster(model_str=f.read())
    preds = pd.read_csv(path / "holdout_predictions.csv.gz")
    ref = {}
    for kind in ("uid", "card", "email"):
        r = pd.read_parquet(path / f"ref_{kind}.parquet").set_index("h")[["n", "s", "ss"]]
        ref[kind] = r[~r.index.duplicated()]
    unsup, unsup_error = None, ""
    if (path / "unsup_models.joblib").exists():
        try:
            import joblib
            import sklearn
            unsup = joblib.load(path / "unsup_models.joblib")
            saved = unsup.get("sklearn_version")
            if saved and saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
                unsup_error = (f"Unsupervised models were saved with scikit-learn {saved} but this runs {sklearn.__version__}. "
                               f"Pin scikit-learn=={saved} in requirements.txt if scoring with them fails.")
        except Exception as e:     # incompatible pickle: keep the rest working
            unsup, unsup_error = None, f"Unsupervised models could not be loaded: {e}"
    eda = None
    if (path / "eda.json.gz").exists():
        with gzip.open(path / "eda.json.gz", "rt") as f:
            eda = json.load(f)
    return Artifacts(meta=meta, booster=booster, preds=preds, ref=ref, path=path,
                     valid=_opt_csv(path, "valid_predictions.csv.gz"), comparison=_opt_csv(path, "comparison_sample.csv.gz"),
                     unsup=unsup, eda=eda, shap_sample=_opt_csv(path, "shap_sample.csv.gz"), unsup_error=unsup_error)
