"""Locate, optionally download, and load the model artifacts exported from the notebook."""
import gzip
import io
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
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
    preds: pd.DataFrame
    ref: dict
    path: Path

    @property
    def is_demo(self):
        return bool(self.meta.get("is_demo"))

    @property
    def models(self):
        return [c for c in MODEL_LABELS if c in self.preds.columns]


def _valid_dir(p):
    return p.is_dir() and all((p / f).exists() for f in REQUIRED)


def _try_download(target):
    """Optional: ARTIFACT_URL (env var or Streamlit secret) pointing to a zip of the artifacts folder."""
    url = os.environ.get("ARTIFACT_URL")
    if not url:
        try:
            url = st.secrets.get("ARTIFACT_URL")
        except Exception:
            url = None
    if not url:
        return False
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            name = Path(member).name
            if name in REQUIRED:
                (target / name).write_bytes(z.read(member))
    return _valid_dir(target)


def find_artifact_dir():
    env = os.environ.get("FRAUD_ARTIFACTS")
    candidates = ([Path(env)] if env else []) + [ROOT / "artifacts"]
    for c in candidates:
        if _valid_dir(c):
            return c
    try:
        if _try_download(ROOT / "artifacts"):
            return ROOT / "artifacts"
    except Exception as e:  # network or zip problem: fall back to the demo and tell the user
        st.session_state["download_error"] = str(e)
    return ROOT / "demo_artifacts" if _valid_dir(ROOT / "demo_artifacts") else None


@st.cache_resource(show_spinner="Loading model and reference data...")
def load_artifacts():
    path = find_artifact_dir()
    if path is None:
        return None
    meta = json.loads((path / "meta.json").read_text())
    with gzip.open(path / "model_lgbm.txt.gz", "rt") as f:
        booster = lgb.Booster(model_str=f.read())
    preds = pd.read_csv(path / "holdout_predictions.csv.gz")
    ref = {}
    for kind in ("uid", "card", "email"):
        r = pd.read_parquet(path / f"ref_{kind}.parquet").set_index("h")[["n", "s", "ss"]]
        ref[kind] = r[~r.index.duplicated()]
    return Artifacts(meta=meta, booster=booster, preds=preds, ref=ref, path=path)


def require_artifacts():
    """Call at the top of every page. Stops the page with instructions if nothing is available."""
    art = load_artifacts()
    if art is None:
        st.error("No model artifacts found.")
        st.markdown(
            "Run the **export cell** at the end of the notebook (section 18), put the generated files into "
            "`artifacts/`, or set the `ARTIFACT_URL` secret to a zip of that folder. See the README."
        )
        st.stop()
    if art.is_demo:
        st.warning(
            "**Demo mode:** these results come from a small *synthetic* dataset bundled with the app, not from "
            "the Kaggle data. Export your own artifacts (README, step 1) to see real results.",
            icon="🧪",
        )
    if "download_error" in st.session_state:
        st.info(f"Could not download artifacts: {st.session_state['download_error']}")
    return art
