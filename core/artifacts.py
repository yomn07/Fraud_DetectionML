"""Locate, optionally download, and cache the exported artifacts for the Streamlit app."""
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import streamlit as st

from core.store import MODEL_LABELS, REQUIRED, Artifacts, load_from_dir, valid_dir  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
SUPERVISED_LABELS = dict(MODEL_LABELS)
_valid_dir = valid_dir


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
            if name and not member.endswith("/") and not name.startswith("."):
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
    return None if path is None else load_from_dir(path)


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
