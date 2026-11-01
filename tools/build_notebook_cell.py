"""Assemble the notebook export cell from the app's own source files (single source of truth) and append it
to a copy of the notebook.   python tools/build_notebook_cell.py <in.ipynb> <out.ipynb>"""
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CALL = '''

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
'''

MD = """## 18. Export artifacts for the Streamlit app

Writes the section-13 LightGBM (`model2`, trained on the training split only, so it matches the holdout numbers and the
validation-chosen thresholds), the hashed client-feature reference tables, the holdout predictions of every model, the
thresholds and the experiment tables into one folder, then zips it. Download `app_artifacts.zip` from the notebook output
and unzip it into the app's `artifacts/` folder.

Run it after sections 13-17 (sections 14-17 are optional; whatever exists is exported)."""


def _body(path):
    text = path.read_text()
    return text.split('"""', 2)[2].lstrip("\n")          # drop the module docstring


def build_cell_source():
    return ("from pathlib import Path\n\n# ---- client-key helpers (identical to the app's core/keys.py) ----\n"
            + _body(ROOT / "core" / "keys.py") + "\n\n# ---- exporter ----\n"
            + _body(ROOT / "tools" / "export_artifacts.py") + CALL)


def main(src, dst):
    nb = json.load(open(src))
    code_src = build_cell_source()
    compile(code_src, "export_cell", "exec")
    (ROOT / "tools" / "notebook_export_cell.py").write_text(code_src)
    nb["cells"].append({"id": uuid.uuid4().hex[:8], "cell_type": "markdown", "source": MD, "metadata": {}})
    nb["cells"].append({"id": uuid.uuid4().hex[:8], "cell_type": "code", "source": code_src,
                        "metadata": {"trusted": True}, "outputs": [], "execution_count": None})
    json.dump(nb, open(dst, "w"), indent=1)
    print("cells:", len(nb["cells"]), "| export cell lines:", code_src.count("\n"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
