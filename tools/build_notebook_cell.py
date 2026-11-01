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
'''

MD = """## 18. Export artifacts for the Streamlit app

Writes the section-13 LightGBM (`model2`, trained on the training split only, so it matches the holdout numbers and the
validation-chosen thresholds), the hashed client-feature reference tables, the holdout predictions of every model, the
thresholds, validation predictions with transaction amounts, the fitted unsupervised detectors, the EDA summaries and a SHAP sample into one folder, then zips it. Download `app_artifacts.zip` from the notebook output
and unzip it into the app's `artifacts/` folder.

Run it after sections 13-17 (sections 14-17 are optional; whatever exists is exported)."""


def _body(path):
    text = path.read_text()
    return text.split('"""', 2)[2].lstrip("\n")          # drop the module docstring


def build_cell_source():
    return ("from pathlib import Path\n\n# ---- client-key helpers (identical to the app's core/keys.py) ----\n"
            + _body(ROOT / "core" / "keys.py") + "\n\n# ---- EDA summaries (core/eda.py) ----\n"
            + _body(ROOT / "core" / "eda.py") + "\n\n# ---- exporter ----\n"
            + _body(ROOT / "tools" / "export_artifacts.py") + "\n\n# ---- extras ----\n"
            + _body(ROOT / "tools" / "export_extras.py") + CALL)


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
