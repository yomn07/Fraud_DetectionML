"""Run the notebook export cell against synthetic variables that mirror the notebook's names."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_export_cell_runs_and_app_loads_it(tmp_path, monkeypatch):
    import tools.make_demo_artifacts as demo
    from tools.build_notebook_cell import build_cell_source
    L = demo.main()
    n, i_tr, i_va = L["n"], L["i_tr"], L["i_va"]
    df, X, X2, A = L["df"], L["X"], L["X2"], L["A"]
    train = df.copy()
    train["TransactionHour"] = (df["TransactionDT"] // 3600) % 24
    ns = {
        "ON_KAGGLE": False, "model2": L["m2"], "A": A.assign(_dt=A["TransactionDT"]), "n": n, "train": train,
        "new_feats": L["new_feats"], "X_train2": X2.iloc[:i_tr], "X_hold": X.iloc[i_va:], "X_hold2": X2.iloc[i_va:],
        "cat_cols": L["cat_cols"], "RANDOM_STATE": 42,
        "y_valid": L["y"].iloc[i_tr:i_va], "y_hold": L["y"].iloc[i_va:],
        "lgb2_valid": L["pv2"], "lgb2_hold": L["ph2"], "lgb_hold_proba": L["ph1"], "lgb_valid_proba": L["pv1"],
        "i_train": i_tr, "i_valid": i_va, "COST_FP": 1, "COST_FN": 300, "MAX_ALERT_RATE": 0.05,
        "eval_idx": L["hold_idx"], "y_eval": L["ev"], "all_scores": L["all_scores"], "feat": L["feat"], "pca": L["pca"],
        "if_model": L["if_model"], "svm_model": L["svm_model"], "ae_model": L["ae_model"],
        "unsup_results": L["unsup_results"], "FEATURE_MODE": "fixed", "num_cols": L["num_cols"], "freq_cols": L["freq_cols"],
        "FIT_ON_LEGIT_ONLY": False,
    }
    monkeypatch.chdir(tmp_path)
    code = build_cell_source()
    exec(compile(code, "export_cell", "exec"), ns)
    out = tmp_path / "artifacts"
    for f in ("meta.json", "model_lgbm.txt.gz", "holdout_predictions.csv.gz", "valid_predictions.csv.gz", "ref_uid.parquet",
              "ref_card.parquet", "ref_email.parquet", "comparison_sample.csv.gz", "unsup_models.joblib", "eda.json.gz",
              "shap_sample.csv.gz"):
        assert (out / f).exists(), f
    assert (tmp_path / "app_artifacts.zip").exists()
    from core.store import load_from_dir
    art = load_from_dir(out)
    assert art.valid is not None and "amount" in art.preds.columns and art.unsup is not None and art.eda is not None
