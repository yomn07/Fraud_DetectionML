"""Run the notebook export cell against synthetic variables that mirror the notebook's names."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_export_cell_runs_and_app_loads_it(tmp_path, monkeypatch):
    import tools.make_demo_artifacts as demo
    L = demo.main()
    ns = {
        "ON_KAGGLE": False, "model2": L["m2"], "A": L["A"].assign(_dt=L["A"]["TransactionDT"]),
        "new_feats": L["new_feats"], "X_train2": L["X2"].iloc[:L["i_tr"]], "cat_cols": L["cat_cols"],
        "y_valid": L["y"].iloc[L["i_tr"]:L["i_va"]], "y_hold": L["y"].iloc[L["i_va"]:],
        "lgb2_valid": L["pv2"], "lgb2_hold": L["ph2"], "lgb_hold_proba": L["ph1"],
        "i_valid": L["i_va"], "n": L["n"], "COST_FP": 1, "COST_FN": 300, "MAX_ALERT_RATE": 0.05,
    }
    monkeypatch.chdir(tmp_path)
    code = (ROOT / "tools" / "notebook_export_cell.py").read_text()
    exec(compile(code, "export_cell", "exec"), ns)
    out = tmp_path / "artifacts"
    for f in ("meta.json", "model_lgbm.txt.gz", "holdout_predictions.csv.gz", "ref_uid.parquet", "ref_card.parquet", "ref_email.parquet"):
        assert (out / f).exists(), f
    assert (tmp_path / "app_artifacts.zip").exists()
