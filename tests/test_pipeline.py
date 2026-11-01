import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json, gzip
import lightgbm as lgb
from core.features import prepare_features
from core.scoring import explain_row, score_frame
from core.metrics import best_threshold, confusion

D = ROOT / "demo_artifacts"


class Art:
    def __init__(self):
        self.meta = json.loads((D / "meta.json").read_text())
        with gzip.open(D / "model_lgbm.txt.gz", "rt") as f:
            self.booster = lgb.Booster(model_str=f.read())
        self.ref = {}
        for k in ("uid", "card", "email"):
            r = pd.read_parquet(D / f"ref_{k}.parquet").set_index("h")[["n", "s", "ss"]]
            self.ref[k] = r


def test_feature_parity_with_training():
    art = Art()
    raw = pd.read_csv(D / "sample_transactions.csv")
    exp = pd.read_csv(D / "sample_expected.csv")
    X, info = prepare_features(raw, art.meta, art.ref, mode="reference")
    assert list(X.columns) == art.meta["feature_names"]
    for f in ["uid_count", "card_count", "email_count", "uid_amt_mean", "card_amt_mean", "email_amt_mean", "uid_amt_std", "amt_to_uid_mean"]:
        a, b = X[f].to_numpy(dtype=float), exp[f].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-3, atol=1e-3, equal_nan=True), f


def test_scores_close_to_training_predictions():
    art = Art()
    raw = pd.read_csv(D / "sample_transactions.csv")
    exp = pd.read_csv(D / "sample_expected.csv")
    s, X, _ = score_frame(art, raw, "reference")
    assert np.corrcoef(s, exp["p_v2"])[0, 1] > 0.95      # uid_prior_count is approximated inside a small batch


def test_missing_columns_and_unseen_categories_do_not_crash():
    art = Art()
    raw = pd.DataFrame({"TransactionDT": [86400 * 100, 86400 * 100 + 5], "TransactionAmt": [20.0, 5000.0],
                        "ProductCD": ["W", "ZZZ"], "card1": [1234, 1234]})
    s, X, info = score_frame(art, raw, "new")
    assert len(s) == 2 and np.isfinite(s).all()
    assert "ProductCD" in info["unseen_category_share"]
    table, base, logit = explain_row(art, X.iloc[[0]])
    assert abs(1 / (1 + np.exp(-logit)) - s[0]) < 1e-6


def test_requires_core_columns():
    art = Art()
    try:
        score_frame(art, pd.DataFrame({"TransactionAmt": [1.0]}), "new")
        assert False
    except ValueError as e:
        assert "TransactionDT" in str(e)


def test_id_dash_columns_are_normalised():
    art = Art()
    raw = pd.DataFrame({"TransactionDT": [86400], "TransactionAmt": [10.0], "id-01": [-5.0]})
    X, info = prepare_features(raw, art.meta, art.ref, "new")
    if "id_01" in art.meta["feature_names"]:
        assert X["id_01"].iloc[0] == -5.0


def test_threshold_helpers():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.6, 90), np.linspace(0.4, 1.0, 10)])
    t = best_threshold(y, p, 1, 300)
    c = confusion(y, p, t)
    assert c["TP"] + c["FN"] == 10 and c["alerts"] == c["TP"] + c["FP"]
