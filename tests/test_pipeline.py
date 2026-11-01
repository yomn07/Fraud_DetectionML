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


def test_chunked_scoring_matches_in_memory_scoring():
    from core.batch import score_stream
    from core.store import load_from_dir
    art = load_from_dir(D)
    raw = pd.read_csv(D / "sample_transactions.csv")
    s_mem, _, _ = score_frame(art, raw, "reference")
    res = score_stream(art, str(D / "sample_transactions.csv"), None, mode="reference", chunk_rows=97, keep_top=10)
    assert len(res["scores"]) == len(raw)
    # features are computed from whole-file statistics in both paths, so scores must agree to float precision
    assert np.allclose(res["scores"], s_mem, atol=1e-9)
    assert set(res["unsup"]) == {"Isolation Forest", "One-Class SVM (approx.)", "Autoencoder"}
    assert len(res["top_X"]) <= 10 * (len(raw) // 97 + 1)


def test_unsupervised_percentiles_and_reason_codes():
    from core.scoring import reason_codes, waterfall_data
    from core.store import load_from_dir
    from core.unsup import score_unsup, to_percentile
    art = load_from_dir(D)
    raw = pd.read_csv(D / "sample_transactions.csv")
    s, X, _ = score_frame(art, raw, "reference")
    for name, v in score_unsup(art.unsup, X).items():
        pct = to_percentile(art.unsup, name, v)
        assert pct.min() >= 0 and pct.max() <= 100
    table, base, logit = explain_row(art, X.iloc[[int(np.argmax(s))]])
    codes = reason_codes(table)
    assert 3 <= len(codes) <= 5 and all(c["effect"] > 0 for c in codes)
    labels, values, measure = waterfall_data(table, base)
    assert abs(values[-1] - logit) < 1e-9 and len(labels) == len(values) == len(measure)


def test_amount_cost_threshold_beats_extremes():
    from core.metrics import amount_cost, best_threshold_amount
    from core.store import load_from_dir
    art = load_from_dir(D)
    v, t = art.valid, art.preds
    thr = best_threshold_amount(v["y_true"].to_numpy(), v["p_v2"].to_numpy(), v["amount"].to_numpy(), 5.0)
    c = amount_cost(v["y_true"].to_numpy(), v["p_v2"].to_numpy(), v["amount"].to_numpy(), thr, 5.0)["total"]
    nothing = float(v["amount"][v["y_true"] == 1].sum())
    everything = 5.0 * len(v)
    assert c <= min(nothing, everything) + 1e-6
    ct = amount_cost(t["y_true"].to_numpy(), t["p_v2"].to_numpy(), t["amount"].to_numpy(), thr, 5.0)["total"]
    assert ct < float(t["amount"][t["y_true"] == 1].sum())      # transfers to the test period


def test_api():
    from fastapi.testclient import TestClient
    import importlib
    api = importlib.import_module("api")
    c = TestClient(api.app)
    assert c.get("/health").json()["status"] == "ok"
    raw = pd.read_csv(D / "sample_transactions.csv").head(30)
    body = {"transactions": json.loads(raw.to_json(orient="records")), "mode": "reference", "rule": "f2"}
    r = c.post("/score", json=body)
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert len(res) == 30 and all(0 <= x["score"] <= 1 for x in res)
    flagged = [x for x in res if x["flagged"]]
    assert all(3 <= len(x["reason_codes"]) <= 5 for x in flagged)
    assert c.post("/score", json={"transactions": [{"TransactionAmt": 1.0}]}).status_code == 422
