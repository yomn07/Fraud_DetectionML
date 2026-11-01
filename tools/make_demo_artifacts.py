"""Create small SYNTHETIC demo artifacts (same schema as the real export) so the app runs end to end.

    python tools/make_demo_artifacts.py            # writes demo_artifacts/
The data is random and unrelated to the Kaggle competition; the app labels it as demo.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.keys import hash_keys  # noqa: E402,F401
from tools.export_artifacts import export_artifacts  # noqa: E402
from tools.export_extras import export_extras  # noqa: E402
from core.eda import build_eda  # noqa: E402
from core.unsup import UnsupFeaturizer  # noqa: E402

RNG = np.random.RandomState(7)
N_CLIENTS, N_ROWS, DAYS = 2500, 70000, 150


def synth():
    cl = pd.DataFrame({
        "card1": RNG.randint(1000, 18000, N_CLIENTS),
        "card2": np.where(RNG.rand(N_CLIENTS) < 0.05, np.nan, RNG.randint(100, 600, N_CLIENTS)).astype(float),
        "card3": RNG.choice([150.0, 185.0, 144.0], N_CLIENTS, p=[0.8, 0.15, 0.05]),
        "card4": RNG.choice(["visa", "mastercard", "american express", "discover"], N_CLIENTS, p=[0.65, 0.25, 0.06, 0.04]),
        "card5": RNG.choice([226.0, 224.0, 166.0, 102.0], N_CLIENTS),
        "card6": RNG.choice(["debit", "credit"], N_CLIENTS, p=[0.75, 0.25]),
        "addr1": RNG.randint(100, 540, N_CLIENTS).astype(float),
        "email": RNG.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], N_CLIENTS,
                            p=[0.45, 0.2, 0.15, 0.08, 0.12]),
        "start": RNG.randint(0, DAYS - 10, N_CLIENTS),
        "amt_scale": np.exp(RNG.normal(3.6, 0.7, N_CLIENTS)),
        "risk": np.where(RNG.rand(N_CLIENTS) < 0.12, 3.0, 0.2),     # a minority of clients is compromised
    })
    popularity = np.exp(RNG.normal(0, 0.8, N_CLIENTS))
    cid = RNG.choice(N_CLIENTS, N_ROWS, p=popularity / popularity.sum())
    c = cl.iloc[cid].reset_index(drop=True)
    day = np.minimum(DAYS - 1, c["start"].to_numpy() + RNG.exponential(35, N_ROWS).astype(int))
    dt = day * 86400 + RNG.randint(0, 86400, N_ROWS)
    amt = np.round(c["amt_scale"].to_numpy() * np.exp(RNG.normal(0, 0.6, N_ROWS)), 2)
    is_night = ((dt // 3600) % 24 < 6).astype(float)
    logit = -4.4 + (c["risk"].to_numpy() > 1) * 1.7 + 0.6 * is_night + 0.2 * np.log1p(amt) \
            + 0.8 * (c["email"].to_numpy() == "anonymous.com")
    fraud = (RNG.rand(N_ROWS) < 1 / (1 + np.exp(-logit))).astype("int8")
    bursty = fraud == 1
    amt = np.where(bursty, np.round(amt * np.exp(RNG.normal(0.5, 0.5, N_ROWS)), 2), amt)
    df = pd.DataFrame({
        "TransactionID": np.arange(3000000, 3000000 + N_ROWS),
        "isFraud": fraud, "TransactionDT": dt, "TransactionAmt": amt,
        "ProductCD": RNG.choice(["W", "C", "R", "H", "S"], N_ROWS, p=[0.7, 0.1, 0.08, 0.07, 0.05]),
        "card1": c["card1"].to_numpy(), "card2": c["card2"].to_numpy(), "card3": c["card3"].to_numpy(),
        "card4": c["card4"].to_numpy(), "card5": c["card5"].to_numpy(), "card6": c["card6"].to_numpy(),
        "addr1": c["addr1"].to_numpy(), "addr2": 87.0,
        "P_emaildomain": c["email"].to_numpy(),
        "R_emaildomain": np.where(RNG.rand(N_ROWS) < 0.6, None, "gmail.com"),
        "C1": np.maximum(1, RNG.poisson(1.5 + 2 * fraud)).astype(float),
        "C2": np.maximum(1, RNG.poisson(1.5 + 2 * fraud)).astype(float),
        "C3": 0.0,
        "D1": np.where(RNG.rand(N_ROWS) < 0.08, np.nan, (day - c["start"].to_numpy()).astype(float)),
        "D2": np.where(RNG.rand(N_ROWS) < 0.5, np.nan, RNG.exponential(40, N_ROWS)),
        "M1": RNG.choice(["T", "F", None], N_ROWS, p=[0.6, 0.1, 0.3]),
        "M2": RNG.choice(["T", "F", None], N_ROWS, p=[0.55, 0.15, 0.3]),
        "M3": RNG.choice(["T", "F", None], N_ROWS, p=[0.5, 0.2, 0.3]),
        "V1": np.where(RNG.rand(N_ROWS) < 0.4, np.nan, RNG.normal(0.2 * fraud, 1)),
        "V2": np.where(RNG.rand(N_ROWS) < 0.4, np.nan, RNG.normal(0, 1, N_ROWS)),
        "V3": RNG.normal(0.3 * fraud, 1),
        "id_01": np.where(RNG.rand(N_ROWS) < 0.75, np.nan, -RNG.exponential(8, N_ROWS)),
        "DeviceType": RNG.choice(["desktop", "mobile", None], N_ROWS, p=[0.15, 0.1, 0.75]),
    })
    return df.sort_values("TransactionDT").reset_index(drop=True)


def main():
    df = synth()
    n = len(df)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    y = df["isFraud"].astype("int8")
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"]).copy()
    hour = (df["TransactionDT"] // 3600) % 24
    X["TransactionHour"] = hour.astype("int16")
    X["TransactionHour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["TransactionHour_cos"] = np.cos(2 * np.pi * hour / 24)
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    levels = {}
    for c in cat_cols:
        cats = pd.Index(X[c].astype("string").dropna().unique())
        levels[c] = list(cats)
        X[c] = pd.Categorical(X[c].astype("string"), categories=cats)

    # client features, written the way the notebook does it (astype(str) keys + groupby transform)
    A = pd.DataFrame({"card1": df["card1"], "card2": df["card2"], "card3": df["card3"], "card5": df["card5"],
                      "addr1": df["addr1"], "D1": df["D1"], "TransactionAmt": df["TransactionAmt"],
                      "P_emaildomain": df["P_emaildomain"].astype(object), "TransactionDT": df["TransactionDT"]})
    # pandas 2 (Kaggle) turns NaN into "nan" with astype(str); pandas 3 keeps NaN, so use map(str) to behave the same.
    S = lambda col: col.map(str)
    day = A["TransactionDT"] // 86400
    start_day = (day - A["D1"]).round()
    card_key = (S(A["card1"]) + "_" + S(A["card2"]) + "_" + S(A["card3"]) + "_" + S(A["card5"]) + "_" + S(A["addr1"]))
    A["uid"] = card_key + "_" + S(start_day)
    A["card_key"] = card_key
    A["email_key"] = S(A["P_emaildomain"].astype(object).where(A["P_emaildomain"].notna(), np.nan))
    new_feats = []
    for key, name in [("uid", "uid"), ("card_key", "card"), ("email_key", "email")]:
        g = A.groupby(key)["TransactionAmt"]
        A[f"{name}_count"] = g.transform("size").astype("int32")
        A[f"{name}_amt_mean"] = g.transform("mean").astype("float32")
        new_feats += [f"{name}_count", f"{name}_amt_mean"]
    A["uid_amt_std"] = A.groupby("uid")["TransactionAmt"].transform("std").astype("float32")
    A["amt_to_uid_mean"] = (A["TransactionAmt"] / A["uid_amt_mean"]).astype("float32")
    new_feats += ["uid_amt_std", "amt_to_uid_mean"]
    A["uid_prior_count"] = A.groupby("uid").cumcount().astype("int32")   # df is time-sorted
    new_feats.append("uid_prior_count")

    X2 = X.copy()
    for f in new_feats:
        X2[f] = A[f].to_numpy()

    def fit(Xf, name):
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, colsample_bytree=0.7,
                               subsample=0.8, subsample_freq=1, random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(Xf.iloc[:i_tr], y.iloc[:i_tr], categorical_feature=cat_cols,
              eval_set=[(Xf.iloc[i_tr:i_va], y.iloc[i_tr:i_va])], callbacks=[lgb.early_stopping(50, verbose=False)])
        pv = m.predict_proba(Xf.iloc[i_tr:i_va])[:, 1]
        ph = m.predict_proba(Xf.iloc[i_va:])[:, 1]
        print(f"{name}: valid AUC {roc_auc_score(y.iloc[i_tr:i_va], pv):.3f}  holdout AP {average_precision_score(y.iloc[i_va:], ph):.3f}")
        return m, pv, ph

    m1, pv1, ph1 = fit(X, "baseline")
    m2, pv2, ph2 = fit(X2, "v2")
    yh = y.iloc[i_va:].to_numpy()
    from core.metrics import summary
    table = pd.DataFrame([{"Model": "LightGBM baseline", **summary(yh, ph1)},
                          {"Model": "LightGBM + client features", **summary(yh, ph2)}])

    out = ROOT / "demo_artifacts"
    export_artifacts(
        out, booster=m2.booster_, A=A, new_feats=new_feats, feature_names=list(X2.columns), cat_cols=cat_cols,
        category_levels=levels, y_valid=y.iloc[i_tr:i_va].to_numpy(), p_valid=pv2, y_hold=yh,
        p_hold={"p_baseline": ph1, "p_v2": ph2}, day_hold=(df["TransactionDT"].iloc[i_va:] // 86400).to_numpy(),
        tables={"model_table": table}, is_demo=True, best_iteration=int(m2.best_iteration_),
        split_note="synthetic data, chronological 70/15/15",
        amount_hold=df["TransactionAmt"].iloc[i_va:].to_numpy(), amount_valid=df["TransactionAmt"].iloc[i_tr:i_va].to_numpy(),
        p_valid_all={"p_baseline": pv1})

    # ---- unsupervised detectors (same recipe as notebook section 8, "fixed" feature mode)
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import SGDOneClassSVM
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    chosen = [c for c in ["C1", "C2", "C3", "D1", "D2", "TransactionAmt", "TransactionHour", "card1", "card2", "card5",
                          "addr1", "P_emaildomain", "card6"] if c in X.columns]
    ID_LIKE = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
    freq_cols = [c for c in chosen if c in cat_cols or c in ID_LIKE]
    num_cols = [c for c in chosen if c not in freq_cols and pd.api.types.is_numeric_dtype(X[c])]
    Xtr = X.iloc[:i_tr]
    big = Xtr.sample(n=min(20000, len(Xtr)), random_state=42)
    feat = UnsupFeaturizer(num_cols, freq_cols).fit_freq(Xtr).fit_scale(big)
    Zb = feat.transform(big)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=42).fit(Zb)
    Pb = pca.transform(Zb)
    if_model = IsolationForest(n_estimators=100, max_samples=1024, random_state=42, n_jobs=-1).fit(Zb)
    svm_model = make_pipeline(Nystroem(gamma=1.0 / Pb.shape[1], n_components=100, random_state=42),
                              SGDOneClassSVM(nu=0.05, random_state=42)).fit(Pb)
    ae_model = MLPRegressor(hidden_layer_sizes=(16, 6, 16), max_iter=60, random_state=42).fit(Zb[:10000], Zb[:10000])
    hold_idx = X.iloc[i_va:].sample(n=min(4000, n - i_va), random_state=42).index
    Ze = feat.transform(X.loc[hold_idx])
    all_scores = {"Isolation Forest": -if_model.decision_function(Ze),
                  "One-Class SVM (approx.)": -svm_model.decision_function(pca.transform(Ze)),
                  "Autoencoder": ((Ze - ae_model.predict(Ze)) ** 2).mean(axis=1)}
    pos = hold_idx - i_va
    ev = y.loc[hold_idx].to_numpy()
    unsup_results = pd.DataFrame([{"Model": k, "ROC-AUC": roc_auc_score(ev, v), "Average Precision": average_precision_score(ev, v)}
                                  for k, v in all_scores.items()]).sort_values("Average Precision", ascending=False)
    print(unsup_results.round(3).to_string(index=False))

    eda_df = df.copy()
    eda_df["TransactionHour"] = hour
    eda = build_eda(eda_df, client=A[["uid_count", "uid_prior_count", "amt_to_uid_mean"]])
    export_extras(
        out, y_eval=ev, day_eval=(df["TransactionDT"].loc[hold_idx] // 86400).to_numpy(),
        sup_on_eval={"p_baseline": ph1[pos], "p_v2": ph2[pos]}, all_scores=all_scores,
        unsup_parts={"num_cols": feat.num_cols, "freq_cols": feat.freq_cols, "freq_maps": feat.freq_maps,
                     "imputer": feat.imputer, "scaler": feat.scaler, "pca": pca},
        unsup_models={"if_model": if_model, "svm_model": svm_model, "ae_model": ae_model},
        unsup_tables={"unsup_results": unsup_results},
        unsup_settings={"feature_mode": "fixed", "num_cols": num_cols, "freq_cols": freq_cols, "fit_on_legit_only": False},
        eda=eda, booster=m2.booster_, X_shap=X2.iloc[i_va:].sample(n=min(1500, n - i_va), random_state=42))

    sample = df.iloc[i_va:].sample(400, random_state=1).sort_values("TransactionDT")
    sample.drop(columns=["isFraud"]).to_csv(out / "sample_transactions.csv", index=False)
    sample[["TransactionID", "isFraud"]].to_csv(out / "sample_labels.csv", index=False)
    expected = X2.loc[sample.index, new_feats].copy()
    expected.insert(0, "TransactionID", sample["TransactionID"].to_numpy())
    expected["p_v2"] = m2.predict_proba(X2.loc[sample.index])[:, 1]
    expected.to_csv(out / "sample_expected.csv", index=False)   # used by tests/test_pipeline.py
    print("sample rows:", len(sample), "| fraud in sample:", int(sample["isFraud"].sum()))
    return locals()


if __name__ == "__main__":
    main()
