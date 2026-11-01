"""Client-key construction shared by the notebook export and the Streamlit app.

Only numpy/pandas are used so the source can be pasted into the Kaggle notebook unchanged.
Keys are canonical strings (ints formatted without ".0", missing = "nan") and are hashed to uint64,
so the same client gets the same key whatever dtype a CSV was parsed with.
"""
import numpy as np
import pandas as pd

KEY_KINDS = ("uid", "card", "email")


def _num_str(s):
    v = pd.to_numeric(s, errors="coerce")
    return v.round().astype("Int64").astype("string").fillna("nan").astype(object)


def make_keys(df):
    """df needs TransactionDT, D1, card1, card2, card3, card5, addr1, P_emaildomain."""
    day = np.floor(pd.to_numeric(df["TransactionDT"], errors="coerce") / 86400)
    start_day = _num_str(day - pd.to_numeric(df["D1"], errors="coerce"))
    card = _num_str(df["card1"])
    for c in ("card2", "card3", "card5", "addr1"):
        card = card + "_" + _num_str(df[c])
    uid = card + "_" + start_day
    email = df["P_emaildomain"].astype("string").fillna("nan").astype(object)
    return {"uid": uid, "card": card, "email": email}


def hash_keys(keys):
    return pd.util.hash_pandas_object(pd.Series(np.asarray(keys, dtype=object)), index=False).to_numpy()


def key_stats(h, amount):
    """Per-key count, sum and sum of squares of the transaction amount (float64)."""
    d = pd.DataFrame({"h": h, "a": np.asarray(amount, dtype="float64")})
    d["a2"] = d["a"] ** 2
    g = d.groupby("h")
    return pd.DataFrame({"n": g["a"].size().astype("int64"), "s": g["a"].sum(), "ss": g["a2"].sum()})


def stats_to_features(n, s, ss):
    n = np.asarray(n, dtype="float64")
    mean = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    var = np.where(n > 1, (ss - n * mean ** 2) / np.maximum(n - 1, 1), np.nan)
    return mean, np.sqrt(np.maximum(var, 0.0))
