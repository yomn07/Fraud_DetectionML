"""Turn raw IEEE-CIS transaction rows into the exact feature frame the v2 LightGBM model was trained on."""
import numpy as np
import pandas as pd

from core.keys import KEY_KINDS, hash_keys, key_stats, make_keys, stats_to_features

REQUIRED_COLUMNS = ["TransactionDT", "TransactionAmt"]
CLIENT_COLUMNS = ["card1", "card2", "card3", "card5", "addr1", "D1", "P_emaildomain"]
DROP_COLUMNS = ["isFraud", "TransactionID", "TransactionDT"]

NEW_FEATURES = ["uid_count", "uid_amt_mean", "card_count", "card_amt_mean", "email_count", "email_amt_mean",
                "uid_amt_std", "amt_to_uid_mean", "uid_prior_count"]


def normalize_columns(df):
    """Test-set identity columns use 'id-01'; train uses 'id_01'."""
    return df.rename(columns=lambda c: str(c).replace("id-", "id_"))


def add_time_features(df):
    out = df.copy()
    dt = pd.to_numeric(out["TransactionDT"], errors="coerce")
    hour = ((dt // 3600) % 24)
    out["TransactionHour"] = hour
    out["TransactionHour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["TransactionHour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def client_features(raw, ref, mode="new"):
    """Recompute the section-13 client features from stored reference statistics.

    mode="new":       rows are new transactions; reference statistics are *added to* the statistics of this batch.
    mode="reference": rows already belong to the reference data (e.g. rows of the competition files); only
                      reference statistics are used (batch statistics are the fallback for unseen clients).
    """
    amt = pd.to_numeric(raw["TransactionAmt"], errors="coerce").to_numpy(dtype="float64")
    keys = make_keys(raw)
    hashes = {k: hash_keys(keys[k]) for k in KEY_KINDS}
    out = {}
    uid_ref_n = None
    for kind in KEY_KINDS:
        h = hashes[kind]
        batch = key_stats(h, amt)
        b = batch.reindex(pd.Index(h)).to_numpy()            # n, s, ss of this batch per row
        r_df = ref[kind]
        pos = r_df.index.get_indexer(h)
        have = pos >= 0
        r = np.zeros((len(h), 3))
        r[have] = r_df.to_numpy()[pos[have]]
        if mode == "new":
            tot = r + b
        else:
            tot = np.where(r[:, [0]] > 0, r, b)
        mean, std = stats_to_features(tot[:, 0], tot[:, 1], tot[:, 2])
        out[f"{kind}_count"] = tot[:, 0]
        out[f"{kind}_amt_mean"] = mean
        if kind == "uid":
            out["uid_amt_std"] = std
            out["amt_to_uid_mean"] = amt / mean
            uid_ref_n = r[:, 0]
    # causal feature: earlier transactions of the same client
    dt = pd.to_numeric(raw["TransactionDT"], errors="coerce").to_numpy(dtype="float64")
    order = np.argsort(np.where(np.isnan(dt), np.inf, dt), kind="stable")
    cum = np.empty(len(raw), dtype="int64")
    cum[order] = pd.Series(hashes["uid"][order]).groupby(hashes["uid"][order]).cumcount().to_numpy()
    out["uid_prior_count"] = (uid_ref_n + cum) if mode == "new" else cum
    feats = pd.DataFrame(out, index=raw.index)
    for c in ("uid_count", "uid_prior_count", "card_count", "email_count"):
        feats[c] = feats[c].astype("int32")
    for c in feats.columns.difference(["uid_count", "uid_prior_count", "card_count", "email_count"]):
        feats[c] = feats[c].astype("float32")
    return feats


def prepare_features(raw, meta, ref, mode="new"):
    """Return (X, info). X has exactly meta['feature_names'] in order, categoricals aligned to training levels."""
    raw = normalize_columns(raw).reset_index(drop=True)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_required:
        raise ValueError(f"Missing required column(s): {', '.join(missing_required)}")
    base_names = [f for f in meta["feature_names"] if f not in meta["new_features"]]
    for c in CLIENT_COLUMNS:
        if c not in raw.columns:
            raw[c] = np.nan
    work = add_time_features(raw)
    cats = meta["categorical_columns"]

    missing_cols = [c for c in base_names if c not in work.columns]
    X = work.reindex(columns=base_names).copy()
    unseen = {}
    for c in base_names:
        if c in cats:
            s = X[c].astype("string")
            levels = meta["category_levels"][c]
            unseen[c] = float((s.notna() & ~s.isin(levels)).mean())
            X[c] = pd.Categorical(s.where(s.isin(levels)), categories=levels)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    cf = client_features(work, ref, mode)
    for f in meta["new_features"]:
        X[f] = cf[f].to_numpy()
    X = X[meta["feature_names"]]
    info = {
        "rows": len(X),
        "missing_columns": missing_cols,
        "unseen_category_share": {k: v for k, v in unseen.items() if v > 0},
        "d1_missing_share": float(pd.to_numeric(work["D1"], errors="coerce").isna().mean()),
        "clients_seen_in_reference_share": float((cf["uid_prior_count"] > 0).mean()),
    }
    return X, info
