"""Memory-safe scoring of large CSVs (e.g. the 0.6 GB train_transaction.csv).

Pass 1 reads only the few columns the client features need and computes them for the whole file at once
(so counts and the prior-transaction count are exact for the file). Pass 2 streams the full file in chunks,
builds the model features, scores them and keeps only scores, a few display columns and the feature rows of the
highest-scoring transactions (for explanations).
"""
import numpy as np
import pandas as pd

from core.features import CLIENT_COLUMNS, attach_client, client_features, normalize_columns, prepare_base
from core.unsup import available_detectors, score_unsup

DISPLAY = ["TransactionID", "TransactionAmt", "ProductCD", "P_emaildomain", "isFraud"]
LIGHT = ["TransactionID", "TransactionDT", "TransactionAmt"] + CLIENT_COLUMNS


def _read(src, **kw):
    if hasattr(src, "seek"):
        src.seek(0)
    return pd.read_csv(src, **kw)


def count_rows(src):
    n = 0
    for ch in _read(src, usecols=[0], chunksize=500_000):
        n += len(ch)
    return n


def score_stream(art, tx_src, id_src=None, mode="new", chunk_rows=50_000, keep_top=300, with_unsup=True, progress=None):
    meta = art.meta
    header = list(_read(tx_src, nrows=0).columns)
    usecols = [c for c in LIGHT if c in header]
    light = _read(tx_src, usecols=usecols)
    cf = client_features(normalize_columns(light).assign(**{c: np.nan for c in CLIENT_COLUMNS if c not in light.columns}),
                         art.ref, mode)
    n_total = len(light)
    seen_share = float((cf["uid_prior_count"] > 0).mean())
    del light

    ident = None
    if id_src is not None:
        ident = normalize_columns(_read(id_src))

    scores = np.empty(n_total)
    use_unsup = with_unsup and art.unsup is not None
    unsup_scores = {n: np.empty(n_total) for n in available_detectors(art.unsup)} if use_unsup else {}
    display, tops = [], []
    pos, infos = 0, []
    for chunk in _read(tx_src, chunksize=chunk_rows, low_memory=False):
        chunk = normalize_columns(chunk)
        if ident is not None and "TransactionID" in chunk.columns:
            chunk = chunk.merge(ident, on="TransactionID", how="left")
        Xb, info, _ = prepare_base(chunk, meta)
        k = len(Xb)
        X = attach_client(Xb, cf.iloc[pos:pos + k].reset_index(drop=True), meta)
        s = art.booster.predict(X)
        scores[pos:pos + k] = s
        if use_unsup:
            for name, v in score_unsup(art.unsup, X).items():
                unsup_scores[name][pos:pos + k] = v
        display.append(chunk[[c for c in DISPLAY if c in chunk.columns]].reset_index(drop=True))
        top = np.argsort(-s)[:keep_top]
        tops.append(X.iloc[top].assign(_row=pos + top))
        infos.append(info)
        pos += k
        if progress:
            progress(min(1.0, pos / n_total))
    scores, display = scores[:pos], pd.concat(display, ignore_index=True)
    top_X = pd.concat(tops).sort_values("_row")
    top_X = top_X.set_index("_row")
    agg = {
        "rows": pos,
        "missing_columns": infos[0]["missing_columns"],
        "unseen_category_share": {c: float(np.average([i["unseen_category_share"].get(c, 0.0) for i in infos],
                                                      weights=[i["rows"] for i in infos]))
                                  for c in set().union(*[i["unseen_category_share"] for i in infos])},
        "d1_missing_share": float(np.average([i["d1_missing_share"] for i in infos], weights=[i["rows"] for i in infos])),
        "clients_seen_in_reference_share": seen_share,
    }
    return {"scores": scores, "display": display, "top_X": top_X, "info": agg,
            "unsup": {k: v[:pos] for k, v in unsup_scores.items()}}
