"""Evaluation helpers (pure numpy/pandas/sklearn)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def topk_recall(y, p, pct):
    k = max(1, int(np.ceil(pct / 100 * len(y))))
    order = np.argsort(-p)[:k]
    return float(y[order].sum() / max(1, y.sum()))


def summary(y, p):
    return {"AUC": roc_auc_score(y, p), "AP": average_precision_score(y, p),
            "Recall @1%": topk_recall(y, p, 1), "Recall @5%": topk_recall(y, p, 5)}


def confusion(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f2 = 5 * prec * rec / max(1e-12, 4 * prec + rec)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "alerts": tp + fp, "precision": prec, "recall": rec, "F2": f2}


def cost_curve(y, p, c_fp, c_fn, max_points=400):
    """Total cost when flagging the top-k transactions, for a grid of k (efficient, exact at each k)."""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    cost = c_fp * fp + c_fn * (ys.sum() - tp)
    idx = np.unique(np.linspace(0, len(ys) - 1, max_points).astype(int))
    return pd.DataFrame({"threshold": ps[idx], "alert_rate": (idx + 1) / len(ys), "cost": cost[idx],
                         "recall": tp[idx] / max(1, ys.sum())})


def best_threshold(y, p, c_fp, c_fn, max_rate=1.0):
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    k_max = max(1, int(max_rate * len(ys)))
    tp = np.cumsum(ys)[:k_max]
    fp = np.arange(1, k_max + 1) - tp
    cost = c_fp * fp + c_fn * (ys.sum() - tp)
    i = int(np.argmin(cost))
    if c_fn * ys.sum() <= cost[i]:
        return float("inf")
    return float(ps[i])


def gain_curve(y, p, points=200):
    order = np.argsort(-p)
    ys = y[order]
    cum = np.cumsum(ys) / max(1, ys.sum())
    idx = np.unique(np.linspace(0, len(ys) - 1, points).astype(int))
    return (idx + 1) / len(ys), cum[idx]


def calibration(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    d = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return d.reset_index(drop=True)


__all__ = ["summary", "confusion", "cost_curve", "best_threshold", "gain_curve", "calibration",
           "roc_curve", "precision_recall_curve", "topk_recall"]
