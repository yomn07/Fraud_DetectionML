"""Unsupervised detectors in the app: rebuild the notebook's featurizer from saved parts and score rows.

UnsupFeaturizer below is the notebook's class (section 8) without changes to the maths, so saved parts
(freq_maps, imputer, scaler) can be re-attached.
"""
import numpy as np
import pandas as pd

DETECTOR_KEYS = {  # name -> key in the saved model dict
    "Isolation Forest": "if_model",
    "One-Class SVM (approx.)": "svm_model",
    "Autoencoder": "ae_model",
}


class UnsupFeaturizer:
    def __init__(self, num_cols, freq_cols):
        self.num_cols, self.freq_cols = num_cols, freq_cols

    @staticmethod
    def _key(s):
        return s.astype(object).where(s.notna(), "__nan__")

    def fit_freq(self, X):
        self.freq_maps = {c: self._key(X[c]).value_counts().to_dict() for c in self.freq_cols}
        return self

    def _raw(self, X):
        parts = []
        if self.num_cols:
            v = X[self.num_cols].astype("float64").to_numpy()
            parts.append(np.sign(v) * np.log1p(np.abs(v)))
        for c in self.freq_cols:
            cnt = self._key(X[c]).map(self.freq_maps[c]).fillna(0).astype("float64").to_numpy()
            parts.append(np.log1p(cnt)[:, None])
        return np.hstack(parts)

    def fit_scale(self, X):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        raw = self._raw(X)
        self.imputer = SimpleImputer(strategy="median").fit(raw)
        self.scaler = RobustScaler().fit(self.imputer.transform(raw))
        return self

    def transform(self, X):
        z = self.scaler.transform(self.imputer.transform(self._raw(X)))
        return np.clip(z, -5, 5).astype("float32")

    @classmethod
    def from_parts(cls, p):
        f = cls(p["num_cols"], p["freq_cols"])
        f.freq_maps, f.imputer, f.scaler = p["freq_maps"], p["imputer"], p["scaler"]
        return f


def available_detectors(unsup):
    return [n for n, k in DETECTOR_KEYS.items() if unsup["models"].get(k) is not None]


def score_unsup(unsup, X):
    """Raw anomaly scores (higher = more anomalous) for each available detector, exactly as in the notebook."""
    feat = UnsupFeaturizer.from_parts(unsup["parts"])
    Z = feat.transform(X)
    P = unsup["parts"]["pca"].transform(Z)
    m = unsup["models"]
    out = {}
    if m.get("if_model") is not None:
        out["Isolation Forest"] = -m["if_model"].decision_function(Z)
    if m.get("svm_model") is not None:
        out["One-Class SVM (approx.)"] = -m["svm_model"].decision_function(P)
    if m.get("ae_model") is not None:
        out["Autoencoder"] = ((Z - m["ae_model"].predict(Z)) ** 2).mean(axis=1)
    return out


def to_percentile(unsup, name, scores):
    """Where each score falls among the holdout-sample scores of the same detector (0-100)."""
    q = np.asarray(unsup["quantiles"][name], dtype="float64")
    grid = np.linspace(0, 100, len(q))
    return np.interp(scores, q, grid)
