<h1 align="center">🛡️ Fraud Detection Studio</h1>

<p align="center">
  <strong>Credit-card fraud detection with LightGBM on the IEEE-CIS dataset</strong><br>
  Client-behaviour features, business-cost thresholds, SHAP reason codes,<br>
  an interactive Streamlit app and a FastAPI scoring service.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=flat" alt="LightGBM">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white" alt="Plotly">
</p>

<!-- Add a screenshot to docs/ and uncomment:
<p align="center"><img src="docs/app.png" width="85%" alt="Fraud Detection Studio"></p>
-->

---

## 🎯 Highlights

- **0.912 holdout AUC** and **62 % of frauds caught when reviewing only 5 % of transactions**
- **Client-ID features** (a reconstructed card-holder identity) lift holdout AP from 0.540 to **0.574**
- **Honest validation**: chronological 70 / 15 / 15 split, so the model is always tested on the future
- **Three decision thresholds**: best F2, lowest business cost (a missed fraud costs 300× a false alarm), and a 5 % analyst-capacity cap
- **Explainable alerts**: each flagged transaction comes with its top reason codes (SHAP contributions)
- **Supervised vs. unsupervised**: LightGBM compared with Isolation Forest, One-Class SVM and an autoencoder

---

## 📊 Results

Holdout set: the last 15 % of transactions in time (88,581 rows, 3.5 % fraud).

| Model | Valid AUC | Holdout AUC | Holdout AP | Recall @ 1 % alerts | Recall @ 5 % alerts |
|---|:-:|:-:|:-:|:-:|:-:|
| LightGBM baseline | 0.923 | 0.903 | 0.540 | 25.7 % | 58.2 % |
| **LightGBM + client features** ← *final model* | **0.937** | **0.912** | **0.574** | 25.4 % | **61.9 %** |

**Experiments that didn't make the cut**

| Experiment | Holdout AUC | Holdout AP | Verdict |
|---|:-:|:-:|---|
| Undersampling + SMOTE-NC | 0.907 | 0.535 | No gain over the baseline |
| Dropping 19 time-inconsistent features | 0.915 | 0.575 | Marginal |
| XGBoost | 0.887 | 0.359 | Clearly worse |
| CatBoost | 0.903 | 0.568 | Close, but below LightGBM |
| Rank blend of LightGBM / XGBoost / CatBoost | 0.912 | 0.573 | No gain; LightGBM alone kept |

---

## 🧠 How a Transaction Is Scored

```
raw transaction ─▶ feature pipeline ─▶ client-ID features ─▶ LightGBM ─▶ fraud probability
                    (categories,        (uid / card / email:       │
                     IEEE-CIS cols)      count, mean amount,       ├─▶ threshold (F2 · cost · 5 % cap) ─▶ flagged?
                                         std, amount vs. mean…)    └─▶ SHAP contributions ─▶ top reason codes
```

The **client-ID features** group transactions by a reconstructed card-holder key (`uid` = card details + address + the day the account started, derived from `D1`), by card and by email domain, then describe each transaction relative to that client's history (how many prior transactions, average amount, how unusual this amount is).

---

## 🖥️ The Streamlit App

| Page | What it shows |
|---|---|
| **Overview** | Model comparison, what was tried, how a transaction is scored |
| **Data exploration (EDA)** | Fraud distribution and amounts; upload your own IEEE-CIS CSVs |
| **Model performance** | ROC / PR curves and metrics on validation and holdout |
| **Supervised vs. unsupervised** | LightGBM vs. Isolation Forest, One-Class SVM, autoencoder |
| **Threshold & business cost** | Interactive simulator: cost of false positives vs. missed frauds, analyst capacity |
| **Score transactions** | Upload transactions, see the highest-risk ones, explain any alert |
| **Feature insights** | SHAP beeswarm, client-ID features, time-consistency check |
| **Method, limits & checklist** | Validation choices, known limits, rejected experiments |

---

## 🔌 Scoring API

```bash
uvicorn api:app --reload        # docs at http://localhost:8000/docs
```

| Endpoint | Description |
|---|---|
| `GET /health` | Status and number of features |
| `GET /meta` | Thresholds, cost settings, feature list, split |
| `POST /score` | Scores up to 5,000 transactions and returns reason codes for flagged ones |

```json
POST /score
{
  "transactions": [{ "TransactionDT": 86400, "TransactionAmt": 117.0, "ProductCD": "W", "card1": 13926 }],
  "rule": "f2",
  "explain": true
}
```
`rule` can be `f2`, `cost` or `capped`, or pass your own `threshold` (0–1).

---

## 🚀 Getting Started

```bash
git clone https://github.com/yomn07/Fraud_DetectionML.git
cd Fraud_DetectionML
pip install -r requirements.txt
streamlit run app.py
```

The trained model and its data are in `artifacts/`. If that folder is missing, the app falls back to a small **demo** model in `demo_artifacts/`. You can also point the app to another artifacts folder with the `FRAUD_ARTIFACTS` environment variable, or to a zip of one with `ARTIFACT_URL`.

**Rebuilding the artifacts:** the model is trained in a Kaggle notebook on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) data. Paste `tools/notebook_export_cell.py` at the end of the notebook to export everything the app needs.

**Tests:**
```bash
pytest tests/
```

---

## 📁 Project Structure

```
Fraud_DetectionML/
├── app.py               # Streamlit entry point (8 pages)
├── app_pages/           # One file per page
├── api.py               # FastAPI scoring service
├── core/                # Features, client keys, scoring, metrics, unsupervised detectors
├── artifacts/           # Trained LightGBM model + reference data
├── demo_artifacts/      # Small demo model and sample transactions
├── tools/               # Notebook export cell, artifact builders
└── tests/               # Pipeline, pages and export tests
```

---

## 👩‍💻 Author

**Yomn Toumia** — [@yomn07](https://github.com/yomn07)
