"""REST API for the fraud model.

    pip install -r requirements-api.txt
    uvicorn api:app --port 8000          # docs at http://localhost:8000/docs

POST /score  {"transactions": [{...raw IEEE-CIS columns...}], "mode": "new", "rule": "f2", "explain": true}
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.scoring import decision_thresholds, explain_row, reason_codes, score_frame
from core.store import load_from_dir, valid_dir

ROOT = Path(__file__).resolve().parent


def _find_dir():
    for c in [os.environ.get("FRAUD_ARTIFACTS"), ROOT / "artifacts", ROOT / "demo_artifacts"]:
        if c and valid_dir(c):
            return Path(c)
    raise RuntimeError("No artifacts found: set FRAUD_ARTIFACTS or put the export in ./artifacts")


ART = load_from_dir(_find_dir())
RULES = decision_thresholds(ART.meta)
RULE_KEYS = {"f2": "F2-optimal (validation)", "cost": "Cost-optimal (validation)", "capped": "Alert-capped cost-optimal (validation)"}

app = FastAPI(title="Fraud detection API", version="1.0",
              description="Scores IEEE-CIS style transactions with the LightGBM + client-features model and explains each alert.")


class ScoreRequest(BaseModel):
    transactions: List[Dict[str, Any]] = Field(..., min_length=1, max_length=5000,
                                               description="Raw transaction rows. TransactionDT and TransactionAmt are required.")
    mode: Literal["new", "reference"] = "new"
    rule: Literal["f2", "cost", "capped"] = "f2"
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Overrides `rule` when set.")
    explain: bool = True


class ReasonCode(BaseModel):
    code: str
    feature: str
    value: str
    effect: float


class Scored(BaseModel):
    score: float
    flagged: bool
    reason_codes: Optional[List[ReasonCode]] = None


class ScoreResponse(BaseModel):
    threshold: float
    results: List[Scored]


@app.get("/health")
def health():
    return {"status": "ok", "demo": ART.is_demo, "features": len(ART.meta["feature_names"])}


@app.get("/meta")
def meta():
    m = ART.meta
    return {"thresholds": m["thresholds"], "cost": m["cost"], "features": m["feature_names"], "created": m.get("created"),
            "split": m.get("split"), "demo": ART.is_demo}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    key = RULE_KEYS[req.rule]
    if req.threshold is None and key not in RULES:
        raise HTTPException(400, f"Rule '{req.rule}' is not available for this model (flagging nothing is cheapest).")
    thr = float(req.threshold if req.threshold is not None else RULES[key])
    try:
        scores, X, _ = score_frame(ART, pd.DataFrame(req.transactions), req.mode)
    except ValueError as e:
        raise HTTPException(422, str(e))
    out = []
    for i, s in enumerate(scores):
        item = {"score": float(s), "flagged": bool(s >= thr)}
        if req.explain and item["flagged"]:
            table, _, _ = explain_row(ART, X.iloc[[i]])
            item["reason_codes"] = [{**r, "value": str(r["value"])} for r in reason_codes(table)]
        out.append(item)
    return {"threshold": thr, "results": out}
