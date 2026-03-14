"""
Risk Engine — Isolation Forest scoring microservice
Deploy: Railway / Render / Fly.io
"""
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import joblib
import numpy as np
import warnings
import os

app = FastAPI()

# โหลด model ครั้งเดียวตอน startup
MODEL_PATH = os.getenv("MODEL_PATH", "isolation_forest_model.pkl")
try:
    clf = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model at '{MODEL_PATH}': {e}")

API_SECRET = os.getenv("RISK_API_SECRET", "change-me")

# ── Normalization constants (calibrated จาก model จริง) ────────
# offset_ = decision boundary ของ IF
# SCORE_MOST_ANOMALOUS = empirical min จาก 50,000 random samples
SCORE_OFFSET         = clf.offset_           # -0.5000
SCORE_MOST_ANOMALOUS = -0.7330               # empirical min


class Stats(BaseModel):
    m: float = Field(ge=0, le=1)
    s: float = Field(ge=0, le=1)

class Features(BaseModel):
    density:    float = Field(ge=0, le=1)
    idle_ratio: float = Field(ge=0, le=1)

class BehaviorPayload(BaseModel):
    mouse:    Stats
    click:    Stats
    key:      Stats
    idle:     Stats
    features: Features


def to_vector(b: BehaviorPayload) -> list[float]:
    return [
        b.mouse.m,    b.mouse.s,
        b.click.m,    b.click.s,
        b.key.m,      b.key.s,
        b.idle.m,     b.idle.s,
        b.features.density,
        b.features.idle_ratio,
    ]


def normalize(raw_score: float) -> float:
    """
    Offset-based normalization:
      score >= offset  →  0.0  (normal)
      score <  offset  →  map [offset → most_anomalous] เป็น [0.0 → 1.0]
    """
    if raw_score >= SCORE_OFFSET:
        return 0.0
    return min(1.0, (SCORE_OFFSET - raw_score) / (SCORE_OFFSET - SCORE_MOST_ANOMALOUS))


@app.post("/score")
def score(
    payload: BehaviorPayload,
    x_api_key: str = Header(..., alias="x-api-key"),
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    vec = to_vector(payload)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = float(clf.score_samples([vec])[0])
    return {
        "raw_score":  round(raw, 4),
        "normalized": round(normalize(raw), 4),
    }


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "offset":  SCORE_OFFSET,
    }
