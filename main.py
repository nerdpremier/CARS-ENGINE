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

MODEL_PATH = os.getenv("MODEL_PATH", "isolation_forest_model.pkl")
try:
    clf = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model at '{MODEL_PATH}': {e}")

API_SECRET = os.getenv("RISK_API_SECRET", "change-me")

# ── Normalization (sigmoid + min-max) ─────────────────────────
# 1. sigmoid(raw) = 1 / (1 + exp(raw))  → เหมือน predict.txt
# 2. min-max normalize sigmoid ให้เป็น [0, 1]
#    SIG_MIN = sigmoid ของ normal สุด (ค่าต่ำ)
#    SIG_MAX = sigmoid ของ anomalous สุด (ค่าสูง)
# empirical จาก 50,000 random samples
SIG_MIN = 0.6291
SIG_MAX = 0.6777


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
    sigmoid + min-max normalization:
      0.0 = normal สุด
      1.0 = anomalous สุด
    """
    sig = 1.0 / (1.0 + np.exp(raw_score))
    return float(np.clip((sig - SIG_MIN) / (SIG_MAX - SIG_MIN), 0.0, 1.0))


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


# แปลง normalized score → action (LOW/MEDIUM/REVOKE) ใน engine
RISK_MEDIUM_THRESHOLD = float(os.getenv("RISK_MEDIUM_THRESHOLD", "0.5"))
RISK_REVOKE_THRESHOLD = float(os.getenv("RISK_REVOKE_THRESHOLD", "0.85"))


@app.post("/decision")
def decision(
    payload: BehaviorPayload,
    x_api_key: str = Header(..., alias="x-api-key"),
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    vec = to_vector(payload)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = float(clf.score_samples([vec])[0])

    normalized = normalize(raw)

    if normalized >= RISK_REVOKE_THRESHOLD:
        action = "revoke"
    elif normalized >= RISK_MEDIUM_THRESHOLD:
        action = "medium"
    else:
        action = "low"

    return {
        "action":      action,
        "raw_score":   round(raw, 4),
        "normalized":  round(normalized, 4),
        "thresholds": {
            "medium": RISK_MEDIUM_THRESHOLD,
            "revoke": RISK_REVOKE_THRESHOLD,
        },
    }


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "sig_min": SIG_MIN,
        "sig_max": SIG_MAX,
    }
