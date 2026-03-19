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
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"] if isinstance(bundle, dict) else bundle
    TRAIN_SCORE_MIN = float(bundle.get("train_score_min", -0.67)) if isinstance(bundle, dict) else -0.67
    TRAIN_SCORE_MAX = float(bundle.get("train_score_max", -0.41)) if isinstance(bundle, dict) else -0.41
except Exception as e:
    raise RuntimeError(f"Cannot load model at '{MODEL_PATH}': {e}")

API_SECRET = os.getenv("RISK_API_SECRET", "change-me")


class Stats(BaseModel):
    m: float = Field(ge=0, le=1)
    s: float = Field(ge=0, le=1)


class Features(BaseModel):
    density: float = Field(ge=0, le=1)
    idle_ratio: float = Field(ge=0, le=1)


class BehaviorPayload(BaseModel):
    mouse: Stats
    click: Stats
    key: Stats
    idle: Stats
    features: Features


def to_vector(b: BehaviorPayload) -> list[float]:
    return [
        b.mouse.m, b.mouse.s,
        b.click.m, b.click.s,
        b.key.m, b.key.s,
        b.idle.m, b.idle.s,
        b.features.density,
        b.features.idle_ratio,
    ]


def normalize(raw_score: float) -> float:
    # IsolationForest: raw score ยิ่งต่ำ = ยิ่งผิดปกติ
    denom = TRAIN_SCORE_MAX - TRAIN_SCORE_MIN
    if denom <= 0:
        return 0.0

    risk = (TRAIN_SCORE_MAX - raw_score) / denom
    return float(np.clip(risk, 0.0, 1.0))


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
        decision = float(clf.decision_function([vec])[0])

    return {
        "raw_score": round(raw, 4),
        "decision": round(decision, 4),
        "normalized": round(normalize(raw), 4),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "train_score_min": round(TRAIN_SCORE_MIN, 4),
        "train_score_max": round(TRAIN_SCORE_MAX, 4),
    }
