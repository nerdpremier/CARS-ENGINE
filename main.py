"""
Risk Engine — Isolation Forest scoring microservice
Decision-centered normalization for streaming behavior scoring.

Why this version:
- The previous min/max or pseudo-percentile normalization still stayed high
  because the model's raw score distribution is very compressed.
- For IsolationForest, the real anomaly boundary is better represented by
  decision_function(), where:
      decision > 0  => normal side
      decision < 0  => anomalous side
- This service keeps normalization, but normalizes around the model boundary
  instead of around train min/max extremes.

Output meaning:
- normalized close to 0.0 => normal behavior
- normalized around 0.5   => near model boundary
- normalized close to 1.0 => increasingly anomalous
"""
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import joblib
import numpy as np
import warnings
import os

app = FastAPI(title="Risk Engine", version="3.1-decision-normalized")

MODEL_PATH = os.getenv("MODEL_PATH", "isolation_forest_model.pkl")
API_SECRET = os.getenv("RISK_API_SECRET", "change-me")

# Controls how quickly normalized score rises around the model boundary.
# Smaller value = smoother / less sensitive.
# Larger value = steeper / more sensitive.
RISK_SCALE = float(os.getenv("RISK_SCALE", "0.06"))
EPS = 1e-12

try:
    loaded = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model at '{MODEL_PATH}': {e}")

# Support both bundle-dict and direct model object
if isinstance(loaded, dict):
    bundle = loaded
    clf = bundle["model"] if "model" in bundle else loaded
else:
    bundle = {}
    clf = loaded

FEATURE_COLS = bundle.get("feature_cols") if isinstance(bundle, dict) else None
TRAIN_SCORE_MIN = float(bundle.get("train_score_min", -0.6667140844647449))
TRAIN_SCORE_MAX = float(bundle.get("train_score_max", -0.40973622752350913))
TRAIN_SCORE_RANGE = float(bundle.get("train_score_range", TRAIN_SCORE_MAX - TRAIN_SCORE_MIN))
MODEL_OFFSET = float(getattr(clf, "offset_", np.nan)) if hasattr(clf, "offset_") else np.nan


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
    vec = [
        b.mouse.m, b.mouse.s,
        b.click.m, b.click.s,
        b.key.m, b.key.s,
        b.idle.m, b.idle.s,
        b.features.density,
        b.features.idle_ratio,
    ]

    if FEATURE_COLS is not None and len(FEATURE_COLS) != len(vec):
        raise HTTPException(status_code=500, detail="Feature dimension mismatch with trained model")

    return vec


def normalize_from_decision(decision: float) -> float:
    """
    Convert decision_function output into [0, 1] risk score.

    decision_function meaning in IsolationForest:
      decision > 0  => normal
      decision = 0  => model boundary
      decision < 0  => anomaly side

    We map it with a sigmoid centered at 0 and flipped so that:
      - strongly positive decision => near 0
      - near zero                 => around 0.5
      - strongly negative         => near 1

    Formula:
      risk = 1 / (1 + exp(decision / scale))

    scale is configurable via RISK_SCALE.
    """
    scale = max(RISK_SCALE, EPS)
    risk = 1.0 / (1.0 + np.exp(decision / scale))
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
        "raw_score": round(raw, 6),
        "decision": round(decision, 6),
        "normalized": round(normalize_from_decision(decision), 6),
        "risk_scale": RISK_SCALE,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "has_bundle": isinstance(loaded, dict),
        "has_feature_cols": FEATURE_COLS is not None,
        "train_score_min": round(TRAIN_SCORE_MIN, 6),
        "train_score_max": round(TRAIN_SCORE_MAX, 6),
        "train_score_range": round(TRAIN_SCORE_RANGE, 6),
        "model_offset": None if np.isnan(MODEL_OFFSET) else round(MODEL_OFFSET, 6),
        "risk_scale": RISK_SCALE,
        "normalization_mode": "decision_sigmoid_centered_at_zero",
    }