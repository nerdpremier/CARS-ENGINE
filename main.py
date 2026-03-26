
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import joblib
import numpy as np
import warnings
import os

app = FastAPI(title="Risk Engine", version="5.0-rawscore-pivot-autoclick")

MODEL_PATH = os.getenv("MODEL_PATH", "isolation_forest_model.pkl")
API_SECRET = os.getenv("RISK_API_SECRET", "change-me")

PIVOT_RATIO = float(os.getenv("PIVOT_RATIO", "0.10"))
ANOMALY_GAMMA = float(os.getenv("ANOMALY_GAMMA", "1.5"))

EPS = 1e-12

try:
    loaded = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model at '{MODEL_PATH}': {e}")

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
        raise HTTPException(
            status_code=500,
            detail="Feature dimension mismatch with trained model"
        )

    return vec

def normalize_from_raw(raw_score: float) -> float:
    
    train_range = max(TRAIN_SCORE_RANGE, EPS)
    pivot = TRAIN_SCORE_MIN + (train_range * PIVOT_RATIO)

    denom = max(TRAIN_SCORE_MAX - pivot, EPS)

    if raw_score >= pivot:
        value = 0.5 * (TRAIN_SCORE_MAX - raw_score) / denom

    else:
        excess = (pivot - raw_score) / train_range
        boosted = np.power(excess, 1.0 / max(ANOMALY_GAMMA, EPS))
        value = 0.5 + 0.5 * boosted

    return float(np.clip(value, 0.0, 1.0))

def detect_auto_click_rule(b: BehaviorPayload) -> dict:
    
    reasons = []
    severity = 0.0

    click_m = b.click.m
    click_s = b.click.s
    mouse_m = b.mouse.m
    mouse_s = b.mouse.s
    key_m = b.key.m
    idle_ratio = b.features.idle_ratio
    density = b.features.density

    if click_m >= 0.75 and mouse_m <= 0.10:
        severity += 0.35
        reasons.append("high_click_low_mouse")

    if click_m >= 0.70 and click_s >= 0.90:
        severity += 0.25
        reasons.append("high_click_high_regular")

    if click_m >= 0.70 and key_m <= 0.05:
        severity += 0.15
        reasons.append("high_click_low_key")

    if click_m >= 0.65 and idle_ratio <= 0.02 and density >= 0.80:
        severity += 0.20
        reasons.append("continuous_dense_clicking")

    if click_m >= 0.70 and mouse_s <= 0.03:
        severity += 0.20
        reasons.append("high_click_low_mouse_variation")

    severity = min(severity, 1.0)
    detected = severity >= 0.25

    return {
        "detected": detected,
        "severity": float(round(severity, 6)),
        "reasons": reasons,
    }

def combine_scores(base_score: float, auto_click: dict) -> float:
    
    final_score = base_score

    if auto_click["detected"]:
        severity = auto_click["severity"]

        final_score = min(1.0, base_score + severity * 0.5)

        if severity >= 0.60:
            final_score = max(final_score, 1.0)
        elif severity >= 0.35:
            final_score = max(final_score, 0.9)

    return float(np.clip(final_score, 0.0, 1.0))

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

    base_score = normalize_from_raw(raw)
    auto_click = detect_auto_click_rule(payload)
    final_score = combine_scores(base_score, auto_click)

    return {
        "raw_score": round(raw, 6),
        "decision": round(decision, 6),
        "base_score": round(base_score, 6),
        "normalized": round(final_score, 6),
        "pivot_ratio": PIVOT_RATIO,
        "anomaly_gamma": ANOMALY_GAMMA,
        "auto_click_detected": auto_click["detected"],
        "auto_click_severity": auto_click["severity"],
        "auto_click_reasons": auto_click["reasons"],
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
        "pivot_ratio": PIVOT_RATIO,
        "anomaly_gamma": ANOMALY_GAMMA,
        "normalization_mode": "raw_score_pivot_power_curve",
        "rule_engine": "auto_click_boost_enabled",
    }
