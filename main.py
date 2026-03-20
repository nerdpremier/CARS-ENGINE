"""
Risk Engine — Isolation Forest scoring microservice
Decision-centered normalization + rule-based auto-click boost.

แนวคิด:
1) ใช้ Isolation Forest ให้คะแนนพฤติกรรมรวม
2) ใช้ decision_function() เป็นแกน normalize เพราะ decision=0 คือเส้นแบ่ง anomaly ของโมเดล
3) เพิ่ม rule-based detector สำหรับ auto click
4) ถ้าเข้าเงื่อนไข auto click ให้ boost score ขึ้นทันที

Output meaning:
- raw_score   : คะแนนดิบจาก Isolation Forest
- decision    : ค่าจาก decision_function()
                > 0 = ปกติ, < 0 = ฝั่ง anomaly
- base_score  : score ที่ normalize จาก decision อย่างเดียว
- normalized  : final score หลังรวม auto-click rule
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import joblib
import numpy as np
import warnings
import os

app = FastAPI(title="Risk Engine", version="4.0-decision-autoclick")

MODEL_PATH = os.getenv("MODEL_PATH", "isolation_forest_model.pkl")
API_SECRET = os.getenv("RISK_API_SECRET", "change-me")

# scale ยิ่งมาก = score นุ่มขึ้น
# scale ยิ่งน้อย = score ชัน/ไวขึ้น
RISK_SCALE = float(os.getenv("RISK_SCALE", "0.12"))
EPS = 1e-12

try:
    loaded = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model at '{MODEL_PATH}': {e}")

# รองรับทั้งกรณี bundle dict และ model ตรง ๆ
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


def normalize_from_decision(decision: float) -> float:
    """
    แปลง decision_function() เป็น risk score ในช่วง [0, 1]

    Isolation Forest:
    - decision > 0  => ปกติ
    - decision = 0  => เส้นแบ่งของโมเดล
    - decision < 0  => ผิดปกติ

    ใช้ sigmoid กลับด้าน:
        risk = 1 / (1 + exp(decision / scale))

    ผลลัพธ์:
    - decision บวกมาก  -> risk ใกล้ 0
    - decision ใกล้ 0   -> risk ใกล้ 0.5
    - decision ติดลบมาก -> risk ใกล้ 1
    """
    scale = max(RISK_SCALE, EPS)
    risk = 1.0 / (1.0 + np.exp(decision / scale))
    return float(np.clip(risk, 0.0, 1.0))


def detect_auto_click_rule(b: BehaviorPayload) -> dict:
    """
    ตรวจ pattern auto click แบบ rule-based

    เหตุผลที่ใช้ rule เพิ่ม:
    - auto click เป็น signature เฉพาะ
    - unsupervised model อาจไม่ไวพอในบางเคส
    - จึงใช้กฎช่วย boost score ให้ตอบสนองเร็วขึ้น
    """
    reasons = []
    severity = 0.0

    click_m = b.click.m
    click_s = b.click.s
    mouse_m = b.mouse.m
    mouse_s = b.mouse.s
    key_m = b.key.m
    idle_ratio = b.features.idle_ratio
    density = b.features.density

    # 1) click สูง แต่ mouse activity ต่ำมาก
    if click_m >= 0.75 and mouse_m <= 0.10:
        severity += 0.35
        reasons.append("high_click_low_mouse")

    # 2) click สูงและสม่ำเสมอมากผิดธรรมชาติ
    if click_m >= 0.70 and click_s >= 0.90:
        severity += 0.25
        reasons.append("high_click_high_regular")

    # 3) click สูง แต่ keyboard activity ต่ำมาก
    if click_m >= 0.70 and key_m <= 0.05:
        severity += 0.15
        reasons.append("high_click_low_key")

    # 4) active ต่อเนื่องมาก แทบไม่ idle
    if click_m >= 0.65 and idle_ratio <= 0.02 and density >= 0.80:
        severity += 0.20
        reasons.append("continuous_dense_clicking")

    # 5) mouse variation ต่ำมาก แต่ click สูง
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
    """
    รวมคะแนนจากโมเดลและกฎ auto click

    แนวคิด:
    - ถ้ายังไม่เจอ auto click -> ใช้ base_score
    - ถ้าเจอ -> boost ตาม severity
    - ถ้า severity สูงมาก -> ดันขั้นต่ำขึ้นทันที
    """
    final_score = base_score

    if auto_click["detected"]:
        severity = auto_click["severity"]

        # boost แบบนุ่ม ๆ ก่อน
        final_score = min(1.0, base_score + severity * 0.5)

        # ถ้ารุนแรงมาก ดันขั้นต่ำขึ้นเลย
        if severity >= 0.60:
            final_score = max(final_score, 1)
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

    base_score = normalize_from_decision(decision)
    auto_click = detect_auto_click_rule(payload)
    final_score = combine_scores(base_score, auto_click)

    return {
        "raw_score": round(raw, 6),
        "decision": round(decision, 6),
        "base_score": round(base_score, 6),
        "normalized": round(final_score, 6),
        "risk_scale": RISK_SCALE,
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
        "risk_scale": RISK_SCALE,
        "normalization_mode": "decision_sigmoid_centered_at_zero",
        "rule_engine": "auto_click_boost_enabled",
    }