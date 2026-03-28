"""FastAPI service: /predict, /predict_with_diagnosis, /logs, /model_info."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import init_db, log_prediction, recent_logs
from feature_engineering import (
    FEATURE_LIST,
    PWF_KW_MAX,
    PWF_KW_MIN,
    build_feature_row,
    explanation_factors,
    row_to_vector,
)
from llm_helper import diagnosis_bundle

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
MODELS_DIR = BACKEND_DIR.parent / "models"
load_dotenv(BACKEND_DIR.parent / ".env")
load_dotenv(BACKEND_DIR / ".env")

MODEL_SOURCE = "unknown"


def _load_model_artifacts():
    global MODEL_SOURCE
    try:
        m = joblib.load(BACKEND_DIR / "model.pkl")
        MODEL_SOURCE = "backend/model.pkl"
        return m
    except Exception:
        rf_path = MODELS_DIR / "random_forest.pkl"
        if rf_path.is_file():
            MODEL_SOURCE = "models/random_forest.pkl (XGBoost/model.pkl unavailable)"
            return joblib.load(rf_path)
        raise RuntimeError(
            "Model artifacts missing. Run: python scripts/train_model.py from project root."
        ) from None


try:
    model = _load_model_artifacts()
    scaler = joblib.load(BACKEND_DIR / "scaler.pkl")
    label_encoder: object = joblib.load(BACKEND_DIR / "label_encoder.pkl")
    FEATURES: list[str] = joblib.load(BACKEND_DIR / "feature_list.pkl")
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "Model artifacts missing. Run: python scripts/train_model.py from project root."
    ) from exc

app = FastAPI(title="Tractor Predictive Maintenance API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TractorInput(BaseModel):
    air_temp: float = Field(..., ge=200.0, le=400.0, description="Ambient air temperature [K]")
    process_temp: float = Field(..., ge=200.0, le=400.0, description="Machine operating temperature [K]")
    rpm: float = Field(..., ge=0.0, le=12000.0, description="Rotational speed [rpm]")
    torque: float = Field(..., ge=0.0, le=800.0, description="Torque [Nm]")
    tool_wear: float = Field(..., ge=0.0, le=5000.0, description="Component wear time [min]")


def _health_band(label: str) -> str:
    if label == "Healthy":
        return "Healthy"
    if label in ("Heat Dissipation Failure", "Power Failure", "Overstrain Failure"):
        return "Critical"
    return "Warning"


def _failure_probability(proba: np.ndarray, classes: list[str]) -> float:
    if "Healthy" in classes:
        hi = classes.index("Healthy")
        return float(1.0 - proba[hi])
    return float(np.max(proba))


def _compute(data: TractorInput) -> dict[str, Any]:
    row = build_feature_row(
        data.air_temp,
        data.process_temp,
        data.rpm,
        data.torque,
        data.tool_wear,
    )
    X = row_to_vector(row, FEATURES)
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    label = str(label_encoder.inverse_transform([int(pred)])[0])
    band = _health_band(label)
    out: dict[str, Any] = {
        "prediction": label,
        "health_status": band,
        "explanation_factors": explanation_factors(row),
        "derived": {
            "temp_diff_k": round(float(row["temp_diff"]), 4),
            "power_kw": round(float(row["power_kw"]), 4),
            "wear_torque": round(float(row["wear_torque"]), 4),
        },
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)[0]
        classes = [str(c) for c in getattr(label_encoder, "classes_", [])]
        fp = _failure_probability(proba, classes)
        out["failure_probability"] = round(fp, 4)
        out["class_probabilities"] = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    else:
        out["failure_probability"] = 0.0 if label == "Healthy" else 0.75
    return out


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model_info")
def model_info() -> dict[str, Any]:
    return {
        "model_source": MODEL_SOURCE,
        "estimator_class": type(model).__name__,
        "n_features": len(FEATURES),
        "feature_names": FEATURES,
        "power_formula": "P_kW = (torque_Nm * rpm) / 9549",
        "pwf_reference_band_kw": [PWF_KW_MIN, PWF_KW_MAX],
        "note": "Probabilities are model outputs (not Platt-calibrated unless you add calibration).",
    }


@app.post("/predict")
def predict(data: TractorInput) -> dict[str, Any]:
    base = _compute(data)
    log_id = log_prediction(
        air_temp=data.air_temp,
        process_temp=data.process_temp,
        rpm=data.rpm,
        torque=data.torque,
        tool_wear=data.tool_wear,
        prediction=base["prediction"],
        health_status=base["health_status"],
        failure_probability=float(base["failure_probability"]),
        llm_advice=None,
    )
    return {**base, "log_id": log_id}


@app.post("/predict_with_diagnosis")
def predict_with_diagnosis(data: TractorInput) -> dict[str, Any]:
    base = _compute(data)
    bundle = diagnosis_bundle(
        air_temp=data.air_temp,
        process_temp=data.process_temp,
        rpm=data.rpm,
        torque=data.torque,
        tool_wear=data.tool_wear,
        prediction=base["prediction"],
        health_status=base["health_status"],
        failure_probability=float(base["failure_probability"]),
    )
    log_id = log_prediction(
        air_temp=data.air_temp,
        process_temp=data.process_temp,
        rpm=data.rpm,
        torque=data.torque,
        tool_wear=data.tool_wear,
        prediction=base["prediction"],
        health_status=base["health_status"],
        failure_probability=float(base["failure_probability"]),
        llm_advice=bundle["llm_diagnosis"],
    )
    return {
        **base,
        "llm_diagnosis": bundle["llm_diagnosis"],
        "manual_citations": bundle["manual_citations"],
        "log_id": log_id,
    }


@app.get("/logs")
def logs(
    limit: int = 50,
    offset: int = 0,
    prediction_contains: Optional[str] = None,
    health_status: Optional[str] = None,
) -> dict[str, Any]:
    items, total = recent_logs(
        limit=limit,
        offset=offset,
        prediction_contains=prediction_contains,
        health_status=health_status,
    )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


_static = FRONTEND_DIR / "static"
if _static.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_static)), name="assets")


@app.get("/")
def spa_index() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail="frontend/index.html missing")
    return FileResponse(index)
