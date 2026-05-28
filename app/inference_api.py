"""
inference_api.py  --  Step 6/7: model deployment for inference (FastAPI)
========================================================================
MLOps purpose: DEPLOYMENT / SERVING. Loads the registered model pipeline and
exposes REST endpoints so the model can be consumed by any client. The pipeline
includes preprocessing, so callers send RAW student feature values.

Run:  uvicorn app.inference_api:app --reload --port 8000
Docs: http://localhost:8000/docs   (interactive Swagger UI -- great for the demo)
"""
import json
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import config

app = FastAPI(title="Student Performance Risk API", version="1.0")

# Load once at startup (the deployed artifact -- not the MLflow server)
_model = joblib.load(config.MODEL_PATH)
_meta = json.loads(config.MODEL_META_PATH.read_text())


class Student(BaseModel):
    Age: int = 21
    Study_Hours_Per_Day: float = 4.0
    Sleep_Hours_Per_Day: float = 7.0
    Attendance_Rate: float = 80.0
    Stress_Level: float = 4.0
    Gender: str = "Female"
    Major: str = "STEM"
    Living_Area: str = "Urban"
    Parent_Education: str = "Bachelor"
    Scholarship_Status: str = "No"
    Part_Time_Job: str = "No"
    Extracurricular_Activities: str = "No"
    Internet_Access: str = "Yes"
    Exam_Proximity: str = "No"


class PredictionOut(BaseModel):
    predicted_score: float
    at_risk: bool
    risk_threshold: float


@app.get("/health")
def health():
    """Liveness probe + which model version is serving."""
    return {"status": "ok", "model": _meta["best_algorithm"],
            "registered_model": _meta["registered_model"]}


@app.post("/predict", response_model=PredictionOut)
def predict(student: Student):
    """Score a single student and flag academic risk."""
    row = pd.DataFrame([student.model_dump()])[config.FEATURES]
    score = float(_model.predict(row)[0])
    return PredictionOut(predicted_score=round(score, 2),
                         at_risk=score < config.RISK_THRESHOLD,
                         risk_threshold=config.RISK_THRESHOLD)


@app.post("/predict_batch")
def predict_batch(students: List[Student]):
    """Score many students at once (used for batch monitoring)."""
    rows = pd.DataFrame([s.model_dump() for s in students])[config.FEATURES]
    scores = _model.predict(rows)
    return [{"predicted_score": round(float(s), 2),
             "at_risk": bool(s < config.RISK_THRESHOLD)} for s in scores]
