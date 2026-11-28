import json
import logging
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class APIPredictionFormatter(logging.Formatter):
    """Write the API's prediction fields as one JSON object per line."""

    def format(self, record):
        return json.dumps(record.prediction_fields, separators=(",", ":"), allow_nan=False)


app = FastAPI(title="MP235 Cold Storage Spoilage API")
logger = logging.getLogger("mp235.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(APIPredictionFormatter())
    logger.addHandler(handler)


# Exact feature order supplied by the ML lead
FEATURE_ORDER = [
    "storage_days",
    "temp_set_c",
    "humidity_pct",
    "power_cuts_hours",
    "bag_weight_kg",
    "door_openings_per_day",
    "crop",
    "chamber",
]

# Locate the champion model relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"


# Eight-field API request schema
class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_days: int = Field(..., le=300)
    temp_set_c: float
    humidity_pct: float
    power_cuts_hours: float
    bag_weight_kg: float = Field(..., ge=20)
    door_openings_per_day: int
    crop: str
    chamber: str


class PredictionResponse(BaseModel):
    spoilage_pct: float


# Load the model only when a prediction is requested
@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Champion model is not available yet."
        )

    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Champion model could not be loaded."
        ) from exc


@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_data = pd.DataFrame(
        [request.model_dump()],
        columns=FEATURE_ORDER
    )

    model = load_model()

    try:
        prediction = model.predict(input_data)
        spoilage_pct = float(prediction[0])

        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "prediction",
            "storage_days": request.storage_days,
            "temp_set_c": request.temp_set_c,
            "humidity_pct": request.humidity_pct,
            "power_cuts_hours": request.power_cuts_hours,
            "bag_weight_kg": request.bag_weight_kg,
            "door_openings_per_day": request.door_openings_per_day,
            "crop": request.crop,
            "chamber": request.chamber,
            "spoilage_pct": spoilage_pct
        }

        logger.info("prediction", extra={"prediction_fields": log_record})

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed."
        ) from exc

    return PredictionResponse(spoilage_pct=spoilage_pct)

