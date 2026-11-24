
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

# Create the FastAPI application
app = FastAPI(title="MP235 Cold Storage Spoilage API")


# Define the eight input fields required for prediction
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


# Health-check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
