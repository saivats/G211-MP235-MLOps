
# MP235 Cold Storage Spoilage Prediction API

## 1. Overview

This REST API provides an interface for predicting cold-storage
spoilage using environmental and storage-related input features.

The API is built using FastAPI and is packaged using Docker.

## 2. Base URL

Local development:

http://127.0.0.1:8000

## 3. Available Endpoints

### GET /health

Checks whether the API service is running.

Successful response (200):

{
  "status": "ok"
}

### POST /predict

Accepts storage and environmental features and returns a
spoilage prediction.

#### Request body

| Field | Type | Constraint |
|---|---|---|
| storage_days | integer | Maximum 300 |
| temp_set_c | number | — |
| humidity_pct | number | — |
| power_cuts_hours | number | — |
| bag_weight_kg | number | Minimum 20 |
| door_openings_per_day | integer | — |
| crop | string | — |
| chamber | string | — |

#### Example request

{
  "storage_days": 100,
  "temp_set_c": 4.0,
  "humidity_pct": 80.0,
  "power_cuts_hours": 2.0,
  "bag_weight_kg": 25.0,
  "door_openings_per_day": 5,
  "crop": "Potato",
  "chamber": "A"
}

#### Successful response (200)

{
  "spoilage_pct": 42.5
}

The value above is an illustrative response, not a real model result.

#### Error responses

| HTTP status | Meaning |
|---|---|
| 422 | Request validation failed |
| 500 | Prediction failed |
| 503 | Champion model is unavailable or could not be loaded |

## 4. Input Validation

- Additional request fields are rejected.
- storage_days must not exceed 300.
- bag_weight_kg must be at least 20.
- Invalid requests receive HTTP 422.

## 5. Model Integration

The API expects the trained model artifact at:

models/model.joblib

The artifact contains the fitted preprocessing and
RandomForestRegressor pipeline.

The API sends the eight raw input features as a named
pandas DataFrame in the agreed feature order.

Preprocessing is handled inside the fitted pipeline.

Until the verified model artifact is available, the prediction
endpoint returns HTTP 503.

## 6. Structured Logging

Successful predictions generate a JSON log event containing:

- timestamp
- event
- storage_days
- temp_set_c
- humidity_pct
- power_cuts_hours
- bag_weight_kg
- door_openings_per_day
- crop
- chamber
- spoilage_pct

Logs are written through the application logger to standard output.

## 7. Running Locally

Install dependencies:

pip install -r requirements.txt

Start the API:

uvicorn app.main:app --reload

Open API documentation:

http://127.0.0.1:8000/docs

## 8. Running with Docker

Build the image:

docker build -t mp235-api:1.0 .

Run the container:

docker run -d --name mp235-api-container -p 8000:8000 mp235-api:1.0

Check health:

http://127.0.0.1:8000/health
