
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app, PredictionRequest

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_input():
    request = PredictionRequest(
        storage_days=300,
        temp_set_c=5.0,
        humidity_pct=80.0,
        power_cuts_hours=2.0,
        bag_weight_kg=20.0,
        door_openings_per_day=5,
        crop="Potato",
        chamber="A"
    )
    assert request.storage_days == 300
    assert request.bag_weight_kg == 20.0


def test_invalid_storage_days():
    with pytest.raises(ValidationError):
        PredictionRequest(
            storage_days=301,
            temp_set_c=5.0,
            humidity_pct=80.0,
            power_cuts_hours=2.0,
            bag_weight_kg=25.0,
            door_openings_per_day=5,
            crop="Potato",
            chamber="A"
        )


def test_invalid_bag_weight():
    with pytest.raises(ValidationError):
        PredictionRequest(
            storage_days=100,
            temp_set_c=5.0,
            humidity_pct=80.0,
            power_cuts_hours=2.0,
            bag_weight_kg=19.0,
            door_openings_per_day=5,
            crop="Potato",
            chamber="A"
        )
