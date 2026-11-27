import json
import logging
import app.main as main
from fastapi.testclient import TestClient

client = TestClient(main.app)


class FakeModel:
    """A temporary model used only for API testing."""

    def __init__(self):
        self.received_columns = None
        self.received_data = None

    def predict(self, data):
        self.received_columns = list(data.columns)
        self.received_data = data
        return [42.5]


def valid_payload():
    return {
        "storage_days": 100,
        "temp_set_c": 4.0,
        "humidity_pct": 80.0,
        "power_cuts_hours": 2.0,
        "bag_weight_kg": 25.0,
        "door_openings_per_day": 5,
        "crop": "Potato",
        "chamber": "A",
    }


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(main, "load_model", lambda: fake_model)

    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    assert response.json()["spoilage_pct"] == 42.5

    assert fake_model.received_columns == main.FEATURE_ORDER
    assert len(fake_model.received_data) == 1


def test_storage_days_above_limit_returns_422():
    payload = valid_payload()
    payload["storage_days"] = 301

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_bag_weight_below_limit_returns_422():
    payload = valid_payload()
    payload["bag_weight_kg"] = 19

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_extra_field_returns_422():
    payload = valid_payload()
    payload["record_id"] = "TEST001"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_accepts_boundary_values(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(main, "load_model", lambda: fake_model)

    payload = valid_payload()
    payload["storage_days"] = 300
    payload["bag_weight_kg"] = 20

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["spoilage_pct"] == 42.5


def test_prediction_writes_structured_json_log(monkeypatch, caplog):
    fake_model = FakeModel()
    monkeypatch.setattr(main, "load_model", lambda: fake_model)

    with caplog.at_level(logging.INFO, logger="mp235.api"):
        response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    log_records = [
        record for record in caplog.records
        if record.name == "mp235.api"
    ]

    assert len(log_records) == 1

    log_data = json.loads(log_records[0].message)

    assert log_data["event"] == "prediction"
    assert log_data["spoilage_pct"] == 42.5
    assert log_data["storage_days"] == 100
    assert log_data["crop"] == "Potato"
    assert "timestamp" in log_data
