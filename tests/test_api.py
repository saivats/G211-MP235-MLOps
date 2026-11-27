import json
import subprocess
import sys
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


def test_prediction_writes_structured_json_log():
    # Capture the process streams from startup, including the logger's bound stdout.
    program = """
from unittest.mock import Mock
from fastapi.testclient import TestClient
import app.main as main
model = Mock()
model.predict.return_value = [42.5]
main.load_model = lambda: model
response = TestClient(main.app).post('/predict', json=PAYLOAD)
assert response.status_code == 200
""".replace("PAYLOAD", repr(valid_payload()))
    result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                            text=True, check=True)
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "prediction"
    assert record["spoilage_pct"] == 42.5
