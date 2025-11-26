import pytest
from fastapi.testclient import TestClient
from src.serving.api import app


client = TestClient(app)


class _DummyModel:
    def predict(self, X):
        return ["BAIK"] * len(X)

    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in range(len(X))]


@pytest.fixture(autouse=True)
def ensure_model_loaded():
    from src.serving import api as serving_api

    serving_api.model_artifacts["model"] = _DummyModel()
    serving_api.model_artifacts["feature_engineering"] = None
    yield
    serving_api.model_artifacts.clear()


def test_root_endpoint():
    """Test root endpoint returns correct information."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "message" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_check_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    assert "message" in data

    assert data["status"] in ["healthy", "degraded"]
    assert isinstance(data["model_loaded"], bool)


def test_predict_endpoint_with_valid_payload(api_test_payload):
    """Test prediction endpoint with valid payload."""
    response = client.post("/predict", json=api_test_payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], str)


def test_predict_endpoint_with_invalid_station():
    """Test prediction endpoint rejects invalid station."""
    payload = {
        "stasiun": "Invalid Station",
        "pm10": 50.5,
        "pm25": 30.2,
        "so2": 15.0,
        "co": 5.5,
        "o3": 45.0,
        "no2": 25.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_with_out_of_range_values():
    """Test prediction endpoint rejects out-of-range values."""
    payload = {
        "stasiun": "DKI1 (Bunderan HI)",
        "pm10": 1000.0,
        "pm25": 30.2,
        "so2": 15.0,
        "co": 5.5,
        "o3": 45.0,
        "no2": 25.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_with_missing_fields():
    """Test prediction endpoint rejects missing required fields."""
    payload = {"stasiun": "DKI1 (Bunderan HI)", "pm10": 50.5}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_with_wrong_data_types():
    """Test prediction endpoint rejects wrong data types."""
    payload = {
        "stasiun": "DKI1 (Bunderan HI)",
        "pm10": "invalid",
        "pm25": 30.2,
        "so2": 15.0,
        "co": 5.5,
        "o3": 45.0,
        "no2": 25.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_model_info_endpoint():
    """Test model info endpoint."""
    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()

    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)

    if data["model_loaded"]:
        assert "model_type" in data
        assert "features" in data


def test_predict_endpoint_boundary_values():
    """Test prediction endpoint with boundary values."""
    payload = {
        "stasiun": "DKI1 (Bunderan HI)",
        "pm10": -1.0,
        "pm25": 400.0,
        "so2": 500.0,
        "co": 100.0,
        "o3": 160.0,
        "no2": 100.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_api_docs_available():
    """Test that API documentation is available."""
    response = client.get("/docs")

    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is available."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    assert "info" in schema
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
