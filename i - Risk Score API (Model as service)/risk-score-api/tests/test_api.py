"""Basic API tests using FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_predict_endpoint():
    """Test /predict endpoint with valid JSON payload."""
    payload = {
        "alert_type": "suspicious_login",
        "source_ip_risk": 0.75,
        "user_risk_score": 0.65,
        "failed_login_count_24h": 12,
        "geo_impossible_travel": 1,
        "asset_criticality": "high",
        "historical_false_positive_rate": 0.15
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "label" in data
    assert data["label"] in ["low", "medium", "high"]
    
    assert "confidence" in data
    assert isinstance(data["confidence"], (int, float))
    assert 0 <= data["confidence"] <= 1
    
    assert "probabilities" in data
    assert isinstance(data["probabilities"], dict)
    
    assert "explanation" in data
    assert isinstance(data["explanation"], str)
    assert len(data["explanation"]) > 0


def test_model_info_endpoint():
    """Test /model-info endpoint."""
    response = client.get("/model-info")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required keys
    assert "features" in data
    assert "target" in data
    assert "classes" in data
    assert "training_samples" in data
    assert "validation_samples" in data
    
    # Check features structure
    assert isinstance(data["features"], dict)
    assert "numeric" in data["features"]
    assert "categorical" in data["features"]
    
    # Check target
    assert data["target"] == "priority"
    
    # Check classes
    assert isinstance(data["classes"], list)
    assert len(data["classes"]) > 0
    assert all(cls in ["low", "medium", "high"] for cls in data["classes"])
    
    # Check sample counts
    assert isinstance(data["training_samples"], int)
    assert data["training_samples"] > 0
    assert isinstance(data["validation_samples"], int)
    assert data["validation_samples"] > 0


def test_metrics_endpoint():
    """Test /metrics endpoint."""
    response = client.get("/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check required keys
    assert "classification_report" in data
    assert "confusion_matrix" in data
    
    # Check classification_report structure
    assert isinstance(data["classification_report"], dict)
    
    # Check confusion_matrix structure
    assert isinstance(data["confusion_matrix"], list)
    assert len(data["confusion_matrix"]) > 0
    assert all(isinstance(row, list) for row in data["confusion_matrix"])


def test_index_endpoint():
    """Test / endpoint returns HTML."""
    response = client.get("/")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"Alert Risk Scoring Demo" in response.content


def test_predict_form_endpoint():
    """Test /predict-form endpoint with form data."""
    form_data = {
        "alert_type": "malware",
        "source_ip_risk": 0.5,
        "user_risk_score": 0.4,
        "failed_login_count_24h": 5,
        "geo_impossible_travel": 0,
        "asset_criticality": "medium",
        "historical_false_positive_rate": 0.3
    }
    
    response = client.post("/predict-form", data=form_data)
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # Check that HTML contains priority information
    content = response.text
    assert "Priority" in content or "priority" in content.lower()


def test_predict_validation_error():
    """Test /predict endpoint with invalid payload."""
    invalid_payload = {
        "alert_type": "invalid_type",  # Invalid value
        "source_ip_risk": 2.0,  # Out of range
        "user_risk_score": 0.5,
        "failed_login_count_24h": -1,  # Negative
        "geo_impossible_travel": 1,
        "asset_criticality": "high",
        "historical_false_positive_rate": 0.2
    }
    
    response = client.post("/predict", json=invalid_payload)
    
    # Should return validation error (422)
    assert response.status_code == 422
