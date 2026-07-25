"""Unit tests for FastAPI production endpoints and latency SLAs."""

import time
import pytest
from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check_endpoint():
    """Assert /health endpoint returns 200 OK and system uptime metrics."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "loaded_model" in data
    assert data["uptime_seconds"] >= 0.0


def test_predict_single_endpoint():
    """Assert /predict endpoint processes customer payload and returns probability & business drivers."""
    payload = {
        "customer_id": "CUST-9999",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure_months": 3,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract_type": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "plan_tier": "Standard",
        "geography": "North America",
        "support_tickets_30d": 4,
        "resolution_satisfaction_score": 1.5,
        "app_logins_30d": 3,
        "payment_failures_90d": 2,
        "change_in_usage_pct": -0.40,
        "competitor_offer_viewed": "Yes",
        "price_increase_applied_30d": "Yes",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "CUST-9999"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]
    assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert len(data["top_business_reasons"]) > 0
    assert len(data["recommended_retention_action"]) > 0


def test_predict_batch_endpoint():
    """Assert /predict/batch endpoint handles multiple customer payloads."""
    payload = {
        "customers": [
            {
                "customer_id": "CUST-0001",
                "tenure_months": 24,
                "contract_type": "Two year",
                "monthly_charges": 45.0,
                "total_charges": 1080.0,
                "support_tickets_30d": 0,
            },
            {
                "customer_id": "CUST-0002",
                "tenure_months": 2,
                "contract_type": "Month-to-month",
                "monthly_charges": 110.0,
                "total_charges": 220.0,
                "support_tickets_30d": 5,
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2


def test_predict_validation_error():
    """Assert invalid input (e.g. negative monthly charges) returns HTTP 422 Unprocessable Entity."""
    invalid_payload = {
        "customer_id": "CUST-BAD",
        "monthly_charges": -50.0,  # Fails ge=0.0 constraint
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_latency_performance_guarantee():
    """Assert single customer inference latency is under 50ms (p99 SLA)."""
    payload = {
        "customer_id": "CUST-PERF",
        "tenure_months": 12,
        "monthly_charges": 75.0,
        "contract_type": "Month-to-month",
    }

    start = time.perf_counter()
    response = client.post("/predict", json=payload)
    latency_ms = (time.perf_counter() - start) * 1000.0

    assert response.status_code == 200
    assert latency_ms < 50.0  # SLA < 50ms
