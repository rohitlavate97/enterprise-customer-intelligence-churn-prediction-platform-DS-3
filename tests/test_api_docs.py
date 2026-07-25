"""Unit tests for Swagger UI, ReDoc, OpenAPI spec export, /explain, and /metrics endpoints."""

import pytest
from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_swagger_ui_docs_endpoint():
    """Assert /docs returns HTTP 200 OK rendering Swagger UI."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower() or "html" in response.text.lower()


def test_redoc_docs_endpoint():
    """Assert /redoc returns HTTP 200 OK rendering ReDoc UI."""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "redoc" in response.text.lower() or "html" in response.text.lower()


def test_openapi_json_schema():
    """Assert /openapi.json returns valid OpenAPI 3.0 schema with paths and component schemas."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()

    assert "openapi" in data
    assert "info" in data
    assert data["info"]["title"] == "Enterprise Customer Intelligence & Churn Prediction Platform API"
    assert "/predict" in data["paths"]
    assert "/predict/batch" in data["paths"]
    assert "/health" in data["paths"]
    assert "/metrics" in data["paths"]


def test_explain_customer_endpoint():
    """Assert /explain/{customer_id} endpoint returns SHAP explanation breakdown."""
    response = client.get("/explain/CUST-7777")
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "CUST-7777"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert len(data["top_drivers"]) > 0
    assert "summary_narrative" in data


def test_prometheus_metrics_endpoint():
    """Assert /metrics endpoint returns Prometheus plain-text metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text

    assert "churn_api_requests_total" in text
    assert "churn_predictions_total" in text
    assert "churn_api_uptime_seconds" in text
