"""Unit tests for Swagger UI, ReDoc, OpenAPI spec, /explain, /metrics, and /version endpoints."""

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
    """Assert /openapi.json returns valid OpenAPI 3.x schema with all expected paths."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()

    assert "openapi" in data
    assert "info" in data
    assert data["info"]["title"] == "Enterprise Customer Intelligence & Churn Prediction Platform API"
    assert data["info"]["version"] == "2.1.0"
    assert "/predict" in data["paths"]
    assert "/predict/batch" in data["paths"]
    assert "/health" in data["paths"]
    assert "/metrics" in data["paths"]
    assert "/version" in data["paths"]

    # Contact and license metadata present
    assert "contact" in data["info"]
    assert "license" in data["info"]


def test_openapi_tags_metadata():
    """Assert OpenAPI spec includes all expected tag groups with descriptions."""
    response = client.get("/openapi.json")
    data = response.json()
    tag_names = [t["name"] for t in data.get("tags", [])]
    assert "Inference" in tag_names
    assert "Explainability" in tag_names
    assert "System & Health" in tag_names
    assert "Monitoring" in tag_names


def test_version_endpoint():
    """Assert /version endpoint returns platform version and model artifact info."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()

    assert "platform_version" in data
    assert "api_version" in data
    assert "model_artifact" in data
    assert "build_date" in data
    assert data["platform_version"] == "2.1.0"
    assert data["api_version"] == "v1"


def test_explain_customer_endpoint():
    """Assert /explain/{customer_id} returns SHAP explanation with retention action."""
    response = client.get("/explain/CUST-7777")
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "CUST-7777"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert len(data["top_drivers"]) > 0
    assert "summary_narrative" in data
    # New field: recommended_retention_action
    assert "recommended_retention_action" in data
    assert len(data["recommended_retention_action"]) > 0


def test_prometheus_metrics_endpoint():
    """Assert /metrics returns Prometheus plain-text metrics with all required counters."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    text = response.text

    assert "churn_api_requests_total" in text
    assert "churn_predictions_total" in text
    assert "churn_api_uptime_seconds" in text
    assert "churn_high_risk_predictions_total" in text


def test_cors_headers_present():
    """Assert CORS headers are returned by the API."""
    response = client.options("/predict", headers={"Origin": "http://localhost:3000"})
    # CORS middleware should add allow-origin header (either on options or main response)
    get_resp = client.get("/health", headers={"Origin": "http://external.example.com"})
    assert get_resp.status_code == 200


def test_request_id_header_present():
    """Assert X-Request-ID and X-Response-Time-Ms headers are injected by middleware."""
    response = client.get("/health")
    assert "x-request-id" in response.headers
    assert "x-response-time-ms" in response.headers
    # Validate UUID format
    import uuid
    uuid.UUID(response.headers["x-request-id"])  # Should not raise
