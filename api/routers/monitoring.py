"""Monitoring router — Prometheus-compatible metrics endpoint."""

from fastapi import APIRouter
from fastapi.responses import Response

from utils.logger import get_logger

logger = get_logger("api.routers.monitoring")

router = APIRouter()


@router.get(
    "/metrics",
    response_class=Response,
    summary="Prometheus Metrics",
    description=(
        "Returns Prometheus-compatible plain-text exposition format metrics for scraping by "
        "Prometheus, Grafana, or any OpenMetrics-compatible monitoring stack. "
        "Includes request counters, prediction counters, high-risk counts, and uptime gauge."
    ),
    response_description="Plain-text Prometheus metrics exposition format.",
    tags=["Monitoring"],
)
def prometheus_metrics() -> Response:
    """Prometheus-compatible plain-text metrics endpoint for operational observability."""
    from api.routers.inference import get_counters
    from api.routers.system import _SERVER_START_TIME
    import time

    uptime = time.time() - _SERVER_START_TIME
    counters = get_counters()

    metrics_text = f"""# HELP churn_api_requests_total Total number of API requests received.
# TYPE churn_api_requests_total counter
churn_api_requests_total {counters.get("requests_total", 0)}

# HELP churn_predictions_total Total customer churn predictions served.
# TYPE churn_predictions_total counter
churn_predictions_total {counters.get("predictions_total", 0)}

# HELP churn_high_risk_predictions_total Count of high-risk churn predictions.
# TYPE churn_high_risk_predictions_total counter
churn_high_risk_predictions_total {counters.get("high_risk_total", 0)}

# HELP churn_api_uptime_seconds API server uptime in seconds.
# TYPE churn_api_uptime_seconds gauge
churn_api_uptime_seconds {uptime:.2f}
"""
    return Response(content=metrics_text, media_type="text/plain")
