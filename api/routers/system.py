"""System & Health router — liveness probe, version info, and OpenAPI export."""

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.schemas import HealthCheckResponse, VersionResponse
from utils.logger import get_logger

logger = get_logger("api.routers.system")

router = APIRouter()

_SERVER_START_TIME = time.time()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health & Liveness Check",
    description=(
        "Kubernetes-compatible liveness probe. Returns `200 OK` when the server is "
        "operational with a loaded ML model. Returns `Degraded` status when operating "
        "in heuristic fallback mode (no model artifact on disk)."
    ),
    response_description="API status, loaded model name, pipeline status, and server uptime.",
    tags=["System & Health"],
)
def health_check() -> HealthCheckResponse:
    """Liveness probe returning server uptime and loaded model status."""
    from api.app import ml_artifacts  # late import

    uptime = time.time() - _SERVER_START_TIME
    model_name = ml_artifacts.get("model_name", "None")
    status_str = (
        "Healthy"
        if ml_artifacts.get("model") is not None
        else "Degraded (Fallback Heuristic Mode)"
    )

    return HealthCheckResponse(
        status=status_str,
        loaded_model=model_name,
        pipeline_status="Active" if ml_artifacts.get("pipeline") else "Inactive",
        uptime_seconds=round(uptime, 2),
    )


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Platform Version Info",
    description=(
        "Returns the current platform version, API version, build date, and model artifact name. "
        "Useful for deployment tracking and CI/CD validation."
    ),
    response_description="Platform versioning information.",
    tags=["System & Health"],
)
def get_version() -> VersionResponse:
    """Return platform and API version metadata."""
    from api.app import ml_artifacts  # late import

    return VersionResponse(
        platform_version="2.1.0",
        api_version="v1",
        model_artifact=ml_artifacts.get("model_name", "None"),
        build_date="2026-07-25",
    )


@router.get(
    "/api/v1/openapi.json",
    include_in_schema=False,
    tags=["System & Health"],
)
def export_openapi_json() -> JSONResponse:
    """Export raw OpenAPI JSON schema for client SDK generation."""
    from api.app import app  # late import

    return JSONResponse(content=app.openapi())
