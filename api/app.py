"""FastAPI Production Inference Server.

Orchestrates all API routers, CORS, request-ID middleware, and
enriched OpenAPI 3.1 documentation (Swagger UI + ReDoc).
"""

import time
import uuid
import joblib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import JSONResponse

from api.routers import inference as inference_router
from api.routers import monitoring as monitoring_router
from api.routers import system as system_router
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("api.app")

# ---------------------------------------------------------------------------
# Global ML artifact cache (loaded once at startup)
# ---------------------------------------------------------------------------
ml_artifacts: dict[str, Any] = {}


def load_model_and_pipeline() -> None:
    """Load fitted preprocessor and champion model from disk artifacts."""
    pipe_path = settings.artifacts_dir / "preprocessing_pipeline.joblib"

    # Resolution order: champion registry artifact → CatBoost → XGBoost fallback
    for candidate in (
        settings.artifacts_dir / "champion_model.joblib",
        settings.artifacts_dir / "catboost_model.joblib",
        settings.artifacts_dir / "xgboost_model.joblib",
    ):
        if candidate.exists():
            model_path = candidate
            break
    else:
        model_path = None

    if pipe_path.exists() and model_path is not None:
        ml_artifacts["pipeline"] = joblib.load(pipe_path)
        ml_artifacts["model"] = joblib.load(model_path)
        ml_artifacts["model_name"] = model_path.name
        logger.info(f"Loaded model '{model_path.name}' and preprocessing pipeline.")
    else:
        ml_artifacts["pipeline"] = None
        ml_artifacts["model"] = None
        ml_artifacts["model_name"] = "FallbackHeuristicModel"
        logger.warning("Artifacts missing. API operating in Fallback Heuristic Mode.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup model loading."""
    load_model_and_pipeline()
    yield
    ml_artifacts.clear()


# ---------------------------------------------------------------------------
# OpenAPI metadata
# ---------------------------------------------------------------------------
_TAGS_METADATA = [
    {
        "name": "Inference",
        "description": (
            "Real-time **single** and **batch** customer churn probability scoring endpoints. "
            "Predictions include SHAP-driven business explanations, CLV estimates, and "
            "actionable retention recommendations — all under the **50 ms SLA**."
        ),
        "externalDocs": {
            "description": "Model comparison report",
            "url": "https://github.com/rohitlavate97/enterprise-customer-intelligence-churn-prediction-platform-DS-3/blob/main/docs/MODEL_COMPARISON_REPORT.md",
        },
    },
    {
        "name": "Explainability",
        "description": (
            "Per-customer SHAP feature-contribution breakdown and plain-language "
            "retention narratives designed for CRM integration and retention-specialist tooling."
        ),
    },
    {
        "name": "System & Health",
        "description": (
            "Kubernetes liveness probes, platform version info, server uptime, "
            "and raw OpenAPI JSON schema export for client-SDK generation."
        ),
    },
    {
        "name": "Monitoring",
        "description": (
            "Prometheus-compatible `/metrics` endpoint for real-time operational "
            "observability. Compatible with Prometheus, Grafana, and OpenMetrics scrapers."
        ),
    },
]

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Customer Intelligence & Churn Prediction Platform API",
    version="2.1.0",
    description="""
# Enterprise Customer Intelligence & Churn Prediction Platform API

Welcome to the official REST API for the **Enterprise Customer Intelligence & Churn Prediction Platform**.

## Key Features
- 🚀 **Real-Time Inference (<50 ms SLA):** Single-row probability scoring powered by CatBoost/LightGBM/XGBoost.
- 💡 **Explainable AI (SHAP):** Every prediction returns plain-language, actionable drivers for retention teams.
- 💰 **Financial Risk Modeling:** Embedded CLV & revenue-at-risk calculations.
- 🛡️ **Leakage Guard:** Hard-stop guards ensuring zero target-leakage field contamination.
- 📈 **Prometheus Monitoring:** Standard `/metrics` endpoint for operational metric scraping.
- 🔄 **Automated Retraining:** Champion vs. Challenger gate with 1% relative PR-AUC margin and rollback.

## Authentication
Currently open for internal use. Bearer-token authentication (API key) will be enforced in v3.0.

## Rate Limits
- `/predict`: 1 000 req/min per client
- `/predict/batch`: 100 req/min per client (max 1 000 customers per request)

## SLA
- p99 single-prediction latency: **< 50 ms**
- Availability target: **99.9%**

---
""",
    openapi_tags=_TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Enterprise ML Architecture Team",
        "url": "https://github.com/rohitlavate97/enterprise-customer-intelligence-churn-prediction-platform-DS-3",
        "email": "ml-platform@enterprise.internal",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    terms_of_service="https://enterprise.internal/tos",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for internal dev; tighten for production
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request-ID + timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_timing_middleware(request: Request, call_next) -> Response:
    """Attach a unique X-Request-ID header and log request timing."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"

    logger.debug(
        f"[{request_id}] {request.method} {request.url.path} → "
        f"{response.status_code} ({latency_ms:.2f} ms)"
    )
    return response

# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------
app.include_router(inference_router.router)
app.include_router(system_router.router)
app.include_router(monitoring_router.router)
