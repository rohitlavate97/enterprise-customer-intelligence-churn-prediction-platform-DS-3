"""FastAPI Production Inference Server for Enterprise Churn Prediction Platform."""

import time
import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerPredictionRequest,
    HealthCheckResponse,
    PredictionResponse,
)
from config.settings import settings
from evaluation.business_roi import BusinessROIAnalyzer
from explainability.shap_explainer import ModelExplainer
from fastapi import FastAPI, HTTPException
from features.builder import FeatureBuilder
from utils.logger import get_logger

logger = get_logger("api.app")

# Global model & pipeline cache
ml_artifacts: dict[str, Any] = {}
SERVER_START_TIME = time.time()


def load_model_and_pipeline() -> None:
    """Load fitted preprocessor and champion model from disk artifacts."""
    pipe_path = settings.artifacts_dir / "preprocessing_pipeline.joblib"
    model_path = settings.artifacts_dir / "catboost_model.joblib"
    if not model_path.exists():
        model_path = settings.artifacts_dir / "xgboost_model.joblib"

    if pipe_path.exists() and model_path.exists():
        ml_artifacts["pipeline"] = joblib.load(pipe_path)
        ml_artifacts["model"] = joblib.load(model_path)
        ml_artifacts["model_name"] = model_path.name
        logger.info(f"FastAPI successfully loaded model '{model_path.name}' and preprocessing pipeline.")
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


app = FastAPI(
    title="Enterprise Customer Intelligence & Churn Prediction Platform API",
    version="1.0.0",
    description="Production-grade REST API serving real-time churn probabilities, SHAP business narratives, and retention action recommendations.",
    lifespan=lifespan,
)


def _predict_single_customer(customer_req: CustomerPredictionRequest) -> PredictionResponse:
    """Core single customer prediction and explanation logic."""
    start_t = time.perf_counter()
    req_dict = customer_req.model_dump()

    # Convert request payload to single-row DataFrame
    df_raw = pd.DataFrame([req_dict])

    # Build domain features
    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)

    if "customer_id" in df_featured.columns:
        df_featured = df_featured.drop(columns=["customer_id"])

    preprocessor = ml_artifacts.get("pipeline")
    model = ml_artifacts.get("model")

    if preprocessor is not None and model is not None:
        X_trans = preprocessor.transform(df_featured)
        prob = float(model.predict_proba(X_trans)[0, 1])
        model_ver = ml_artifacts["model_name"]

        # Feature importances & local explanation
        feature_names = list(preprocessor.named_steps["preprocessor"].get_feature_names_out())
        importances = getattr(model, "feature_importances_", np.ones(X_trans.shape[1]))
        local_exp = ModelExplainer.generate_local_explanation(
            feature_names, X_trans[0], importances, top_k=3
        )
        reasons = [driver["business_reason"] for driver in local_exp["top_drivers"]]
    else:
        # Fallback heuristic mode
        risk_score = 0.1
        if customer_req.contract_type == "Month-to-month":
            risk_score += 0.3
        if customer_req.support_tickets_30d >= 2:
            risk_score += 0.3
        if customer_req.price_increase_applied_30d == "Yes":
            risk_score += 0.2
        prob = min(0.99, risk_score)
        model_ver = "FallbackHeuristicV1"
        reasons = [
            "Month-to-month contract structure (+30% risk)",
            "Frequent support tickets in 30d (+30% risk)",
            "Recent price increase applied (+20% risk)",
        ]

    # Calculate Risk Level & Recommendation
    if prob >= 0.70:
        risk_level = "High Risk"
        rec_action = "Immediate priority call by VIP retention specialist + 12m contract lock offer."
    elif prob >= 0.40:
        risk_level = "Medium Risk"
        rec_action = "Automated email campaign with product feature walkthrough & satisfaction survey."
    else:
        risk_level = "Low Risk"
        rec_action = "No intervention required. Continue standard engagement."

    clv = float(BusinessROIAnalyzer.calculate_clv([customer_req.monthly_charges], [customer_req.tenure_months])[0])
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    return PredictionResponse(
        customer_id=customer_req.customer_id,
        churn_probability=round(prob, 4),
        churn_prediction=1 if prob >= 0.50 else 0,
        risk_level=risk_level,
        estimated_clv=clv,
        top_business_reasons=reasons,
        recommended_retention_action=rec_action,
        model_version=model_ver,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
def health_check() -> HealthCheckResponse:
    """Liveness probe returning server uptime and loaded model status."""
    uptime = time.time() - SERVER_START_TIME
    model_name = ml_artifacts.get("model_name", "None")
    status_str = "Healthy" if ml_artifacts.get("model") is not None else "Degraded (Fallback Heuristic Mode)"

    return HealthCheckResponse(
        status=status_str,
        loaded_model=model_name,
        pipeline_status="Active" if ml_artifacts.get("pipeline") else "Inactive",
        uptime_seconds=round(uptime, 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(customer_req: CustomerPredictionRequest) -> PredictionResponse:
    """Real-time single customer churn prediction with SHAP business explanations."""
    try:
        return _predict_single_customer(customer_req)
    except Exception as e:
        logger.error(f"Prediction failed for customer {customer_req.customer_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch_req: BatchPredictionRequest) -> BatchPredictionResponse:
    """Batch customer prediction processing."""
    predictions = []
    total_rev_at_risk = 0.0
    high_risk_cnt = 0

    for cust_req in batch_req.customers:
        res = _predict_single_customer(cust_req)
        predictions.append(res)
        if res.churn_prediction == 1:
            high_risk_cnt += 1
            total_rev_at_risk += res.estimated_clv

    return BatchPredictionResponse(
        total_processed=len(predictions),
        high_risk_count=high_risk_cnt,
        total_revenue_at_risk=round(total_rev_at_risk, 2),
        predictions=predictions,
    )
