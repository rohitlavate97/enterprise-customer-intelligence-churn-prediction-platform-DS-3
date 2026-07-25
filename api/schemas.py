"""Pydantic schemas for FastAPI production request and response payloads."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class CustomerPredictionRequest(BaseModel):
    """Input customer profile payload with strict validation bounds."""

    customer_id: str = Field(default="CUST-0001", description="Unique customer ID")
    gender: Literal["Male", "Female"] = Field(default="Female")
    senior_citizen: int = Field(default=0, ge=0, le=1)
    partner: Literal["Yes", "No"] = Field(default="No")
    dependents: Literal["Yes", "No"] = Field(default="No")
    tenure_months: int = Field(default=12, ge=0, le=120)
    phone_service: Literal["Yes", "No"] = Field(default="Yes")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(default="No")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(default="Fiber optic")
    online_security: Literal["Yes", "No", "No internet service"] = Field(default="No")
    online_backup: Literal["Yes", "No", "No internet service"] = Field(default="Yes")
    device_protection: Literal["Yes", "No", "No internet service"] = Field(default="No")
    tech_support: Literal["Yes", "No", "No internet service"] = Field(default="No")
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(default="Yes")
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(default="Yes")
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(default="Month-to-month")
    paperless_billing: Literal["Yes", "No"] = Field(default="Yes")
    payment_method: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(default="Electronic check")
    monthly_charges: float = Field(default=85.5, ge=0.0, le=1000.0)
    total_charges: float = Field(default=1026.0, ge=0.0, le=100000.0)
    plan_tier: Literal["Basic", "Standard", "Premium", "Enterprise"] = Field(default="Standard")
    geography: Literal["North America", "Europe", "Asia-Pacific", "Latin America"] = Field(default="North America")
    support_tickets_30d: int = Field(default=2, ge=0, le=50)
    resolution_satisfaction_score: float = Field(default=2.5, ge=1.0, le=5.0)
    app_logins_30d: int = Field(default=8, ge=0, le=500)
    payment_failures_90d: int = Field(default=1, ge=0, le=20)
    change_in_usage_pct: float = Field(default=-0.15, ge=-1.0, le=5.0)
    competitor_offer_viewed: Literal["Yes", "No"] = Field(default="Yes")
    price_increase_applied_30d: Literal["Yes", "No"] = Field(default="Yes")


class LocalExplanationItem(BaseModel):
    feature: str
    value: Any
    impact_score: float
    business_reason: str


class PredictionResponse(BaseModel):
    """Output prediction response with risk level, probability, and business explanations."""

    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_level: Literal["Low Risk", "Medium Risk", "High Risk"]
    estimated_clv: float
    top_business_reasons: list[str]
    recommended_retention_action: str
    model_version: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerPredictionRequest]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    high_risk_count: int
    total_revenue_at_risk: float
    predictions: list[PredictionResponse]


class HealthCheckResponse(BaseModel):
    status: str
    loaded_model: str
    pipeline_status: str
    uptime_seconds: float
