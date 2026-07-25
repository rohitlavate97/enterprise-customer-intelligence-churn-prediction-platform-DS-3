"""Pydantic schemas for FastAPI request/response payloads with full Swagger UI documentation."""

from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field


class CustomerPredictionRequest(BaseModel):
    """Input customer profile with strict validation bounds and Swagger UI example."""

    customer_id: str = Field(
        default="CUST-0001",
        description="Unique Customer Account Identifier",
        examples=["CUST-84920"],
    )
    gender: Literal["Male", "Female"] = Field(
        default="Female", description="Customer Gender", examples=["Female"]
    )
    senior_citizen: int = Field(
        default=0, ge=0, le=1, description="Senior Citizen Flag (0 = No, 1 = Yes)", examples=[0]
    )
    partner: Literal["Yes", "No"] = Field(
        default="No", description="Has Partner", examples=["No"]
    )
    dependents: Literal["Yes", "No"] = Field(
        default="No", description="Has Dependents", examples=["No"]
    )
    tenure_months: int = Field(
        default=12, ge=0, le=120, description="Account Tenure in Months", examples=[3]
    )
    phone_service: Literal["Yes", "No"] = Field(
        default="Yes", description="Phone Service Subscribed", examples=["Yes"]
    )
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        default="No", description="Multiple Phone Lines", examples=["No"]
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        default="Fiber optic", description="Internet Service Type", examples=["Fiber optic"]
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Online Security Add-on", examples=["No"]
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        default="Yes", description="Online Backup Add-on", examples=["No"]
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Device Protection Plan", examples=["No"]
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        default="No", description="Tech Support Add-on", examples=["No"]
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        default="Yes", description="Streaming TV Subscribed", examples=["Yes"]
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        default="Yes", description="Streaming Movies Subscribed", examples=["Yes"]
    )
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(
        default="Month-to-month", description="Contract Term Structure", examples=["Month-to-month"]
    )
    paperless_billing: Literal["Yes", "No"] = Field(
        default="Yes", description="Paperless Billing Opt-in", examples=["Yes"]
    )
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        default="Electronic check",
        description="Billing Payment Method",
        examples=["Electronic check"],
    )
    monthly_charges: float = Field(
        default=85.5,
        ge=0.0,
        le=1000.0,
        description="Monthly Recurring Bill (USD)",
        examples=[95.50],
    )
    total_charges: float = Field(
        default=1026.0,
        ge=0.0,
        le=100000.0,
        description="Total Historical Billing (USD)",
        examples=[286.50],
    )
    plan_tier: Literal["Basic", "Standard", "Premium", "Enterprise"] = Field(
        default="Standard", description="Subscription Plan Tier", examples=["Standard"]
    )
    geography: Literal["North America", "Europe", "Asia-Pacific", "Latin America"] = Field(
        default="North America", description="Customer Region", examples=["North America"]
    )
    support_tickets_30d: int = Field(
        default=2,
        ge=0,
        le=50,
        description="Support Tickets Opened in Last 30 Days",
        examples=[4],
    )
    resolution_satisfaction_score: float = Field(
        default=2.5,
        ge=1.0,
        le=5.0,
        description="Support CSAT Rating (1.0 = Very Poor, 5.0 = Excellent)",
        examples=[1.5],
    )
    app_logins_30d: int = Field(
        default=8,
        ge=0,
        le=500,
        description="Mobile App Logins in Last 30 Days",
        examples=[3],
    )
    payment_failures_90d: int = Field(
        default=1,
        ge=0,
        le=20,
        description="Payment Failure Events in Last 90 Days",
        examples=[2],
    )
    change_in_usage_pct: float = Field(
        default=-0.15,
        ge=-1.0,
        le=5.0,
        description="Percentage Change in Usage (e.g. -0.40 = 40% decline)",
        examples=[-0.40],
    )
    competitor_offer_viewed: Literal["Yes", "No"] = Field(
        default="Yes",
        description="Customer Viewed Competitor Retention Campaign",
        examples=["Yes"],
    )
    price_increase_applied_30d: Literal["Yes", "No"] = Field(
        default="Yes",
        description="Price Increase Applied in Last 30 Days",
        examples=["Yes"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST-84920",
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
                "monthly_charges": 95.50,
                "total_charges": 286.50,
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
        }
    }


class LocalExplanationItem(BaseModel):
    """A single SHAP feature contribution driver."""

    feature: str = Field(description="Feature name as used in the ML pipeline")
    value: Any = Field(description="Observed raw feature value for this customer")
    impact_score: float = Field(
        description="SHAP / feature-importance impact magnitude (positive = increases churn risk)"
    )
    business_reason: str = Field(
        description="Plain-language business explanation for retention team use"
    )


class PredictionResponse(BaseModel):
    """Churn prediction response with risk tier, SHAP explanations, and retention recommendation."""

    customer_id: str = Field(description="Customer Account ID", examples=["CUST-84920"])
    churn_probability: float = Field(
        description="Model-predicted churn probability (0.0 = Retained, 1.0 = Churned)",
        examples=[0.8924],
    )
    churn_prediction: int = Field(
        description="Binary classification label (1 = Likely Churn, 0 = Likely Retained)",
        examples=[1],
    )
    risk_level: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(
        description="Risk severity tier: Low (<40%), Medium (40–70%), High (>70%)",
        examples=["High Risk"],
    )
    estimated_clv: float = Field(
        description="Estimated Customer Lifetime Value in USD", examples=[2362.50]
    )
    top_business_reasons: list[str] = Field(
        description="Top SHAP-driven plain-language churn risk drivers",
        examples=[
            [
                "Frequent support tickets opened in last 30 days (+24.0% risk)",
                "Flexible Month-to-month contract structure (+18.0% risk)",
                "Recent monthly price increase applied (+12.0% risk)",
            ]
        ],
    )
    recommended_retention_action: str = Field(
        description="Actionable retention recommendation tailored to risk tier",
        examples=["Immediate priority call by VIP retention specialist + 12m contract lock offer."],
    )
    model_version: str = Field(
        description="Model artifact name / registry version tag",
        examples=["catboost_model.joblib"],
    )
    latency_ms: float = Field(
        description="Server-side inference latency in milliseconds", examples=[4.12]
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload. Maximum 1 000 customers per call."""

    customers: Annotated[
        list[CustomerPredictionRequest],
        Field(
            description="Array of customer profile objects to score. Maximum 1 000 per call.",
            max_length=1000,
            min_length=1,
        ),
    ]


class BatchPredictionResponse(BaseModel):
    """Batch prediction result with aggregate revenue-at-risk summary."""

    total_processed: int = Field(
        description="Total customer records processed", examples=[2]
    )
    high_risk_count: int = Field(
        description="Count of customers classified as High Risk (≥70% churn probability)",
        examples=[1],
    )
    total_revenue_at_risk: float = Field(
        description="Aggregate CLV at risk across all predicted churners (USD)", examples=[2362.50]
    )
    predictions: list[PredictionResponse] = Field(
        description="Individual prediction results for each input customer"
    )


class ExplanationResponse(BaseModel):
    """SHAP explanation response with feature breakdown and narrative summary."""

    customer_id: str = Field(description="Customer Account ID", examples=["CUST-84920"])
    churn_probability: float = Field(
        description="Predicted churn probability (0.0–1.0)", examples=[0.8924]
    )
    risk_level: str = Field(description="Risk severity tier", examples=["High Risk"])
    recommended_retention_action: str = Field(
        description="Actionable retention recommendation for this customer",
        examples=["Immediate priority call by VIP retention specialist + 12m contract lock offer."],
    )
    summary_narrative: str = Field(
        description="Plain-language SHAP summary narrative for retention team",
        examples=[
            "Primary churn drivers: Frequent support tickets in 30d; "
            "Month-to-month contract; Recent price increase."
        ],
    )
    top_drivers: list[LocalExplanationItem] = Field(
        description="Detailed per-feature SHAP contribution breakdown"
    )


class HealthCheckResponse(BaseModel):
    """API health and liveness status."""

    status: str = Field(description="API operational status", examples=["Healthy"])
    loaded_model: str = Field(
        description="Currently loaded ML model artifact name",
        examples=["catboost_model.joblib"],
    )
    pipeline_status: str = Field(
        description="Feature preprocessing pipeline status", examples=["Active"]
    )
    uptime_seconds: float = Field(
        description="Server uptime in seconds since last startup", examples=[142.85]
    )


class VersionResponse(BaseModel):
    """Platform and API version metadata."""

    platform_version: str = Field(
        description="Full semantic version of the platform", examples=["2.1.0"]
    )
    api_version: str = Field(description="API major version string", examples=["v1"])
    model_artifact: str = Field(
        description="Currently active model artifact name", examples=["catboost_model.joblib"]
    )
    build_date: str = Field(description="ISO 8601 build/release date", examples=["2026-07-25"])
