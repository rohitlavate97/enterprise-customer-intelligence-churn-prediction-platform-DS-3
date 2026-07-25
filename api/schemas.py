"""Pydantic schemas for FastAPI production request and response payloads with Swagger UI examples."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class CustomerPredictionRequest(BaseModel):
    """Input customer profile payload with strict validation bounds and Swagger UI documentation."""

    customer_id: str = Field(default="CUST-0001", description="Unique Customer Account Identifier", examples=["CUST-84920"])
    gender: Literal["Male", "Female"] = Field(default="Female", description="Customer Gender", examples=["Female"])
    senior_citizen: int = Field(default=0, ge=0, le=1, description="Senior Citizen Flag (0 or 1)", examples=[0])
    partner: Literal["Yes", "No"] = Field(default="No", description="Has Partner", examples=["No"])
    dependents: Literal["Yes", "No"] = Field(default="No", description="Has Dependents", examples=["No"])
    tenure_months: int = Field(default=12, ge=0, le=120, description="Account Tenure in Months", examples=[3])
    phone_service: Literal["Yes", "No"] = Field(default="Yes", description="Phone Service Subscribed", examples=["Yes"])
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(default="No", description="Multiple Phone Lines", examples=["No"])
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(default="Fiber optic", description="Internet Service Type", examples=["Fiber optic"])
    online_security: Literal["Yes", "No", "No internet service"] = Field(default="No", description="Online Security Add-on", examples=["No"])
    online_backup: Literal["Yes", "No", "No internet service"] = Field(default="Yes", description="Online Backup Add-on", examples=["No"])
    device_protection: Literal["Yes", "No", "No internet service"] = Field(default="No", description="Device Protection Plan", examples=["No"])
    tech_support: Literal["Yes", "No", "No internet service"] = Field(default="No", description="Tech Support Add-on", examples=["No"])
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(default="Yes", description="Streaming TV Subscribed", examples=["Yes"])
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(default="Yes", description="Streaming Movies Subscribed", examples=["Yes"])
    contract_type: Literal["Month-to-month", "One year", "Two year"] = Field(default="Month-to-month", description="Contract Term Structure", examples=["Month-to-month"])
    paperless_billing: Literal["Yes", "No"] = Field(default="Yes", description="Paperless Billing Opt-in", examples=["Yes"])
    payment_method: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(default="Electronic check", description="Billing Payment Method", examples=["Electronic check"])
    monthly_charges: float = Field(default=85.5, ge=0.0, le=1000.0, description="Monthly Recurring Bill ($)", examples=[95.50])
    total_charges: float = Field(default=1026.0, ge=0.0, le=100000.0, description="Total Historical Billing ($)", examples=[286.50])
    plan_tier: Literal["Basic", "Standard", "Premium", "Enterprise"] = Field(default="Standard", description="Subscription Plan Tier", examples=["Standard"])
    geography: Literal["North America", "Europe", "Asia-Pacific", "Latin America"] = Field(default="North America", description="Customer Region", examples=["North America"])
    support_tickets_30d: int = Field(default=2, ge=0, le=50, description="Support Tickets Opened in Last 30 Days", examples=[4])
    resolution_satisfaction_score: float = Field(default=2.5, ge=1.0, le=5.0, description="Support CSAT Rating (1.0 to 5.0)", examples=[1.5])
    app_logins_30d: int = Field(default=8, ge=0, le=500, description="Mobile App Logins in Last 30 Days", examples=[3])
    payment_failures_90d: int = Field(default=1, ge=0, le=20, description="Payment Failure Events in Last 90 Days", examples=[2])
    change_in_usage_pct: float = Field(default=-0.15, ge=-1.0, le=5.0, description="Percentage Change in Usage (-1.0 to +5.0)", examples=[-0.40])
    competitor_offer_viewed: Literal["Yes", "No"] = Field(default="Yes", description="Viewed Competitor Retention Campaign", examples=["Yes"])
    price_increase_applied_30d: Literal["Yes", "No"] = Field(default="Yes", description="Price Increase Applied in Last 30 Days", examples=["Yes"])

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
    feature: str = Field(description="Feature Name")
    value: Any = Field(description="Observed Feature Value")
    impact_score: float = Field(description="SHAP / Feature Importance Impact Score")
    business_reason: str = Field(description="Plain-Language Business Reason for Retention Teams")


class PredictionResponse(BaseModel):
    """Output prediction response with risk level, probability, and business explanations."""

    customer_id: str = Field(description="Customer Account ID", examples=["CUST-84920"])
    churn_probability: float = Field(description="Model Predicted Churn Probability (0.0 to 1.0)", examples=[0.8924])
    churn_prediction: int = Field(description="Binary Classification Label (1 = Churn, 0 = Retained)", examples=[1])
    risk_level: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(description="Risk Severity Tier", examples=["High Risk"])
    estimated_clv: float = Field(description="Estimated Customer Lifetime Value ($)", examples=[2362.50])
    top_business_reasons: list[str] = Field(description="Top SHAP-driven business explanation drivers", examples=[
        "Frequent support tickets opened in last 30 days (+24.0% risk)",
        "Flexible Month-to-month contract structure (+18.0% risk)",
        "Recent monthly price increase applied (+12.0% risk)"
    ])
    recommended_retention_action: str = Field(description="Actionable Retention Specialist Recommendation", examples=["Immediate priority call by VIP retention specialist + 12m contract lock offer."])
    model_version: str = Field(description="Artifact Name / Version", examples=["catboost_model.joblib"])
    latency_ms: float = Field(description="Server Inference Latency in Milliseconds", examples=[4.12])


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerPredictionRequest] = Field(description="List of customer profile requests")


class BatchPredictionResponse(BaseModel):
    total_processed: int = Field(description="Total Customer Records Processed", examples=[2])
    high_risk_count: int = Field(description="Count of High-Risk Churn Prospects Identified", examples=[1])
    total_revenue_at_risk: float = Field(description="Aggregate Revenue at Risk ($)", examples=[2362.50])
    predictions: list[PredictionResponse] = Field(description="Array of Customer Prediction Results")


class ExplanationResponse(BaseModel):
    customer_id: str = Field(description="Customer ID", examples=["CUST-84920"])
    churn_probability: float = Field(description="Predicted Churn Probability", examples=[0.8924])
    risk_level: str = Field(description="Risk Level", examples=["High Risk"])
    summary_narrative: str = Field(description="Plain-Language SHAP Summary Narrative", examples=["Primary churn drivers: Frequent support tickets in 30d; Month-to-month contract; Recent price increase."])
    top_drivers: list[LocalExplanationItem] = Field(description="Detailed Feature Contribution Breakdown")


class HealthCheckResponse(BaseModel):
    status: str = Field(description="API Operational Status", examples=["Healthy"])
    loaded_model: str = Field(description="Loaded ML Model Artifact", examples=["catboost_model.joblib"])
    pipeline_status: str = Field(description="Feature Preprocessing Pipeline Status", examples=["Active"])
    uptime_seconds: float = Field(description="Server Uptime in Seconds", examples=[142.85])
