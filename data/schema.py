"""Schema definitions, column classifications, and leakage field registry."""

from typing import Final

# Target Column
TARGET_COL: Final[str] = "churn_label"

# Primary Key / Unique Identifiers
ID_COLS: Final[list[str]] = ["customer_id"]

# Target Leakage Fields — MUST BE EXCLUDED before modeling
LEAKAGE_FIELDS: Final[list[str]] = [
    "cancellation_processed_date",
    "final_invoice_flag",
    "account_status_deactivated",
    "churn_reason_recorded",
]

# Categorical Features
CATEGORICAL_FEATURES: Final[list[str]] = [
    "gender",
    "partner",
    "dependents",
    "geography",
    "contract_type",
    "payment_method",
    "auto_pay_enabled",
    "plan_tier",
    "credit_card_expiring_soon",
    "price_increase_applied_30d",
    "competitor_offer_viewed",
    "retention_offer_accepted",
]

# Numerical Features
NUMERICAL_FEATURES: Final[list[str]] = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_monthly_gb_download",
    "active_days_per_month",
    "login_frequency_30d",
    "app_usage_hours",
    "feature_usage_score",
    "change_in_usage_pct",
    "support_tickets_30d",
    "support_tickets_90d",
    "complaints_opened",
    "resolution_satisfaction_score",
    "chat_interactions",
    "payment_failures_90d",
    "late_payments_12m",
    "nps_score",
    "marketing_emails_opened_30d",
]

# All legitimate predictor features (excluding ID and Leakage)
ALL_PREDICTOR_FEATURES: Final[list[str]] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
