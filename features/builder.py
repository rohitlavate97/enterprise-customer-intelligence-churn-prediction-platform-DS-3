"""Domain Feature Engineering module for customer churn prediction."""

import numpy as np
import pandas as pd
from features.leakage_guard import LeakageGuard
from utils.logger import get_logger

logger = get_logger("features.builder")


class FeatureBuilder:
    """Computes domain-specific interaction ratios and engagement indicators cleanly and safely."""

    def __init__(self, enforce_leakage_guard: bool = True) -> None:
        self.enforce_leakage_guard = enforce_leakage_guard

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain feature engineering transformations."""
        if self.enforce_leakage_guard:
            # Strip or assert no leakage fields
            df_work = LeakageGuard.filter_leakage_fields(df.copy())
        else:
            df_work = df.copy()

        logger.info(f"Building domain features on input dataframe of shape {df_work.shape}...")

        # 1. Financial / Charge Ratios
        df_work["charges_per_tenure"] = np.round(
            df_work["total_charges"] / np.maximum(df_work["tenure_months"], 1), 2
        )
        df_work["monthly_to_total_ratio"] = np.round(
            df_work["monthly_charges"] / np.maximum(df_work["total_charges"], 1.0), 4
        )

        # 2. Support & Dissatisfaction Ratios
        df_work["support_tickets_per_tenure"] = np.round(
            df_work["support_tickets_90d"] / np.maximum(df_work["tenure_months"], 1), 3
        )
        df_work["has_high_support_tickets"] = (df_work["support_tickets_30d"] >= 3).astype(int)
        df_work["is_dissatisfied"] = (df_work["resolution_satisfaction_score"] <= 2).astype(int)

        # 3. Usage & Engagement Signals
        df_work["gb_per_active_day"] = np.round(
            df_work["avg_monthly_gb_download"] / np.maximum(df_work["active_days_per_month"], 1), 2
        )
        df_work["severe_usage_drop"] = (df_work["change_in_usage_pct"] < -0.25).astype(int)
        df_work["low_engagement_flag"] = (
            (df_work["active_days_per_month"] <= 5) & (df_work["login_frequency_30d"] <= 3)
        ).astype(int)

        # 4. Payment Risk Score Index
        df_work["has_payment_issue"] = (
            (df_work["payment_failures_90d"] > 0) | (df_work["late_payments_12m"] > 0)
        ).astype(int)
        df_work["is_month_to_month"] = (df_work["contract_type"] == "Month-to-month").astype(int)
        df_work["risk_score_index"] = (
            df_work["has_high_support_tickets"] * 2
            + df_work["is_dissatisfied"] * 2
            + df_work["severe_usage_drop"] * 2
            + df_work["has_payment_issue"] * 2
            + df_work["is_month_to_month"] * 1
        )

        logger.info(f"Feature engineering complete. Output dataframe shape: {df_work.shape}")
        return df_work
