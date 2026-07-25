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
        total_chg = df_work["total_charges"] if "total_charges" in df_work.columns else df_work["monthly_charges"] * df_work["tenure_months"]
        tenure_m = np.maximum(df_work.get("tenure_months", pd.Series([12] * len(df_work))), 1)

        df_work["charges_per_tenure"] = np.round(total_chg / tenure_m, 2)
        df_work["monthly_to_total_ratio"] = np.round(
            df_work["monthly_charges"] / np.maximum(total_chg, 1.0), 4
        )

        # 2. Support & Dissatisfaction Ratios
        tickets_90d = df_work.get("support_tickets_90d", df_work.get("support_tickets_30d", pd.Series([0] * len(df_work))) * 3)
        df_work["support_tickets_per_tenure"] = np.round(tickets_90d / tenure_m, 3)

        tickets_30d = df_work.get("support_tickets_30d", pd.Series([0] * len(df_work)))
        df_work["has_high_support_tickets"] = (tickets_30d >= 3).astype(int)

        sat_score = df_work.get("resolution_satisfaction_score", pd.Series([3.0] * len(df_work)))
        df_work["is_dissatisfied"] = (sat_score <= 2).astype(int)

        # 3. Usage & Engagement Signals
        gb_down = df_work.get("avg_monthly_gb_download", pd.Series([100.0] * len(df_work)))
        act_days = np.maximum(df_work.get("active_days_per_month", pd.Series([20] * len(df_work))), 1)
        df_work["gb_per_active_day"] = np.round(gb_down / act_days, 2)

        usage_change = df_work.get("change_in_usage_pct", pd.Series([0.0] * len(df_work)))
        df_work["severe_usage_drop"] = (usage_change < -0.25).astype(int)

        login_freq = df_work.get("login_frequency_30d", df_work.get("app_logins_30d", pd.Series([10] * len(df_work))))
        df_work["low_engagement_flag"] = (
            (act_days <= 5) & (login_freq <= 3)
        ).astype(int)

        # 4. Payment Risk Score Index
        pay_fail_90d = df_work.get("payment_failures_90d", pd.Series([0] * len(df_work)))
        late_pay_12m = df_work.get("late_payments_12m", pd.Series([0] * len(df_work)))
        df_work["has_payment_issue"] = (
            (pay_fail_90d > 0) | (late_pay_12m > 0)
        ).astype(int)

        contract_t = df_work.get("contract_type", pd.Series(["Month-to-month"] * len(df_work)))
        df_work["is_month_to_month"] = (contract_t == "Month-to-month").astype(int)

        df_work["risk_score_index"] = (
            df_work["has_high_support_tickets"] * 2
            + df_work["is_dissatisfied"] * 2
            + df_work["severe_usage_drop"] * 2
            + df_work["has_payment_issue"] * 2
            + df_work["is_month_to_month"] * 1
        )

        logger.info(f"Feature engineering complete. Output dataframe shape: {df_work.shape}")
        return df_work
