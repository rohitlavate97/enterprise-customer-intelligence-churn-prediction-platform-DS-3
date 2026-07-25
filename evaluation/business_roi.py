"""Business Analytics and Retention Campaign ROI Engine."""

from typing import Any
import numpy as np
import pandas as pd
from explainability.shap_explainer import ModelExplainer
from utils.logger import get_logger

logger = get_logger("evaluation.business_roi")


class BusinessROIAnalyzer:
    """Calculates CLV, Total Revenue at Risk, Retention Intervention ROI, and High-Risk Priority Call Lists."""

    @staticmethod
    def calculate_clv(
        monthly_charges: pd.Series | np.ndarray,
        tenure_months: pd.Series | np.ndarray,
        average_lifespan_months: float = 36.0,
        gross_margin: float = 0.75,
    ) -> np.ndarray:
        """Calculate Customer Lifetime Value (CLV) based on monthly charges and remaining lifespan."""
        remaining_months = np.maximum(1, average_lifespan_months - np.asarray(tenure_months))
        clv = np.asarray(monthly_charges) * remaining_months * gross_margin
        return np.round(clv, 2)

    @classmethod
    def analyze_financial_impact(
        cls,
        df_customers: pd.DataFrame,
        churn_prob_col: str = "churn_probability",
        probability_threshold: float = 0.50,
        intervention_cost_per_cust: float = 50.0,
        intervention_success_rate: float = 0.25,
    ) -> dict[str, Any]:
        """Analyze total revenue at risk and calculate expected ROI for a targeted retention campaign."""
        df = df_customers.copy()

        if "clv" not in df.columns:
            df["clv"] = cls.calculate_clv(df["monthly_charges"], df["tenure_months"])

        df["expected_loss"] = df[churn_prob_col] * df["clv"]
        df["is_high_risk"] = (df[churn_prob_col] >= probability_threshold).astype(int)

        total_customers = len(df)
        total_portfolio_clv = float(df["clv"].sum())

        high_risk_df = df[df["is_high_risk"] == 1]
        n_targeted = len(high_risk_df)
        total_revenue_at_risk = float(high_risk_df["clv"].sum())

        # Retention Intervention Modeling
        saved_customers = int(np.round(n_targeted * intervention_success_rate))
        total_campaign_cost = float(n_targeted * intervention_cost_per_cust)
        gross_saved_revenue = float(high_risk_df["clv"].mean() * saved_customers) if n_targeted > 0 else 0.0
        net_saved_revenue = gross_saved_revenue - total_campaign_cost

        roi_pct = float((net_saved_revenue / total_campaign_cost) * 100.0) if total_campaign_cost > 0 else 0.0

        metrics = {
            "total_customers": total_customers,
            "total_portfolio_clv": total_portfolio_clv,
            "high_risk_customer_count": n_targeted,
            "high_risk_prevalence": float(n_targeted / total_customers),
            "total_revenue_at_risk": total_revenue_at_risk,
            "campaign_params": {
                "probability_threshold": probability_threshold,
                "intervention_cost_per_customer": intervention_cost_per_cust,
                "assumed_success_rate": intervention_success_rate,
            },
            "campaign_outcomes": {
                "targeted_count": n_targeted,
                "projected_saved_customers": saved_customers,
                "total_campaign_cost": total_campaign_cost,
                "gross_saved_revenue": gross_saved_revenue,
                "net_saved_revenue": net_saved_revenue,
                "campaign_roi_pct": roi_pct,
            },
        }

        logger.info(
            f"Business ROI Analysis: {n_targeted} High-Risk Customers | Revenue at Risk: ${total_revenue_at_risk:,.2f} | "
            f"Projected Net Saved Revenue: ${net_saved_revenue:,.2f} (ROI: {roi_pct:.1f}%)"
        )
        return metrics

    @classmethod
    def generate_high_risk_call_list(
        cls,
        df_customers: pd.DataFrame,
        feature_names: list[str] | None = None,
        churn_prob_col: str = "churn_probability",
        top_n: int = 100,
    ) -> pd.DataFrame:
        """Generate prioritized retention call list sorted by expected dollar loss (Churn Probability * CLV)."""
        df = df_customers.copy()
        if "clv" not in df.columns:
            df["clv"] = cls.calculate_clv(df["monthly_charges"], df["tenure_months"])

        df["expected_dollar_risk"] = np.round(df[churn_prob_col] * df["clv"], 2)

        sorted_df = df.sort_values(by="expected_dollar_risk", ascending=False).head(top_n).reset_index(drop=True)

        call_list = []
        for idx, row in sorted_df.iterrows():
            # Action recommendation based on contract & tickets
            if row.get("contract_type") == "Month-to-month" and row.get("support_tickets_30d", 0) >= 2:
                rec = "Offer 12-month contract upgrade discount + priority support callback."
            elif row.get("price_increase_applied_30d") == "Yes":
                rec = "Offer price lock guarantee for 6 months."
            else:
                rec = "Proactive account review call by VIP retention specialist."

            call_list.append(
                {
                    "rank": idx + 1,
                    "customer_id": row["customer_id"],
                    "churn_probability": float(row[churn_prob_col]),
                    "clv": float(row["clv"]),
                    "expected_dollar_risk": float(row["expected_dollar_risk"]),
                    "contract_type": row.get("contract_type", "N/A"),
                    "support_tickets_30d": row.get("support_tickets_30d", 0),
                    "action_recommendation": rec,
                }
            )

        return pd.DataFrame(call_list)
