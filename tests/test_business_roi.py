"""Unit tests for Business Analytics, CLV, Revenue at Risk, and Retention Call List."""

import numpy as np
import pandas as pd
import pytest
from evaluation.business_roi import BusinessROIAnalyzer


def test_calculate_clv():
    """Assert CLV calculation formula output."""
    monthly_charges = np.array([100.0, 50.0])
    tenure_months = np.array([12, 24])

    clv = BusinessROIAnalyzer.calculate_clv(monthly_charges, tenure_months, average_lifespan_months=36, gross_margin=0.75)

    # Cust 1: 100 * (36 - 12) * 0.75 = 100 * 24 * 0.75 = 1800.0
    # Cust 2: 50 * (36 - 24) * 0.75 = 50 * 12 * 0.75 = 450.0
    assert clv[0] == 1800.0
    assert clv[1] == 450.0


def test_analyze_financial_impact():
    """Assert financial impact metrics and intervention ROI calculation."""
    df = pd.DataFrame(
        {
            "customer_id": [f"C-{i}" for i in range(10)],
            "monthly_charges": [100.0] * 10,
            "tenure_months": [12] * 10,  # CLV = 1800.0
            "churn_probability": [0.9] * 5 + [0.1] * 5,
        }
    )

    res = BusinessROIAnalyzer.analyze_financial_impact(
        df, churn_prob_col="churn_probability", probability_threshold=0.50, intervention_cost_per_cust=50.0, intervention_success_rate=0.20
    )

    assert res["high_risk_customer_count"] == 5
    assert res["total_revenue_at_risk"] == 5 * 1800.0  # 9000.0
    assert res["campaign_outcomes"]["targeted_count"] == 5
    assert res["campaign_outcomes"]["total_campaign_cost"] == 5 * 50.0  # 250.0
    assert res["campaign_outcomes"]["projected_saved_customers"] == 1  # 5 * 0.20
    assert res["campaign_outcomes"]["gross_saved_revenue"] == 1800.0
    assert res["campaign_outcomes"]["net_saved_revenue"] == 1550.0  # 1800 - 250
    assert res["campaign_outcomes"]["campaign_roi_pct"] == (1550.0 / 250.0) * 100.0  # 620.0%


def test_generate_high_risk_call_list():
    """Assert high-risk call list is sorted by expected dollar risk."""
    df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "monthly_charges": [100.0, 200.0, 50.0],
            "tenure_months": [12, 12, 12],
            "churn_probability": [0.80, 0.90, 0.10],
            "contract_type": ["Month-to-month", "Two year", "Month-to-month"],
            "support_tickets_30d": [3, 0, 0],
            "price_increase_applied_30d": ["No", "Yes", "No"],
        }
    )

    call_list = BusinessROIAnalyzer.generate_high_risk_call_list(df, churn_prob_col="churn_probability", top_n=2)

    assert len(call_list) == 2
    assert call_list.iloc[0]["customer_id"] == "C2"  # Expected dollar risk: 0.90 * 3600 = 3240
    assert call_list.iloc[1]["customer_id"] == "C1"  # Expected dollar risk: 0.80 * 1800 = 1440
    assert "action_recommendation" in call_list.columns
