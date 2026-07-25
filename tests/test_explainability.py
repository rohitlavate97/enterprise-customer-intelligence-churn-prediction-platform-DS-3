"""Unit tests for ModelExplainer, Partial Dependence, and Segment Fairness Auditor."""

import numpy as np
import pandas as pd
import pytest
from explainability.pdp_analysis import PartialDependenceAnalyzer
from explainability.segment_fairness import SegmentFairnessAuditor
from explainability.shap_explainer import ModelExplainer
from sklearn.dummy import DummyClassifier


def test_translate_feature_to_business_reason():
    """Assert feature names are cleanly translated into plain-language business explanations."""
    text1 = ModelExplainer.translate_feature_to_business_reason("support_tickets_30d", 0.25)
    assert "support tickets" in text1.lower()
    assert "increasing" in text1.lower()

    text2 = ModelExplainer.translate_feature_to_business_reason("contract_type_Month-to-month", 0.18)
    assert "month-to-month" in text2.lower()


def test_generate_local_explanation():
    """Assert local explanation trace outputs top K drivers and narrative summary."""
    feature_names = ["support_tickets_30d", "tenure_months", "monthly_charges"]
    feature_values = [4, 3, 99.5]
    importance_scores = [0.45, -0.30, 0.10]

    local_exp = ModelExplainer.generate_local_explanation(
        feature_names, feature_values, importance_scores, top_k=2
    )

    assert len(local_exp["top_drivers"]) == 2
    assert local_exp["top_drivers"][0]["feature"] == "support_tickets_30d"
    assert "support tickets" in local_exp["summary_narrative"].lower()


def test_partial_dependence_computation():
    """Assert PartialDependenceAnalyzer calculates 1D PDP grid and average responses."""
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = DummyClassifier(strategy="prior")
    clf.fit(X, y)

    res = PartialDependenceAnalyzer.compute_1d_partial_dependence(
        clf, X, feature_index=0, feature_name="feature_0", grid_resolution=10
    )

    assert res["feature_name"] == "feature_0"
    assert len(res["grid_values"]) == 10
    assert len(res["average_predicted_probability"]) == 10


def test_segment_fairness_auditor():
    """Assert SegmentFairnessAuditor calculates error rates per segment and flags disparities."""
    df = pd.DataFrame(
        {
            "customer_id": [f"C-{i}" for i in range(100)],
            "tenure_months": np.random.randint(1, 48, size=100),
            "contract_type": np.random.choice(["Month-to-month", "Two year"], size=100),
            "churn_label": np.random.choice([0, 1], size=100, p=[0.8, 0.2]),
            "churn_probability": np.random.uniform(0.1, 0.9, size=100),
        }
    )

    audit = SegmentFairnessAuditor.audit_segment_fairness(
        df, segment_columns=["tenure_band", "contract_type"]
    )

    assert "overall_metrics" in audit
    assert "segment_breakdown" in audit
    assert "tenure_band" in audit["segment_breakdown"]
    assert "contract_type" in audit["segment_breakdown"]
    assert isinstance(audit["audit_passed"], bool)
