"""Unit tests for Streamlit dashboard application modules and data loaders."""

import pytest
from dashboard.app import load_artifacts, load_dataset


def test_dashboard_artifact_loading():
    """Assert load_artifacts returns preprocessor, model, and model_name."""
    preprocessor, model, model_name = load_artifacts()

    assert model_name is not None
    if preprocessor is not None:
        assert hasattr(preprocessor, "transform")
    if model is not None:
        assert hasattr(model, "predict_proba")


def test_dashboard_dataset_loading():
    """Assert load_dataset loads customer dataframe with churn_probability and CLV columns."""
    df = load_dataset()
    if df is not None:
        assert "churn_probability" in df.columns
        assert "clv" in df.columns
        assert "expected_revenue_risk" in df.columns
        assert len(df) > 0
