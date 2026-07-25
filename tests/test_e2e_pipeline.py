"""End-to-End System Integration Test Suite validating full platform lifecycle."""

import time
import pytest
from config.settings import settings
from scripts.run_full_pipeline import run_all_stages


def test_e2e_platform_lifecycle():
    """Assert master workflow orchestrator executes all 11 stages and generates all required artifacts."""
    success = run_all_stages()
    assert success is True

    # Verify key output artifacts
    artifacts = [
        settings.raw_data_dir / "customer_churn_dataset.csv",
        settings.artifacts_dir / "preprocessing_pipeline.joblib",
        settings.artifacts_dir / "catboost_model.joblib",
        settings.artifacts_dir / "data_quality_report.json",
    ]

    for art in artifacts:
        assert art.exists(), f"Artifact missing: {art}"
