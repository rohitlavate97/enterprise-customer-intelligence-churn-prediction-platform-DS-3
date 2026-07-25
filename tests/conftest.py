"""Shared Pytest fixtures for platform test suite."""

from pathlib import Path
import pytest
from config.settings import Settings


@pytest.fixture
def sample_config() -> Settings:
    """Fixture providing isolated Settings instance."""
    return Settings(environment="testing", seed=42)


@pytest.fixture
def sample_customer_record() -> dict:
    """Fixture providing a mock single customer record for feature testing."""
    return {
        "customer_id": "CUST-10001",
        "age": 35,
        "tenure_months": 24,
        "monthly_charges": 75.5,
        "total_charges": 1812.0,
        "contract_type": "One year",
        "payment_method": "Credit card",
        "support_tickets_30d": 1,
        "payment_failures_90d": 0,
        "churn_label": 0,
    }
