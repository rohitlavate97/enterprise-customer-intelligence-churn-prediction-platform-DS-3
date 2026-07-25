"""Unit tests for synthetic dataset generator, schema, leakage fields, and ground-truth signal recovery."""

import pandas as pd
import pytest
from data.generator import CustomerDataGenerator
from data.schema import (
    ALL_PREDICTOR_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLS,
    LEAKAGE_FIELDS,
    NUMERICAL_FEATURES,
    TARGET_COL,
)


def test_generator_reproducibility():
    """Assert generator produces identical dataframes when initialized with identical seeds."""
    gen1 = CustomerDataGenerator(n_samples=1000, seed=42)
    gen2 = CustomerDataGenerator(n_samples=1000, seed=42)

    df1 = gen1.generate()
    df2 = gen2.generate()

    pd.testing.assert_frame_equal(df1, df2)


def test_generator_schema_compliance():
    """Assert generated dataframe contains all required features, ID, target, and leakage fields."""
    gen = CustomerDataGenerator(n_samples=1000, seed=123)
    df = gen.generate()

    # Check row count
    assert len(df) == 1000

    # Check target existence
    assert TARGET_COL in df.columns

    # Check ID column
    for col in ID_COLS:
        assert col in df.columns

    # Check all predictor features exist
    for col in ALL_PREDICTOR_FEATURES:
        assert col in df.columns

    # Check leakage fields exist
    for col in LEAKAGE_FIELDS:
        assert col in df.columns


def test_target_leakage_fields_behavior():
    """Assert target leakage fields occur ONLY when churn_label is 1."""
    gen = CustomerDataGenerator(n_samples=2000, seed=99)
    df = gen.generate()

    non_churners = df[df[TARGET_COL] == 0]

    # Non-churners must have null / zero leakage indicators
    assert non_churners["cancellation_processed_date"].isna().all()
    assert (non_churners["final_invoice_flag"] == 0).all()
    assert (non_churners["account_status_deactivated"] == 0).all()
    assert non_churners["churn_reason_recorded"].isna().all()

    churners = df[df[TARGET_COL] == 1]
    assert churners["cancellation_processed_date"].notna().all()
    assert (churners["final_invoice_flag"] == 1).all()
    assert (churners["account_status_deactivated"] == 1).all()
    assert churners["churn_reason_recorded"].notna().all()


def test_ground_truth_signal_recovery():
    """Ground-truth recovery test: assert key engineered predictors have strong empirical relationship with target."""
    gen = CustomerDataGenerator(n_samples=10000, seed=42)
    df = gen.generate()

    # 1. Support tickets 30d correlation with churn
    corr_tickets = df["support_tickets_30d"].corr(df[TARGET_COL])
    assert corr_tickets > 0.15, f"Support tickets correlation too low: {corr_tickets:.4f}"

    # 2. Payment failures 90d correlation with churn
    corr_failures = df["payment_failures_90d"].corr(df[TARGET_COL])
    assert corr_failures > 0.15, f"Payment failures correlation too low: {corr_failures:.4f}"

    # 3. Month-to-month contract churn rate higher than Two year
    m2m_churn_rate = df[df["contract_type"] == "Month-to-month"][TARGET_COL].mean()
    two_year_churn_rate = df[df["contract_type"] == "Two year"][TARGET_COL].mean()
    assert m2m_churn_rate > two_year_churn_rate + 0.10, (
        f"Contract type signal failed: M2M ({m2m_churn_rate:.2f}) vs 2-Yr ({two_year_churn_rate:.2f})"
    )


def test_generator_save(tmp_path):
    """Assert generator saves CSV and JSON manifest to specified directory."""
    gen = CustomerDataGenerator(n_samples=500, seed=7)
    df = gen.generate()

    csv_path = gen.save(df, output_dir=tmp_path)
    assert csv_path.exists()
    manifest_path = tmp_path / "data_manifest.json"
    assert manifest_path.exists()
