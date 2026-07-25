"""Unit tests for data pipeline, cleaning, versioning, quality reporting, and leakage guard."""

import pandas as pd
import pytest
from data.cleaner import DataCleaner
from data.generator import CustomerDataGenerator
from data.quality_report import DataQualityReporter
from data.schema import LEAKAGE_FIELDS, TARGET_COL
from data.versioning import DatasetVersionManager
from features.builder import FeatureBuilder
from features.leakage_guard import LeakageGuard


def test_data_cleaner_deduplication():
    """Assert DataCleaner detects and drops duplicate customer records."""
    gen = CustomerDataGenerator(n_samples=500, seed=42)
    df = gen.generate()

    # Append duplicate row
    df_dup = pd.concat([df, df.iloc[:10]], ignore_index=True)
    assert len(df_dup) == 510

    cleaner = DataCleaner(drop_duplicates=True)
    df_clean = cleaner.clean(df_dup)
    assert len(df_clean) == 500


def test_dataset_version_manager(tmp_path):
    """Assert version manager computes SHA256 checksum and writes manifest JSON."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    filepath, checksum = DatasetVersionManager.register_version(
        df, dataset_name="test_ds", output_dir=tmp_path
    )

    assert filepath.exists()
    assert len(checksum) == 64  # SHA256 length
    manifests = list(tmp_path.glob("*_manifest.json"))
    assert len(manifests) == 1


def test_data_quality_reporter():
    """Assert quality reporter computes stats, missing values, and duplicate counts."""
    gen = CustomerDataGenerator(n_samples=200, seed=12)
    df = gen.generate()

    report = DataQualityReporter.generate_report(df)
    assert report["summary"]["total_rows"] == 200
    assert report["summary"]["duplicate_rows"] == 0
    assert "missing_values" in report
    assert "target_distribution" in report
    assert report["target_distribution"]["total_count"] == 200


def test_leakage_guard_raises_on_leakage():
    """Assert LeakageGuard raises ValueError when forbidden leakage fields are present."""
    gen = CustomerDataGenerator(n_samples=100, seed=42)
    df = gen.generate()

    # df has leakage fields cancellation_processed_date, final_invoice_flag, etc.
    with pytest.raises(ValueError, match="CRITICAL DATA LEAKAGE VIOLATION"):
        LeakageGuard.assert_no_leakage(df)


def test_leakage_guard_passes_clean_dataframe():
    """Assert LeakageGuard passes without error when no leakage fields are present."""
    gen = CustomerDataGenerator(n_samples=100, seed=42)
    df = gen.generate()

    df_clean = df.drop(columns=LEAKAGE_FIELDS)
    # Should not raise exception
    LeakageGuard.assert_no_leakage(df_clean)


def test_feature_builder_leakage_stripping():
    """Assert FeatureBuilder automatically strips leakage fields and builds domain features."""
    gen = CustomerDataGenerator(n_samples=100, seed=42)
    df_raw = gen.generate()

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_feat = builder.transform(df_raw)

    # Assert all leakage fields are stripped from df_feat
    for field in LEAKAGE_FIELDS:
        assert field not in df_feat.columns

    # Assert new domain features exist
    assert "charges_per_tenure" in df_feat.columns
    assert "risk_score_index" in df_feat.columns
    assert "has_high_support_tickets" in df_feat.columns
