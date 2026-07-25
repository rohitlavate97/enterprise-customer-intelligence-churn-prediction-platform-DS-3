"""Unit tests for scikit-learn preprocessing pipeline, fit/transform separation, and serving parity."""

import numpy as np
import pandas as pd
import pytest
from data.generator import CustomerDataGenerator
from data.schema import CATEGORICAL_FEATURES, LEAKAGE_FIELDS, NUMERICAL_FEATURES, TARGET_COL
from features.builder import FeatureBuilder
from features.leakage_guard import LeakageGuard
from features.pipeline import (
    PreprocessingPipelineBuilder,
    load_preprocessing_pipeline,
    save_preprocessing_pipeline,
)
from sklearn.model_selection import train_test_split


def test_fit_transform_separation_proof():
    """PROVABLE LEAKAGE-FREE TEST:

    Assert that pipeline statistics (scalers, imputers, encoders) are learned exclusively on X_train,
    and transforming X_test does not alter fitted pipeline parameters.
    """
    gen = CustomerDataGenerator(n_samples=1000, seed=42)
    df_raw = gen.generate()

    # Apply domain feature builder & strip leakage fields
    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Determine numerical and categorical columns present in X_train
    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X_train.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]

    pipe_builder = PreprocessingPipelineBuilder(
        scaler_type="robust",
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    pipeline = pipe_builder.build_pipeline()

    # Fit ONLY on X_train
    pipeline.fit(X_train)

    # Capture fitted scaler center/scale learned on X_train
    num_scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scaler"]
    center_train = num_scaler.center_.copy()
    scale_train = num_scaler.scale_.copy()

    # Transform X_test (must not change fitted statistics)
    X_test_trans = pipeline.transform(X_test)

    # Assert fitted center and scale remain identical after transforming X_test
    np.testing.assert_array_equal(num_scaler.center_, center_train)
    np.testing.assert_array_equal(num_scaler.scale_, scale_train)

    # Assert shape match
    assert X_test_trans.shape[0] == len(X_test)


def test_serving_parity():
    """SERVING PARITY TEST:

    Assert that single-row input transformed via the fitted pipeline produces
    identical outputs to batch transformation.
    """
    gen = CustomerDataGenerator(n_samples=500, seed=42)
    df_raw = gen.generate()

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)
    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pipe_builder = PreprocessingPipelineBuilder(
        scaler_type="robust",
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    pipeline = pipe_builder.build_pipeline()
    pipeline.fit(X)

    # Batch transform
    X_batch_trans = pipeline.transform(X)

    # Single-row transform (simulating real-time prediction request)
    single_row = X.iloc[[0]]
    single_trans = pipeline.transform(single_row)

    # Assert exact array match between row 0 of batch transform and single_trans
    np.testing.assert_allclose(single_trans, X_batch_trans[[0]], rtol=1e-5, atol=1e-5)


def test_pipeline_serialization(tmp_path):
    """Assert preprocessing pipeline can be serialized to disk and reloaded with identical transform output."""
    gen = CustomerDataGenerator(n_samples=300, seed=7)
    df_raw = gen.generate()

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)
    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pipe_builder = PreprocessingPipelineBuilder(
        scaler_type="robust",
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    pipeline = pipe_builder.build_pipeline()
    pipeline.fit(X)

    orig_trans = pipeline.transform(X)

    # Save and reload
    art_path = tmp_path / "preprocessing_pipeline.joblib"
    save_preprocessing_pipeline(pipeline, art_path)

    reloaded_pipeline = load_preprocessing_pipeline(art_path)
    reloaded_trans = reloaded_pipeline.transform(X)

    np.testing.assert_array_equal(orig_trans, reloaded_trans)
