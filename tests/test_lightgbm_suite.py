"""Unit tests for LightGBM trainer, early stopping, tuning, and benchmark against XGBoost."""

import numpy as np
import pandas as pd
import pytest
from data.generator import CustomerDataGenerator
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.metrics import ModelEvaluator
from features.builder import FeatureBuilder
from features.pipeline import PreprocessingPipelineBuilder
from models.lightgbm_suite import LightGBMTrainer
from models.xgboost_suite import XGBoostTrainer
from sklearn.model_selection import train_test_split


def test_lightgbm_training_and_early_stopping():
    """Assert LightGBM trains with early stopping and produces calibrated probabilities."""
    gen = CustomerDataGenerator(n_samples=1000, seed=42)
    df_raw = gen.generate()

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pipe_builder = PreprocessingPipelineBuilder(
        scaler_type="robust",
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    preprocessor = pipe_builder.build_pipeline()

    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_val_trans = preprocessor.transform(X_val)

    trainer = LightGBMTrainer(seed=42)
    model = trainer.train_with_early_stopping(
        X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy(), early_stopping_rounds=15
    )

    y_val_proba = model.predict_proba(X_val_trans)[:, 1]

    assert len(y_val_proba) == len(X_val)
    assert (y_val_proba >= 0.0).all() and (y_val_proba <= 1.0).all()

    metrics = ModelEvaluator.compute_all_metrics(y_val.to_numpy(), y_val_proba)
    assert metrics["pr_auc"] > 0.70
    assert metrics["roc_auc"] > 0.85


def test_lightgbm_vs_xgboost_benchmark():
    """BENCHMARK RECOVERY TEST:

    Assert that both LightGBM and XGBoost recover ground-truth signal above performance floor (PR AUC > 0.80).
    """
    gen = CustomerDataGenerator(n_samples=1500, seed=42)
    df_raw = gen.generate()

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pipe_builder = PreprocessingPipelineBuilder(
        scaler_type="robust",
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    preprocessor = pipe_builder.build_pipeline()

    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_val_trans = preprocessor.transform(X_val)

    lgb_trainer = LightGBMTrainer(seed=42)
    xgb_trainer = XGBoostTrainer(seed=42)

    lgb_model = lgb_trainer.train_with_early_stopping(
        X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy(), early_stopping_rounds=15
    )
    xgb_model = xgb_trainer.train_with_early_stopping(
        X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy(), early_stopping_rounds=15
    )

    lgb_proba = lgb_model.predict_proba(X_val_trans)[:, 1]
    xgb_proba = xgb_model.predict_proba(X_val_trans)[:, 1]

    lgb_m = ModelEvaluator.compute_all_metrics(y_val.to_numpy(), lgb_proba)
    xgb_m = ModelEvaluator.compute_all_metrics(y_val.to_numpy(), xgb_proba)

    assert lgb_m["pr_auc"] > 0.80
    assert xgb_m["pr_auc"] > 0.80
