"""Unit tests for XGBoost trainer, early stopping, hyperparameter tuning, and SHAP explainability."""

import numpy as np
import pandas as pd
import pytest
from data.generator import CustomerDataGenerator
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.metrics import ModelEvaluator
from features.builder import FeatureBuilder
from features.pipeline import PreprocessingPipelineBuilder
from models.xgboost_suite import XGBoostTrainer
from sklearn.model_selection import train_test_split


def test_xgboost_training_and_early_stopping():
    """Assert XGBoost trains with early stopping and produces calibrated probabilities."""
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

    trainer = XGBoostTrainer(seed=42)
    model = trainer.train_with_early_stopping(
        X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy(), early_stopping_rounds=15
    )

    y_val_proba = model.predict_proba(X_val_trans)[:, 1]

    assert len(y_val_proba) == len(X_val)
    assert (y_val_proba >= 0.0).all() and (y_val_proba <= 1.0).all()

    metrics = ModelEvaluator.compute_all_metrics(y_val.to_numpy(), y_val_proba)
    assert metrics["pr_auc"] > 0.70
    assert metrics["roc_auc"] > 0.85


def test_xgboost_shap_explanations():
    """Assert SHAP or native feature importance produces valid importance vector and dataframe."""
    gen = CustomerDataGenerator(n_samples=500, seed=42)
    df_raw = gen.generate()

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pipe_builder = PreprocessingPipelineBuilder(
        scaler_type="robust",
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    preprocessor = pipe_builder.build_pipeline()
    X_trans = preprocessor.fit_transform(X)

    trainer = XGBoostTrainer(seed=42)
    model = trainer.build_default_model()
    model.fit(X_trans, y.to_numpy())

    feature_names = list(preprocessor.named_steps["preprocessor"].get_feature_names_out())
    shap_out = trainer.compute_shap_explanations(model, X_trans[:50], feature_names=feature_names)

    assert len(shap_out["mean_abs_shap"]) == X_trans.shape[1]
    assert shap_out["shap_dataframe"] is not None
    assert len(shap_out["shap_dataframe"]) == X_trans.shape[1]
