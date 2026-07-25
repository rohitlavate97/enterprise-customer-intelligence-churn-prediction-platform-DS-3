"""Unit tests for evaluator metrics, baseline models, and StratifiedKFold evaluation."""

import numpy as np
import pandas as pd
import pytest
from data.generator import CustomerDataGenerator
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.metrics import ModelEvaluator
from features.builder import FeatureBuilder
from models.baselines import BaselineEvaluator, get_baseline_models
from sklearn.model_selection import StratifiedKFold


def test_model_evaluator_metrics():
    """Assert ModelEvaluator computes all metrics accurately."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = ModelEvaluator.compute_all_metrics(y_true, y_proba, threshold=0.5)

    assert metrics["pr_auc"] > 0.80
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["confusion_matrix"]["true_positives"] == 4
    assert metrics["confusion_matrix"]["true_negatives"] == 4


def test_baseline_models_dictionary():
    """Assert get_baseline_models returns 8 configured scikit-learn estimators."""
    models = get_baseline_models(seed=42)
    assert len(models) == 8
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "Gradient Boosting" in models
    assert "SVM" in models
    assert "Naive Bayes" in models


def test_baseline_evaluator_cv():
    """Assert StratifiedKFold baseline evaluation runs cleanly and returns metric summary."""
    gen = CustomerDataGenerator(n_samples=500, seed=42)
    df_raw = gen.generate()

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    evaluator = BaselineEvaluator(
        numerical_features=num_cols,
        categorical_features=cat_cols,
        seed=42,
    )

    models = get_baseline_models(seed=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Evaluate Logistic Regression
    res = evaluator.evaluate_model_cv("Logistic Regression", models["Logistic Regression"], X, y, cv)

    assert res["model_name"] == "Logistic Regression"
    assert 0.0 <= res["pr_auc_mean"] <= 1.0
    assert 0.0 <= res["roc_auc_mean"] <= 1.0
    assert res["training_time_sec"] > 0.0


def test_ground_truth_recovery_performance_floor():
    """GROUND-TRUTH RECOVERY TEST:

    Assert that baseline models recover engineered ground-truth signal above performance floor (PR AUC > 0.40).
    """
    gen = CustomerDataGenerator(n_samples=2000, seed=42)
    df_raw = gen.generate()

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_raw)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    evaluator = BaselineEvaluator(
        numerical_features=num_cols,
        categorical_features=cat_cols,
        seed=42,
    )

    models = get_baseline_models(seed=42)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    res_lr = evaluator.evaluate_model_cv("Logistic Regression", models["Logistic Regression"], X, y, cv)
    res_rf = evaluator.evaluate_model_cv("Random Forest", models["Random Forest"], X, y, cv)

    # Base prevalence is 0.18. Signal recovery floor must be > 0.40 PR AUC!
    assert res_lr["pr_auc_mean"] > 0.40, f"Logistic Regression PR AUC too low: {res_lr['pr_auc_mean']:.4f}"
    assert res_rf["pr_auc_mean"] > 0.40, f"Random Forest PR AUC too low: {res_rf['pr_auc_mean']:.4f}"
