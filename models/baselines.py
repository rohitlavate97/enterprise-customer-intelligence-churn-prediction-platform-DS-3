"""Baseline model definitions and StratifiedKFold cross-validation suite."""

import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from evaluation.metrics import ModelEvaluator
from features.pipeline import PreprocessingPipelineBuilder
from utils.logger import get_logger

logger = get_logger("models.baselines")


def get_baseline_models(seed: int = 42) -> dict[str, Any]:
    """Instantiate dictionary of 8 baseline estimators."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
    }


class BaselineEvaluator:
    """Evaluates baseline models using StratifiedKFold cross-validation with leakage-isolated preprocessing."""

    def __init__(
        self,
        scaler_type: str = "robust",
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.scaler_type = scaler_type
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.seed = seed

    def evaluate_model_cv(
        self,
        model_name: str,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        cv_folds: Any,
    ) -> dict[str, Any]:
        """Run StratifiedKFold cross-validation for a single model with leakage-isolated preprocessing fold-by-fold."""
        logger.info(f"Evaluating baseline model '{model_name}' across {cv_folds.get_n_splits()} stratified folds...")
        y = np.asarray(y)

        pr_aucs, roc_aucs, f1s, precisions, recalls, log_losses, train_times = [], [], [], [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(cv_folds.split(X, y), 1):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            # Build fresh preprocessor
            pipe_builder = PreprocessingPipelineBuilder(
                scaler_type=self.scaler_type,
                numerical_features=self.numerical_features,
                categorical_features=self.categorical_features,
            )
            preprocessor = pipe_builder.build_pipeline()

            # Fit preprocessor ONLY on training fold
            X_tr_trans = preprocessor.fit_transform(X_tr)
            X_val_trans = preprocessor.transform(X_val)

            # Fit estimator ONLY on transformed training fold
            start_t = time.perf_counter()
            estimator.fit(X_tr_trans, y_tr)
            train_t = time.perf_counter() - start_t

            # Predict probabilities on validation fold
            if hasattr(estimator, "predict_proba"):
                y_val_proba = estimator.predict_proba(X_val_trans)[:, 1]
            else:
                y_val_proba = estimator.decision_function(X_val_trans)
                y_val_proba = (y_val_proba - y_val_proba.min()) / (y_val_proba.max() - y_val_proba.min() + 1e-8)

            metrics = ModelEvaluator.compute_all_metrics(y_val, y_val_proba)

            pr_aucs.append(metrics["pr_auc"])
            roc_aucs.append(metrics["roc_auc"])
            f1s.append(metrics["f1_score"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])
            log_losses.append(metrics["log_loss"])
            train_times.append(train_t)

        summary = {
            "model_name": model_name,
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "pr_auc_std": float(np.std(pr_aucs)),
            "roc_auc_mean": float(np.mean(roc_aucs)),
            "roc_auc_std": float(np.std(roc_aucs)),
            "f1_score_mean": float(np.mean(f1s)),
            "precision_mean": float(np.mean(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "log_loss_mean": float(np.mean(log_losses)),
            "training_time_sec": float(np.sum(train_times)),
        }

        logger.info(
            f"Baseline '{model_name}' Results -> PR AUC: {summary['pr_auc_mean']:.4f} ± {summary['pr_auc_std']:.4f} | "
            f"ROC AUC: {summary['roc_auc_mean']:.4f} | F1: {summary['f1_score_mean']:.4f}"
        )
        return summary
