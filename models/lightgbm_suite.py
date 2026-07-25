"""LightGBM model training, leaf-wise histogram tuning, early stopping, and benchmarking."""

import time
import warnings
from typing import Any
import lightgbm as lgb
import numpy as np
import pandas as pd
from evaluation.metrics import ModelEvaluator
from features.pipeline import PreprocessingPipelineBuilder
from sklearn.model_selection import RandomizedSearchCV
from utils.logger import get_logger

logger = get_logger("models.lightgbm_suite")


class LightGBMTrainer:
    """LightGBM classifier pipeline with leaf-wise histogram growth, early stopping, and tuning."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.best_model: lgb.LGBMClassifier | None = None
        self.best_params: dict[str, Any] | None = None

    def build_default_model(self, scale_pos_weight: float = 1.0) -> lgb.LGBMClassifier:
        """Instantiate default LightGBM classifier."""
        return lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=-1,
        )

    def tune_hyperparameters(
        self,
        X_tr_trans: np.ndarray,
        y_tr: np.ndarray,
        cv_folds: int = 3,
        n_iter: int = 8,
    ) -> dict[str, Any]:
        """Tune LightGBM hyperparameters via RandomizedSearchCV on training data."""
        logger.info(f"Tuning LightGBM hyperparameters via RandomizedSearchCV ({n_iter} iterations)...")

        param_grid = {
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [4, 6, 8, -1],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "scale_pos_weight": [1.0, 2.0, 4.0],
        }

        base_model = lgb.LGBMClassifier(n_estimators=100, random_state=self.seed, n_jobs=-1, verbosity=-1)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv_folds,
            random_state=self.seed,
            n_jobs=-1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_tr_trans, y_tr)

        self.best_params = search.best_params_
        logger.info(f"Best LightGBM Hyperparameters: {self.best_params} (Best CV PR AUC: {search.best_score_:.4f})")
        return self.best_params

    def train_with_early_stopping(
        self,
        X_tr_trans: np.ndarray,
        y_tr: np.ndarray,
        X_val_trans: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any] | None = None,
        early_stopping_rounds: int = 20,
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model using early stopping callback on a dedicated validation fold."""
        model_params = params or self.best_params or {}

        model = lgb.LGBMClassifier(
            n_estimators=500,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=-1,
            **model_params,
        )

        callbacks = [lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]

        start_t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_tr_trans,
                y_tr,
                eval_set=[(X_val_trans, y_val)],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
        train_time = time.perf_counter() - start_t

        self.best_model = model
        logger.info(f"Trained LightGBM with early stopping in {train_time:.3f}s. Best iteration: {model.best_iteration_}")
        return model
