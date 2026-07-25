"""CatBoost model training, ordered boosting, native categorical handling, and benchmarking."""

import time
from typing import Any
import catboost as cb
import numpy as np
import pandas as pd
from evaluation.metrics import ModelEvaluator
from features.pipeline import PreprocessingPipelineBuilder
from sklearn.model_selection import RandomizedSearchCV
from utils.logger import get_logger

logger = get_logger("models.catboost_suite")


class CatBoostTrainer:
    """CatBoost classifier pipeline with ordered target encoding, early stopping, and hyperparameter tuning."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.best_model: cb.CatBoostClassifier | None = None
        self.best_params: dict[str, Any] | None = None

    def build_default_model(self, scale_pos_weight: float = 1.0) -> cb.CatBoostClassifier:
        """Instantiate default CatBoost classifier with ordered boosting."""
        return cb.CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_seed=self.seed,
            thread_count=-1,
            verbose=False,
        )

    def tune_hyperparameters(
        self,
        X_tr_trans: np.ndarray,
        y_tr: np.ndarray,
        cv_folds: int = 3,
        n_iter: int = 6,
    ) -> dict[str, Any]:
        """Tune CatBoost hyperparameters via RandomizedSearchCV on training data."""
        logger.info(f"Tuning CatBoost hyperparameters via RandomizedSearchCV ({n_iter} iterations)...")

        param_grid = {
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5, 7],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "scale_pos_weight": [1.0, 2.0, 4.0],
        }

        base_model = cb.CatBoostClassifier(iterations=150, random_seed=self.seed, thread_count=-1, verbose=False)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv_folds,
            random_state=self.seed,
            n_jobs=-1,
        )

        search.fit(X_tr_trans, y_tr)
        self.best_params = search.best_params_
        logger.info(f"Best CatBoost Hyperparameters: {self.best_params} (Best CV PR AUC: {search.best_score_:.4f})")
        return self.best_params

    def train_with_early_stopping(
        self,
        X_tr_trans: np.ndarray,
        y_tr: np.ndarray,
        X_val_trans: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any] | None = None,
        early_stopping_rounds: int = 20,
    ) -> cb.CatBoostClassifier:
        """Train CatBoost model using early stopping on a dedicated validation fold."""
        model_params = params or self.best_params or {}

        model = cb.CatBoostClassifier(
            iterations=600,
            early_stopping_rounds=early_stopping_rounds,
            random_seed=self.seed,
            thread_count=-1,
            verbose=False,
            **model_params,
        )

        start_t = time.perf_counter()
        model.fit(
            X_tr_trans,
            y_tr,
            eval_set=(X_val_trans, y_val),
            use_best_model=True,
        )
        train_time = time.perf_counter() - start_t

        self.best_model = model
        logger.info(f"Trained CatBoost with early stopping in {train_time:.3f}s. Best iteration: {model.get_best_iteration()}")
        return model
