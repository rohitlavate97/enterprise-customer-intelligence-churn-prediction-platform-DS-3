"""XGBoost model training, hyperparameter tuning, early stopping, and SHAP explainability."""

import time
from typing import Any
import numpy as np
import pandas as pd
import xgboost as xgb
from evaluation.metrics import ModelEvaluator
from features.pipeline import PreprocessingPipelineBuilder
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from utils.logger import get_logger

logger = get_logger("models.xgboost_suite")

try:
    import shap
    HAS_SHAP = True
except Exception as err:
    logger.warning(f"SHAP import unavailable ({err}). Falling back to native tree feature importances.")
    HAS_SHAP = False


class XGBoostTrainer:
    """XGBoost classifier pipeline with early stopping, hyperparameter tuning, and SHAP analysis."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.best_model: xgb.XGBClassifier | None = None
        self.best_params: dict[str, Any] | None = None
        self.preprocessor: Any = None

    def build_default_model(self, scale_pos_weight: float = 1.0) -> xgb.XGBClassifier:
        """Instantiate default XGBoost classifier."""
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.seed,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def tune_hyperparameters(
        self,
        X_tr_trans: np.ndarray,
        y_tr: np.ndarray,
        cv_folds: int = 3,
        n_iter: int = 8,
    ) -> dict[str, Any]:
        """Tune XGBoost hyperparameters via RandomizedSearchCV on training data."""
        logger.info(f"Tuning XGBoost hyperparameters via RandomizedSearchCV ({n_iter} iterations)...")

        param_grid = {
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "scale_pos_weight": [1.0, 2.0, 4.0],
        }

        base_model = xgb.XGBClassifier(n_estimators=100, random_state=self.seed, eval_metric="logloss", n_jobs=-1)
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
        logger.info(f"Best XGBoost Hyperparameters: {self.best_params} (Best CV PR AUC: {search.best_score_:.4f})")
        return self.best_params

    def train_with_early_stopping(
        self,
        X_tr_trans: np.ndarray,
        y_tr: np.ndarray,
        X_val_trans: np.ndarray,
        y_val: np.ndarray,
        params: dict[str, Any] | None = None,
        early_stopping_rounds: int = 20,
    ) -> xgb.XGBClassifier:
        """Train XGBoost model using early stopping on a dedicated validation fold."""
        model_params = params or self.best_params or {}

        model = xgb.XGBClassifier(
            n_estimators=500,
            early_stopping_rounds=early_stopping_rounds,
            random_state=self.seed,
            eval_metric="logloss",
            n_jobs=-1,
            **model_params,
        )

        model.fit(
            X_tr_trans,
            y_tr,
            eval_set=[(X_val_trans, y_val)],
            verbose=False,
        )

        self.best_model = model
        logger.info(f"Trained XGBoost with early stopping. Best iteration: {model.best_iteration}")
        return model

    def compute_shap_explanations(
        self,
        model: xgb.XGBClassifier,
        X_trans: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute TreeExplainer SHAP values or native tree feature importances."""
        if HAS_SHAP:
            try:
                logger.info(f"Computing SHAP values for {X_trans.shape[0]} samples...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_trans)
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

                shap_df = None
                if feature_names and len(feature_names) == len(mean_abs_shap):
                    shap_df = pd.DataFrame(
                        {"feature": feature_names, "importance": mean_abs_shap}
                    ).sort_values(by="importance", ascending=False)

                return {
                    "explainer": explainer,
                    "shap_values": shap_values,
                    "mean_abs_shap": mean_abs_shap,
                    "shap_dataframe": shap_df,
                }
            except Exception as e:
                logger.warning(f"SHAP computation failed ({e}). Falling back to native XGBoost feature importances.")

        # Native feature importances fallback
        importances = model.feature_importances_
        shap_df = None
        if feature_names and len(feature_names) == len(importances):
            shap_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values(by="importance", ascending=False)

        return {
            "explainer": None,
            "shap_values": None,
            "mean_abs_shap": importances,
            "shap_dataframe": shap_df,
        }
