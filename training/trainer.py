"""Unified Model Training and Cross-Validation Suite."""

import json
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from config.settings import settings
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from models.baselines import BaselineEvaluator, get_baseline_models
from utils.logger import get_logger

logger = get_logger("training.trainer")


class ModelTrainer:
    """Orchestrates model training, StratifiedKFold evaluation, and benchmark logging."""

    def __init__(
        self,
        n_splits: int = 5,
        seed: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.seed = seed
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def train_and_evaluate_baselines(
        self,
        df_featured: pd.DataFrame,
        selected_models: list[str] | None = None,
    ) -> pd.DataFrame:
        """Evaluate baseline models on dataset using StratifiedKFold."""
        X = df_featured.drop(columns=[TARGET_COL, "customer_id"], errors="ignore")
        y = df_featured[TARGET_COL]

        num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

        evaluator = BaselineEvaluator(
            numerical_features=num_cols,
            categorical_features=cat_cols,
            seed=self.seed,
        )

        models_dict = get_baseline_models(seed=self.seed)
        if selected_models:
            models_dict = {name: model for name, model in models_dict.items() if name in selected_models}

        results = []
        for model_name, estimator in models_dict.items():
            res = evaluator.evaluate_model_cv(
                model_name=model_name,
                estimator=estimator,
                X=X,
                y=y,
                cv_folds=self.cv,
            )
            results.append(res)

        results_df = pd.DataFrame(results).sort_values(by="pr_auc_mean", ascending=False).reset_index(drop=True)
        return results_df
