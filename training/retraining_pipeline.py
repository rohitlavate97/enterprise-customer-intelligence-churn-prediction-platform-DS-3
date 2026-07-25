"""Automated Retraining Pipeline & Champion vs Challenger Validation Gate."""

import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.cleaner import DataCleaner
from data.generator import CustomerDataGenerator
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.metrics import ModelEvaluator
from features.builder import FeatureBuilder
from features.pipeline import PreprocessingPipelineBuilder
from models.catboost_suite import CatBoostTrainer
from training.model_registry import ModelRegistry
from utils.logger import get_logger

logger = get_logger("training.retraining_pipeline")


class AutomatedRetrainingPipeline:
    """Retrains candidate model and evaluates Champion vs Challenger validation gate (1.0% relative PR AUC margin)."""

    def __init__(self, relative_margin_threshold: float = 0.01) -> None:
        self.relative_margin_threshold = relative_margin_threshold
        self.registry = ModelRegistry()

    def run_retraining_cycle(self, n_samples: int = 1500, seed: int = 42) -> dict[str, Any]:
        """Execute full retraining cycle: generate data, train Challenger, compare vs Champion, promote if gate passed."""
        logger.info("Starting Automated Retraining Cycle...")

        # 1. Fetch / Generate Fresh Training Data Window
        gen = CustomerDataGenerator(n_samples=n_samples, seed=seed + 10)
        df_raw = gen.generate()
        df_clean = DataCleaner().clean(df_raw)
        df_featured = FeatureBuilder(enforce_leakage_guard=True).transform(df_clean)

        X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
        y = df_featured[TARGET_COL]

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

        num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

        pipe_builder = PreprocessingPipelineBuilder(scaler_type="robust", numerical_features=num_cols, categorical_features=cat_cols)
        preprocessor = pipe_builder.build_pipeline()

        X_tr_trans = preprocessor.fit_transform(X_tr)
        X_val_trans = preprocessor.transform(X_val)

        # 2. Train Challenger Candidate Model
        trainer = CatBoostTrainer(seed=seed)
        challenger_model = trainer.train_with_early_stopping(
            X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy(), early_stopping_rounds=20
        )

        y_val_challenger_prob = challenger_model.predict_proba(X_val_trans)[:, 1]
        challenger_metrics = ModelEvaluator.compute_all_metrics(y_val.to_numpy(), y_val_challenger_prob)

        version_tag = f"v1_{int(time.time())}"
        self.registry.register_version(challenger_model, version_tag, challenger_metrics, description="Challenger model candidate")

        # 3. Champion vs Challenger Validation Gate
        manifest = self.registry.load_manifest()
        champion_entry = manifest.get("active_champion")

        if champion_entry is None:
            # First run: Automatically promote Challenger to initial Champion
            self.registry.promote_to_champion(version_tag)
            gate_passed = True
            margin = 0.0
            gate_reason = "No active Champion found. Initial Challenger automatically promoted."
        else:
            champion_pr_auc = champion_entry["metrics"]["pr_auc"]
            challenger_pr_auc = challenger_metrics["pr_auc"]

            relative_gain = (challenger_pr_auc - champion_pr_auc) / max(champion_pr_auc, 1e-6)
            margin = float(np.round(relative_gain, 4))

            if relative_gain >= self.relative_margin_threshold:
                gate_passed = True
                self.registry.promote_to_champion(version_tag)
                gate_reason = f"Challenger outperformed Champion by {relative_gain:.2%} (Threshold >= {self.relative_margin_threshold:.1%})."
            else:
                gate_passed = False
                gate_reason = f"Challenger failed gate (Gain: {relative_gain:.2%} vs Required: {self.relative_margin_threshold:.1%}). Champion retained."

        logger.info(f"Champion vs Challenger Gate Result: Passed={gate_passed} | Reason: {gate_reason}")

        return {
            "version_tag": version_tag,
            "gate_passed": gate_passed,
            "relative_gain_pct": margin * 100.0,
            "challenger_pr_auc": challenger_metrics["pr_auc"],
            "gate_reason": gate_reason,
            "active_champion_version": self.registry.load_manifest().get("active_champion", {}).get("version_tag"),
        }
