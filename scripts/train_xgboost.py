"""CLI script to train, tune, evaluate, and explain XGBoost churn prediction model."""

import argparse
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import settings
from data.cleaner import DataCleaner
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.metrics import ModelEvaluator
from features.builder import FeatureBuilder
from features.pipeline import PreprocessingPipelineBuilder, save_preprocessing_pipeline
from models.xgboost_suite import XGBoostTrainer
from utils.logger import get_logger

logger = get_logger("scripts.train_xgboost")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost model with SHAP explanations.")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    raw_path = settings.raw_data_dir / "customer_churn_dataset.csv"
    if not raw_path.exists():
        logger.error(f"Raw dataset not found at {raw_path}. Run generate_data script first!")
        return

    logger.info(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)

    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_clean)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    # Train / Val / Test Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.seed, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=args.seed, stratify=y_train_full
    )

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pipe_builder = PreprocessingPipelineBuilder(
        scaler_type="robust",
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    preprocessor = pipe_builder.build_pipeline()

    # Fit preprocessor ONLY on X_tr
    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Save fitted preprocessing pipeline artifact
    save_preprocessing_pipeline(preprocessor)

    trainer = XGBoostTrainer(seed=args.seed)

    best_params = None
    if args.tune:
        best_params = trainer.tune_hyperparameters(X_tr_trans, y_tr.to_numpy(), cv_folds=3, n_iter=6)

    # Calculate class weight ratio
    scale_pos = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))
    if best_params is None:
        best_params = {"scale_pos_weight": scale_pos}

    model = trainer.train_with_early_stopping(
        X_tr_trans,
        y_tr.to_numpy(),
        X_val_trans,
        y_val.to_numpy(),
        params=best_params,
        early_stopping_rounds=25,
    )

    # Evaluate on held-out Test set
    y_test_proba = model.predict_proba(X_test_trans)[:, 1]
    metrics = ModelEvaluator.compute_all_metrics(y_test.to_numpy(), y_test_proba)

    logger.info(
        f"\n==================== XGBOOST TEST EVALUATION ====================\n"
        f"PR AUC:     {metrics['pr_auc']:.4f}\n"
        f"ROC AUC:    {metrics['roc_auc']:.4f}\n"
        f"F1 Score:   {metrics['f1_score']:.4f}\n"
        f"Precision:  {metrics['precision']:.4f}\n"
        f"Recall:     {metrics['recall']:.4f}\n"
        f"Log Loss:   {metrics['log_loss']:.4f}\n"
        f"Brier Score:{metrics['brier_score']:.4f}\n"
    )

    # Save model artifact
    xgb_artifact_path = settings.artifacts_dir / "xgboost_model.joblib"
    joblib.dump(model, xgb_artifact_path)
    logger.info(f"Saved XGBoost model artifact to {xgb_artifact_path}")

    # Compute SHAP
    feature_names = list(preprocessor.named_steps["preprocessor"].get_feature_names_out())
    shap_results = trainer.compute_shap_explanations(model, X_test_trans[:200], feature_names=feature_names)
    if shap_results["shap_dataframe"] is not None:
        logger.info(f"\nTop 5 Important Features via SHAP:\n{shap_results['shap_dataframe'].head(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
