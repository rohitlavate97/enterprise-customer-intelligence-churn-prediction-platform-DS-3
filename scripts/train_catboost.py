"""CLI script to train, tune, evaluate CatBoost and benchmark against XGBoost & LightGBM."""

import argparse
import time
import joblib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import settings
from data.cleaner import DataCleaner
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.metrics import ModelEvaluator
from features.builder import FeatureBuilder
from features.pipeline import PreprocessingPipelineBuilder
from models.catboost_suite import CatBoostTrainer
from models.lightgbm_suite import LightGBMTrainer
from models.xgboost_suite import XGBoostTrainer
from utils.logger import get_logger

logger = get_logger("scripts.train_catboost")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CatBoost and benchmark against XGBoost & LightGBM.")
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

    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    scale_pos = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))

    # 1. Train CatBoost
    cb_trainer = CatBoostTrainer(seed=args.seed)
    cb_params = None
    if args.tune:
        cb_params = cb_trainer.tune_hyperparameters(X_tr_trans, y_tr.to_numpy(), cv_folds=3, n_iter=6)

    if cb_params is None:
        cb_params = {"scale_pos_weight": scale_pos}

    start_cb_t = time.perf_counter()
    cb_model = cb_trainer.train_with_early_stopping(
        X_tr_trans,
        y_tr.to_numpy(),
        X_val_trans,
        y_val.to_numpy(),
        params=cb_params,
        early_stopping_rounds=25,
    )
    cb_train_time = time.perf_counter() - start_cb_t

    start_cb_inf = time.perf_counter()
    y_test_cb_proba = cb_model.predict_proba(X_test_trans)[:, 1]
    cb_inf_time = (time.perf_counter() - start_cb_inf) / len(X_test) * 1000

    cb_metrics = ModelEvaluator.compute_all_metrics(y_test.to_numpy(), y_test_cb_proba)

    # 2. Train LightGBM Benchmark
    lgb_trainer = LightGBMTrainer(seed=args.seed)
    start_lgb_t = time.perf_counter()
    lgb_model = lgb_trainer.train_with_early_stopping(
        X_tr_trans,
        y_tr.to_numpy(),
        X_val_trans,
        y_val.to_numpy(),
        params={"scale_pos_weight": scale_pos},
        early_stopping_rounds=25,
    )
    lgb_train_time = time.perf_counter() - start_lgb_t

    start_lgb_inf = time.perf_counter()
    y_test_lgb_proba = lgb_model.predict_proba(X_test_trans)[:, 1]
    lgb_inf_time = (time.perf_counter() - start_lgb_inf) / len(X_test) * 1000

    lgb_metrics = ModelEvaluator.compute_all_metrics(y_test.to_numpy(), y_test_lgb_proba)

    # 3. Train XGBoost Benchmark
    xgb_trainer = XGBoostTrainer(seed=args.seed)
    start_xgb_t = time.perf_counter()
    xgb_model = xgb_trainer.train_with_early_stopping(
        X_tr_trans,
        y_tr.to_numpy(),
        X_val_trans,
        y_val.to_numpy(),
        params={"scale_pos_weight": scale_pos},
        early_stopping_rounds=25,
    )
    xgb_train_time = time.perf_counter() - start_xgb_t

    start_xgb_inf = time.perf_counter()
    y_test_xgb_proba = xgb_model.predict_proba(X_test_trans)[:, 1]
    xgb_inf_time = (time.perf_counter() - start_xgb_inf) / len(X_test) * 1000

    xgb_metrics = ModelEvaluator.compute_all_metrics(y_test.to_numpy(), y_test_xgb_proba)

    # 4. Print Benchmark Comparison Table
    benchmark_df = pd.DataFrame(
        [
            {
                "Model": "CatBoost",
                "PR AUC": cb_metrics["pr_auc"],
                "ROC AUC": cb_metrics["roc_auc"],
                "F1 Score": cb_metrics["f1_score"],
                "Precision": cb_metrics["precision"],
                "Recall": cb_metrics["recall"],
                "Log Loss": cb_metrics["log_loss"],
                "Train Time (s)": cb_train_time,
                "Inference Latency (ms/sample)": cb_inf_time,
            },
            {
                "Model": "LightGBM",
                "PR AUC": lgb_metrics["pr_auc"],
                "ROC AUC": lgb_metrics["roc_auc"],
                "F1 Score": lgb_metrics["f1_score"],
                "Precision": lgb_metrics["precision"],
                "Recall": lgb_metrics["recall"],
                "Log Loss": lgb_metrics["log_loss"],
                "Train Time (s)": lgb_train_time,
                "Inference Latency (ms/sample)": lgb_inf_time,
            },
            {
                "Model": "XGBoost",
                "PR AUC": xgb_metrics["pr_auc"],
                "ROC AUC": xgb_metrics["roc_auc"],
                "F1 Score": xgb_metrics["f1_score"],
                "Precision": xgb_metrics["precision"],
                "Recall": xgb_metrics["recall"],
                "Log Loss": xgb_metrics["log_loss"],
                "Train Time (s)": xgb_train_time,
                "Inference Latency (ms/sample)": xgb_inf_time,
            },
        ]
    )

    logger.info(f"\n==================== GRADIENT BOOSTING TRIPLE BENCHMARK ====================\n{benchmark_df.to_string(index=False)}")

    # Save CatBoost artifact
    cb_artifact_path = settings.artifacts_dir / "catboost_model.joblib"
    joblib.dump(cb_model, cb_artifact_path)
    logger.info(f"Saved CatBoost model artifact to {cb_artifact_path}")


if __name__ == "__main__":
    main()
