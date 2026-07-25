"""CLI script to run full model comparison benchmark across Baselines, XGBoost, LightGBM, and CatBoost."""

import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import settings
from data.cleaner import DataCleaner
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.comparison_report import ModelComparisonReporter
from evaluation.metrics import ModelEvaluator
from features.builder import FeatureBuilder
from features.pipeline import PreprocessingPipelineBuilder
from models.baselines import get_baseline_models
from models.catboost_suite import CatBoostTrainer
from models.lightgbm_suite import LightGBMTrainer
from models.xgboost_suite import XGBoostTrainer
from utils.logger import get_logger

logger = get_logger("scripts.run_model_comparison")


def main() -> None:
    raw_path = settings.raw_data_dir / "customer_churn_dataset.csv"
    if not raw_path.exists():
        logger.error(f"Raw dataset not found at {raw_path}. Run generate_data script first!")
        return

    logger.info("Loading and cleaning dataset for full model comparison benchmark...")
    df_raw = pd.read_csv(raw_path)
    df_clean = DataCleaner().clean(df_raw)
    df_featured = FeatureBuilder(enforce_leakage_guard=True).transform(df_clean)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.15, random_state=42, stratify=y_tr_full
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

    evaluations = []
    scale_pos = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))

    # 1. CatBoost
    cb_trainer = CatBoostTrainer(seed=42)
    t0 = time.perf_counter()
    cb_model = cb_trainer.train_with_early_stopping(
        X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy(), params={"scale_pos_weight": scale_pos}
    )
    cb_t = time.perf_counter() - t0
    t_inf = time.perf_counter()
    cb_proba = cb_model.predict_proba(X_test_trans)[:, 1]
    cb_latency = (time.perf_counter() - t_inf) / len(X_test) * 1000
    cb_metrics = ModelEvaluator.compute_all_metrics(y_test.to_numpy(), cb_proba)
    cb_metrics["model_name"] = "CatBoost"
    cb_metrics["training_time_sec"] = cb_t
    cb_metrics["inference_latency_ms"] = cb_latency
    evaluations.append(cb_metrics)

    # 2. LightGBM
    lgb_trainer = LightGBMTrainer(seed=42)
    t0 = time.perf_counter()
    lgb_model = lgb_trainer.train_with_early_stopping(
        X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy(), params={"scale_pos_weight": scale_pos}
    )
    lgb_t = time.perf_counter() - t0
    t_inf = time.perf_counter()
    lgb_proba = lgb_model.predict_proba(X_test_trans)[:, 1]
    lgb_latency = (time.perf_counter() - t_inf) / len(X_test) * 1000
    lgb_metrics = ModelEvaluator.compute_all_metrics(y_test.to_numpy(), lgb_proba)
    lgb_metrics["model_name"] = "LightGBM"
    lgb_metrics["training_time_sec"] = lgb_t
    lgb_metrics["inference_latency_ms"] = lgb_latency
    evaluations.append(lgb_metrics)

    # 3. XGBoost
    xgb_trainer = XGBoostTrainer(seed=42)
    t0 = time.perf_counter()
    xgb_model = xgb_trainer.train_with_early_stopping(
        X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy(), params={"scale_pos_weight": scale_pos}
    )
    xgb_t = time.perf_counter() - t0
    t_inf = time.perf_counter()
    xgb_proba = xgb_model.predict_proba(X_test_trans)[:, 1]
    xgb_latency = (time.perf_counter() - t_inf) / len(X_test) * 1000
    xgb_metrics = ModelEvaluator.compute_all_metrics(y_test.to_numpy(), xgb_proba)
    xgb_metrics["model_name"] = "XGBoost"
    xgb_metrics["training_time_sec"] = xgb_t
    xgb_metrics["inference_latency_ms"] = xgb_latency
    evaluations.append(xgb_metrics)

    # 4. Baselines
    baselines = get_baseline_models(seed=42)
    X_tr_full_trans = preprocessor.transform(X_tr_full)
    for b_name, b_est in baselines.items():
        if b_name in ["Logistic Regression", "Random Forest", "Naive Bayes"]:
            t0 = time.perf_counter()
            b_est.fit(X_tr_full_trans, y_tr_full.to_numpy())
            b_t = time.perf_counter() - t0
            t_inf = time.perf_counter()
            b_proba = b_est.predict_proba(X_test_trans)[:, 1]
            b_latency = (time.perf_counter() - t_inf) / len(X_test) * 1000
            b_metrics = ModelEvaluator.compute_all_metrics(y_test.to_numpy(), b_proba)
            b_metrics["model_name"] = b_name
            b_metrics["training_time_sec"] = b_t
            b_metrics["inference_latency_ms"] = b_latency
            evaluations.append(b_metrics)

    reporter = ModelComparisonReporter(primary_metric="pr_auc")
    df_matrix = reporter.generate_comparison_matrix(evaluations)
    md_path, json_path = reporter.save_reports(df_matrix)

    logger.info(f"\n==================== FULL MODEL COMPARISON MATRIX ====================\n{df_matrix.to_string(index=False)}")


if __name__ == "__main__":
    main()
