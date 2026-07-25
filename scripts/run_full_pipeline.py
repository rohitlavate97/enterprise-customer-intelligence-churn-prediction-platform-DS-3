"""Master Workflow Orchestrator Script executing all platform stages sequentially."""

import time
from config.settings import settings
from data.cleaner import DataCleaner
from data.generator import generate_and_save_dataset
from data.quality_report import DataQualityReporter
from evaluation.business_roi import BusinessROIAnalyzer
from evaluation.comparison_report import ModelComparisonReporter
from explainability.segment_fairness import SegmentFairnessAuditor
from explainability.shap_explainer import ModelExplainer
from features.builder import FeatureBuilder
from features.pipeline import PreprocessingPipelineBuilder
from models.catboost_suite import CatBoostTrainer
from models.lightgbm_suite import LightGBMTrainer
from models.xgboost_suite import XGBoostTrainer
from monitoring.concept_drift import ConceptDriftMonitor
from monitoring.drift_detector import DriftDetector
from streaming.consumer import StreamProcessorConsumer
from streaming.producer import CustomerEventProducer
from training.retraining_pipeline import AutomatedRetrainingPipeline
from utils.logger import get_logger

logger = get_logger("scripts.run_full_pipeline")


def run_all_stages() -> bool:
    """Execute complete 11-stage enterprise ML lifecycle end-to-end."""
    start_total_t = time.perf_counter()
    logger.info("==========================================================================")
    logger.info("[START] STARTING ENTERPRISE CHURN PREDICTION PLATFORM — MASTER WORKFLOW")
    logger.info("==========================================================================")

    # STAGE 1: Synthetic Dataset Generation
    logger.info("\n--- STAGE 1: Synthetic Dataset Generation ---")
    raw_df = generate_and_save_dataset(n_samples=5000, seed=42)

    # STAGE 2: Data Cleaning & Quality Audit
    logger.info("\n--- STAGE 2: Data Cleaning & Quality Profiling ---")
    cleaner = DataCleaner()
    clean_df = cleaner.clean(raw_df)
    DataQualityReporter.save_report(clean_df, settings.artifacts_dir / "data_quality_report.json")

    # STAGE 3: Domain Feature Engineering & Preprocessing Pipeline
    logger.info("\n--- STAGE 3: Domain Feature Engineering & Preprocessing Pipeline ---")
    builder = FeatureBuilder(enforce_leakage_guard=True)
    featured_df = builder.transform(clean_df)

    # STAGE 4: Model Training (CatBoost Champion)
    logger.info("\n--- STAGE 4: Model Training & Early Stopping ---")
    from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
    from sklearn.model_selection import train_test_split

    X = featured_df.drop(columns=[TARGET_COL, "customer_id"])
    y = featured_df[TARGET_COL]

    X_tr_full, X_test, y_tr_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr_full, y_tr_full, test_size=0.15, random_state=42, stratify=y_tr_full)

    num_cols = [c for c in NUMERICAL_FEATURES + ["charges_per_tenure", "risk_score_index"] if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    pipe_builder = PreprocessingPipelineBuilder(scaler_type="robust", numerical_features=num_cols, categorical_features=cat_cols)
    preprocessor = pipe_builder.build_pipeline()

    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    from features.pipeline import save_preprocessing_pipeline
    save_preprocessing_pipeline(preprocessor)

    cb_trainer = CatBoostTrainer(seed=42)
    cb_model = cb_trainer.train_with_early_stopping(X_tr_trans, y_tr.to_numpy(), X_val_trans, y_val.to_numpy())

    # STAGE 5: SHAP Explainability & Segment Fairness Audit
    logger.info("\n--- STAGE 5: Explainability & Segment Fairness Audit ---")
    y_test_proba = cb_model.predict_proba(X_test_trans)[:, 1]
    df_test_full = clean_df.iloc[X_test.index].copy()
    df_test_full["churn_probability"] = y_test_proba
    df_test_full[TARGET_COL] = y_test.values

    fairness_res = SegmentFairnessAuditor.audit_segment_fairness(df_test_full)

    # STAGE 6: Business ROI & Retention Call List
    logger.info("\n--- STAGE 6: Business ROI & Retention Call List Generation ---")
    roi_metrics = BusinessROIAnalyzer.analyze_financial_impact(df_test_full)
    call_list_df = BusinessROIAnalyzer.generate_high_risk_call_list(df_test_full, top_n=50)

    # STAGE 7: Real-Time Event Streaming Simulation
    logger.info("\n--- STAGE 7: Real-Time Event Streaming & Alert Simulation ---")
    producer = CustomerEventProducer(seed=42)
    consumer = StreamProcessorConsumer(alert_threshold=0.80)
    for event in producer.stream_events(max_events=15, delay_sec=0.0):
        consumer.process_event(event)

    # STAGE 8: Model Drift Audit (PSI / KS)
    logger.info("\n--- STAGE 8: Model Drift Audit (PSI / KS) ---")
    df_cur_shifted = featured_df.copy()
    df_cur_shifted["support_tickets_30d"] = df_cur_shifted["support_tickets_30d"] * 2.0
    drift_audit = DriftDetector().audit_dataset_drift(featured_df, df_cur_shifted)

    # STAGE 9: Automated Retraining & Validation Gate
    logger.info("\n--- STAGE 9: Automated Retraining & Champion vs Challenger Gate ---")
    retrain_pipeline = AutomatedRetrainingPipeline()
    retrain_res = retrain_pipeline.run_retraining_cycle(n_samples=1000, seed=42)

    total_time = time.perf_counter() - start_total_t
    logger.info("==========================================================================")
    logger.info(f"[SUCCESS] MASTER WORKFLOW COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    logger.info(f"Active Champion: {retrain_res['active_champion_version']}")
    logger.info(f"Projected Retention Campaign Profit: ${roi_metrics['campaign_outcomes']['net_saved_revenue']:,.2f}")
    logger.info("==========================================================================")
    return True


if __name__ == "__main__":
    run_all_stages()
