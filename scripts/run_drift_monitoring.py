"""CLI script to execute feature distribution drift, KS tests, concept drift, and retraining trigger audit."""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import settings
from data.cleaner import DataCleaner
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.metrics import ModelEvaluator
from features.builder import FeatureBuilder
from monitoring.concept_drift import ConceptDriftMonitor
from monitoring.drift_detector import DriftDetector
from utils.logger import get_logger

logger = get_logger("scripts.run_drift_monitoring")


def main() -> None:
    raw_path = settings.raw_data_dir / "customer_churn_dataset.csv"
    if not raw_path.exists():
        logger.error(f"Raw dataset not found at {raw_path}. Run generate_data script first!")
        return

    pipe_path = settings.artifacts_dir / "preprocessing_pipeline.joblib"
    model_path = settings.artifacts_dir / "catboost_model.joblib"
    if not model_path.exists():
        model_path = settings.artifacts_dir / "xgboost_model.joblib"

    if not pipe_path.exists() or not model_path.exists():
        logger.error("Fitted pipeline or model artifact missing. Run training scripts first!")
        return

    logger.info("Loading dataset for model drift audit...")
    df_raw = pd.read_csv(raw_path)
    df_clean = DataCleaner().clean(df_raw)
    df_featured = FeatureBuilder(enforce_leakage_guard=True).transform(df_clean)

    # Split into Reference baseline (80%) vs Current serving batch (20%)
    df_ref, df_cur = train_test_split(df_featured, test_size=0.20, random_state=42, stratify=df_featured[TARGET_COL])

    # 1. Simulate mild feature drift in current batch (e.g. support ticket increase)
    df_cur_shifted = df_cur.copy()
    df_cur_shifted["support_tickets_30d"] = df_cur_shifted["support_tickets_30d"] * 2.5
    df_cur_shifted["monthly_charges"] = df_cur_shifted["monthly_charges"] * 1.3

    # 2. Audit Feature Distribution Drift (PSI & KS)
    drift_detector = DriftDetector()
    drift_audit = drift_detector.audit_dataset_drift(
        df_ref, df_cur_shifted, numerical_features=NUMERICAL_FEATURES, psi_threshold=0.25
    )

    # 3. Audit Concept Drift & Accuracy Decay
    preprocessor = joblib.load(pipe_path)
    model = joblib.load(model_path)

    X_cur = df_cur_shifted.drop(columns=[TARGET_COL, "customer_id"])
    y_cur = df_cur_shifted[TARGET_COL]

    X_cur_trans = preprocessor.transform(X_cur)
    y_cur_proba = model.predict_proba(X_cur_trans)[:, 1]

    concept_monitor = ConceptDriftMonitor(pr_auc_threshold=0.80, psi_threshold=0.25)
    concept_audit = concept_monitor.evaluate_performance_decay(y_cur.to_numpy(), y_cur_proba, baseline_pr_auc=0.9099)

    # 4. Evaluate Retraining Trigger
    retrain_decision = concept_monitor.check_retraining_trigger(drift_audit, concept_audit)

    report_payload = {
        "model_artifact": str(model_path.name),
        "drift_audit": drift_audit,
        "concept_audit": concept_audit,
        "retraining_decision": retrain_decision,
    }

    report_path = settings.artifacts_dir / "drift_monitoring_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.info(f"\n==================== DRIFT MONITORING SUMMARY ====================")
    logger.info(f"Audited Features: {drift_audit['total_features_audited']} | High Drift Features (PSI >= 0.25): {drift_audit['high_drift_feature_count']}")
    logger.info(f"Current PR AUC: {concept_audit['current_pr_auc']:.4f} (Baseline: {concept_audit['baseline_pr_auc']:.4f})")
    logger.info(f"Automated Retraining Required: {retrain_decision['retraining_required']}")
    logger.info(f"Saved Drift Monitoring Report JSON to {report_path}")


if __name__ == "__main__":
    main()
