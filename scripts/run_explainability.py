"""CLI script to execute model explainability suite, local customer traces, and segment fairness audit."""

import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import settings
from data.cleaner import DataCleaner
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from explainability.pdp_analysis import PartialDependenceAnalyzer
from explainability.segment_fairness import SegmentFairnessAuditor
from explainability.shap_explainer import ModelExplainer
from features.builder import FeatureBuilder
from utils.logger import get_logger

logger = get_logger("scripts.run_explainability")


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

    logger.info("Loading preprocessing pipeline and model artifact...")
    preprocessor = joblib.load(pipe_path)
    model = joblib.load(model_path)

    logger.info("Loading raw customer dataset for explainability and fairness auditing...")
    df_raw = pd.read_csv(raw_path)
    df_clean = DataCleaner().clean(df_raw)
    df_featured = FeatureBuilder(enforce_leakage_guard=True).transform(df_clean)

    X = df_featured.drop(columns=[TARGET_COL, "customer_id"])
    y = df_featured[TARGET_COL]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_test_trans = preprocessor.transform(X_test)
    y_test_proba = model.predict_proba(X_test_trans)[:, 1]

    df_test_raw = df_clean.iloc[X_test.index].copy()
    df_test_raw["churn_probability"] = y_test_proba
    df_test_raw["churn_prediction"] = (y_test_proba >= 0.50).astype(int)

    # 1. Segment Fairness Audit
    fairness_results = SegmentFairnessAuditor.audit_segment_fairness(
        df_test_raw,
        segment_columns=["tenure_band", "contract_type", "plan_tier", "geography"],
    )

    # 2. Local Explanation Sample for High Risk Customer
    high_risk_idx = int(np.argmax(y_test_proba))
    feature_names = list(preprocessor.named_steps["preprocessor"].get_feature_names_out())

    importances = getattr(model, "feature_importances_", np.ones(X_test_trans.shape[1]))
    sample_row = X_test_trans[high_risk_idx]
    local_exp = ModelExplainer.generate_local_explanation(
        feature_names, sample_row, importances, top_k=4
    )

    sample_cust_id = str(df_test_raw.iloc[high_risk_idx]["customer_id"])
    sample_cust_prob = float(y_test_proba[high_risk_idx])

    report_payload = {
        "model_artifact": str(model_path.name),
        "fairness_audit": fairness_results,
        "sample_customer_explanation": {
            "customer_id": sample_cust_id,
            "churn_probability": sample_cust_prob,
            "narrative": local_exp["summary_narrative"],
            "top_drivers": local_exp["top_drivers"],
        },
    }

    report_path = settings.artifacts_dir / "explainability_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    logger.info(f"\n==================== EXPLAINABILITY & FAIRNESS AUDIT ====================")
    logger.info(f"Audit Passed: {fairness_results['audit_passed']} (Disparity Alerts: {len(fairness_results['disparity_alerts'])})")
    logger.info(f"High-Risk Customer ({sample_cust_id}, Prob={sample_cust_prob:.1%}): {local_exp['summary_narrative']}")
    logger.info(f"Saved Explainability Report to {report_path}")


if __name__ == "__main__":
    main()
