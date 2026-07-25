"""CLI script to calculate Customer Lifetime Value (CLV), Revenue at Risk, Retention Campaign ROI, and Call List."""

import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import settings
from data.cleaner import DataCleaner
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COL
from evaluation.business_roi import BusinessROIAnalyzer
from features.builder import FeatureBuilder
from utils.logger import get_logger

logger = get_logger("scripts.run_business_analytics")


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

    logger.info("Loading preprocessing pipeline and model artifact for Business Analytics...")
    preprocessor = joblib.load(pipe_path)
    model = joblib.load(model_path)

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

    df_test_full = df_clean.iloc[X_test.index].copy()
    df_test_full["churn_probability"] = y_test_proba
    df_test_full[TARGET_COL] = y_test.values

    # 1. Run Financial Impact Analysis
    roi_metrics = BusinessROIAnalyzer.analyze_financial_impact(
        df_test_full,
        churn_prob_col="churn_probability",
        probability_threshold=0.50,
        intervention_cost_per_cust=50.0,
        intervention_success_rate=0.25,
    )

    # 2. Generate High-Risk Call List
    call_list_df = BusinessROIAnalyzer.generate_high_risk_call_list(
        df_test_full, churn_prob_col="churn_probability", top_n=50
    )

    # Save artifacts
    report_path = settings.artifacts_dir / "business_analytics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(roi_metrics, f, indent=2)

    call_list_path = settings.artifacts_dir / "high_risk_call_list.csv"
    call_list_df.to_csv(call_list_path, index=False)

    outcomes = roi_metrics["campaign_outcomes"]
    logger.info(f"\n==================== FINANCIAL IMPACT & INTERVENTION ROI ====================")
    logger.info(f"High-Risk Customers: {roi_metrics['high_risk_customer_count']} | Revenue at Risk: ${roi_metrics['total_revenue_at_risk']:,.2f}")
    logger.info(f"Targeted Campaign Cost ($50/cust): ${outcomes['total_campaign_cost']:,.2f}")
    logger.info(f"Projected Saved Revenue (25% conversion): ${outcomes['gross_saved_revenue']:,.2f}")
    logger.info(f"Net Campaign Profit: ${outcomes['net_saved_revenue']:,.2f} (ROI: {outcomes['campaign_roi_pct']:.1f}%)")
    logger.info(f"Saved Business Report JSON to {report_path}")
    logger.info(f"Saved High-Risk Retention Call List CSV to {call_list_path}")


if __name__ == "__main__":
    main()
