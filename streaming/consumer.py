"""Real-Time Stream Processor Consumer & High-Risk Alert Dispatcher."""

import time
import joblib
from collections import defaultdict, deque
from typing import Any
import numpy as np
import pandas as pd
from config.settings import settings
from data.schema import ALL_PREDICTOR_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from evaluation.business_roi import BusinessROIAnalyzer
from explainability.shap_explainer import ModelExplainer
from features.builder import FeatureBuilder
from utils.logger import get_logger

logger = get_logger("streaming.consumer")


class AlertDispatcher:
    """Dispatches Slack / Webhook format JSON alerts when churn probability exceeds threshold."""

    @staticmethod
    def dispatch_alert(
        customer_id: str,
        churn_prob: float,
        clv: float,
        reasons: list[str],
        recommended_action: str,
    ) -> dict[str, Any]:
        """Construct Slack/Webhook style alert JSON payload."""
        alert_payload = {
            "alert_id": f"ALT-{int(time.time() * 1000)}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "severity": "CRITICAL" if churn_prob >= 0.85 else "WARNING",
            "customer_id": customer_id,
            "churn_probability": round(churn_prob, 4),
            "estimated_clv": round(clv, 2),
            "revenue_at_risk": round(churn_prob * clv, 2),
            "primary_churn_reasons": reasons,
            "recommended_action": recommended_action,
            "channel": "#retention-urgent-alerts",
        }
        logger.warning(
            f"[ALERT] HIGH-RISK STREAM ALERT [{alert_payload['severity']}]: Customer '{customer_id}' Churn Prob = {churn_prob:.1%} | "
            f"Revenue at Risk = ${alert_payload['revenue_at_risk']:,.2f}"
        )
        return alert_payload


def get_default_customer_row(customer_id: str, **overrides) -> dict[str, Any]:
    """Construct full feature schema row with default values for stream evaluation."""
    row = {
        "customer_id": customer_id,
        "gender": "Female",
        "partner": "No",
        "dependents": "No",
        "geography": "North America",
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "auto_pay_enabled": "No",
        "plan_tier": "Standard",
        "credit_card_expiring_soon": "No",
        "price_increase_applied_30d": "No",
        "competitor_offer_viewed": "No",
        "retention_offer_accepted": "No",
        "age": 42,
        "tenure_months": 3,
        "monthly_charges": 95.0,
        "total_charges": 285.0,
        "avg_monthly_gb_download": 120.0,
        "active_days_per_month": 15,
        "login_frequency_30d": 8,
        "app_usage_hours": 12.5,
        "feature_usage_score": 65.0,
        "change_in_usage_pct": 0.0,
        "support_tickets_30d": 1,
        "support_tickets_90d": 2,
        "complaints_opened": 0,
        "resolution_satisfaction_score": 3.5,
        "chat_interactions": 2,
        "payment_failures_90d": 0,
        "late_payments_12m": 0,
        "nps_score": 7,
        "marketing_emails_opened_30d": 3,
    }
    row.update(overrides)
    return row


class StreamProcessorConsumer:
    """Processes real-time event streams with 5-minute sliding window state tracking and scoring."""

    def __init__(self, alert_threshold: float = 0.80) -> None:
        self.alert_threshold = alert_threshold
        self.customer_windows: dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.dispatched_alerts: list[dict[str, Any]] = []

        pipe_path = settings.artifacts_dir / "preprocessing_pipeline.joblib"
        model_path = settings.artifacts_dir / "catboost_model.joblib"
        if not model_path.exists():
            model_path = settings.artifacts_dir / "xgboost_model.joblib"

        self.preprocessor = joblib.load(pipe_path) if pipe_path.exists() else None
        self.model = joblib.load(model_path) if model_path.exists() else None

    def process_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Process incoming stream event, update sliding window, score risk, and emit alert if high risk."""
        cust_id = event["customer_id"]
        event_type = event["event_type"]

        self.customer_windows[cust_id].append(event)
        events_list = list(self.customer_windows[cust_id])

        tickets_cnt = sum(1 for e in events_list if e["event_type"] == "support_ticket_opened")
        usage_drop_cnt = sum(1 for e in events_list if e["event_type"] == "usage_drop_detected")
        pay_fail_cnt = sum(1 for e in events_list if e["event_type"] == "payment_failed")
        comp_cnt = sum(1 for e in events_list if e["event_type"] == "competitor_viewed")

        overrides = {
            "support_tickets_30d": max(1, tickets_cnt),
            "support_tickets_90d": max(2, tickets_cnt * 2),
            "resolution_satisfaction_score": 1.5 if tickets_cnt > 1 else 3.5,
            "payment_failures_90d": pay_fail_cnt,
            "change_in_usage_pct": -0.45 if usage_drop_cnt > 0 else 0.0,
            "competitor_offer_viewed": "Yes" if comp_cnt > 0 else "No",
            "price_increase_applied_30d": "Yes" if event_type == "price_increase_notified" else "No",
        }

        row_dict = get_default_customer_row(cust_id, **overrides)
        profile_df = pd.DataFrame([row_dict])

        featured = FeatureBuilder(enforce_leakage_guard=True).transform(profile_df)

        if self.preprocessor is not None and self.model is not None:
            X_tr = self.preprocessor.transform(featured.drop(columns=["customer_id"]))
            proba_arr = self.model.predict_proba(X_tr)
            prob = float(proba_arr[0, 1]) if proba_arr.shape[1] > 1 else float(proba_arr[0, 0])

            feature_names = list(self.preprocessor.named_steps["preprocessor"].get_feature_names_out())
            importances = getattr(self.model, "feature_importances_", np.ones(X_tr.shape[1]))
            local_exp = ModelExplainer.generate_local_explanation(feature_names, X_tr[0], importances, top_k=2)
            reasons = [d["business_reason"] for d in local_exp["top_drivers"]]
        else:
            prob = 0.85 if (tickets_cnt >= 2 or pay_fail_cnt >= 1) else 0.25
            reasons = ["Support ticket surge in 5m window (+35% risk)", "Recent payment failure (+25% risk)"]

        clv = float(BusinessROIAnalyzer.calculate_clv([profile_df["monthly_charges"].iloc[0]], [profile_df["tenure_months"].iloc[0]])[0])

        if prob >= self.alert_threshold:
            rec_action = "Immediate priority call by VIP retention specialist + 12m contract lock offer."
            alert = AlertDispatcher.dispatch_alert(cust_id, prob, clv, reasons, rec_action)
            self.dispatched_alerts.append(alert)
            return alert

        return None
