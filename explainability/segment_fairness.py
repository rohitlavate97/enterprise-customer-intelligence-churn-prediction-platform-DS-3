"""Segment Fairness and Error Rate Disparity Auditor."""

from typing import Any
import numpy as np
import pandas as pd
from evaluation.metrics import ModelEvaluator
from utils.logger import get_logger

logger = get_logger("explainability.segment_fairness")


class SegmentFairnessAuditor:
    """Audits model error rates, precision, recall, and PR AUC across customer demographic and contract segments."""

    @staticmethod
    def add_tenure_band_column(df: pd.DataFrame) -> pd.DataFrame:
        """Add tenure_band categorical column to dataframe if tenure_months exists."""
        df_out = df.copy()
        if "tenure_months" in df_out.columns:
            bins = [-1, 6, 12, 24, 72]
            labels = ["0-6 Months", "6-12 Months", "12-24 Months", "24+ Months"]
            df_out["tenure_band"] = pd.cut(df_out["tenure_months"], bins=bins, labels=labels)
        return df_out

    @classmethod
    def audit_segment_fairness(
        cls,
        df_raw_with_predictions: pd.DataFrame,
        segment_columns: list[str] | None = None,
        target_col: str = "churn_label",
        prob_col: str = "churn_probability",
    ) -> dict[str, Any]:
        """Audit error rates and metrics per segment and flag performance disparities."""
        logger.info("Running Segment Fairness & Error Disparity Audit...")
        df = cls.add_tenure_band_column(df_raw_with_predictions)

        segments_to_check = segment_columns or ["tenure_band", "contract_type", "plan_tier", "geography"]
        valid_segments = [col for col in segments_to_check if col in df.columns]

        overall_metrics = ModelEvaluator.compute_all_metrics(df[target_col].to_numpy(), df[prob_col].to_numpy())
        overall_error_rate = 1.0 - overall_metrics["f1_score"]

        segment_results = {}
        disparity_alerts = []

        for seg_col in valid_segments:
            seg_group = {}
            for seg_val, group_df in df.groupby(seg_col, observed=True):
                if len(group_df) < 20:
                    continue  # Skip tiny segments

                y_true = group_df[target_col].to_numpy()
                y_prob = group_df[prob_col].to_numpy()

                g_metrics = ModelEvaluator.compute_all_metrics(y_true, y_prob)
                g_error = 1.0 - g_metrics["f1_score"]

                seg_group[str(seg_val)] = {
                    "count": len(group_df),
                    "actual_churn_rate": float(y_true.mean()),
                    "pr_auc": g_metrics["pr_auc"],
                    "roc_auc": g_metrics["roc_auc"],
                    "f1_score": g_metrics["f1_score"],
                    "precision": g_metrics["precision"],
                    "recall": g_metrics["recall"],
                    "error_rate": float(g_error),
                }

                # Check if segment error rate exceeds 1.5x aggregate error rate
                if overall_error_rate > 0 and g_error > 1.5 * overall_error_rate:
                    alert_msg = (
                        f"SEGMENT DISPARITY ALERT: Segment '{seg_col}={seg_val}' error rate ({g_error:.2%}) "
                        f"exceeds 1.5x aggregate error rate ({overall_error_rate:.2%})."
                    )
                    disparity_alerts.append(alert_msg)
                    logger.warning(alert_msg)

            segment_results[seg_col] = seg_group

        return {
            "overall_metrics": overall_metrics,
            "segment_breakdown": segment_results,
            "disparity_alerts": disparity_alerts,
            "audit_passed": len(disparity_alerts) == 0,
        }
