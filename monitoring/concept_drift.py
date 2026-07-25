"""Concept Drift Monitoring and Automated Retraining Trigger Engine."""

from typing import Any
import numpy as np

from evaluation.metrics import ModelEvaluator
from monitoring.drift_detector import DriftDetector
from utils.logger import get_logger

logger = get_logger("monitoring.concept_drift")


class ConceptDriftMonitor:
    """Monitors model accuracy decay over ground-truth evaluation windows and evaluates automated retraining criteria."""

    def __init__(
        self,
        pr_auc_threshold: float = 0.80,
        psi_threshold: float = 0.25,
    ) -> None:
        self.pr_auc_threshold = pr_auc_threshold
        self.psi_threshold = psi_threshold

    def evaluate_performance_decay(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        baseline_pr_auc: float = 0.90,
    ) -> dict[str, Any]:
        """Evaluate model performance metrics on labeled ground truth window and calculate decay vs baseline."""
        current_metrics = ModelEvaluator.compute_all_metrics(y_true, y_prob)

        cur_pr_auc = current_metrics["pr_auc"]
        performance_decay = baseline_pr_auc - cur_pr_auc
        concept_drift_detected = cur_pr_auc < self.pr_auc_threshold

        if concept_drift_detected:
            logger.warning(
                f"CONCEPT DRIFT DECAY ALERT: Current PR AUC ({cur_pr_auc:.4f}) dropped below threshold ({self.pr_auc_threshold:.4f}) "
                f"[Decay from baseline: -{performance_decay:.4f}]"
            )

        return {
            "baseline_pr_auc": baseline_pr_auc,
            "current_pr_auc": cur_pr_auc,
            "current_roc_auc": current_metrics["roc_auc"],
            "current_f1_score": current_metrics["f1_score"],
            "performance_decay": float(np.round(performance_decay, 4)),
            "concept_drift_detected": concept_drift_detected,
        }

    def check_retraining_trigger(
        self,
        drift_audit_result: dict[str, Any],
        concept_drift_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate retraining trigger criteria (Feature PSI >= 0.25 OR PR AUC < 0.80)."""
        feature_drift = drift_audit_result.get("dataset_drift_detected", False)
        concept_drift = concept_drift_result.get("concept_drift_detected", False)

        retraining_required = feature_drift or concept_drift

        reasons = []
        if feature_drift:
            reasons.append(f"Feature distribution PSI >= {self.psi_threshold} in {drift_audit_result['high_drift_feature_count']} features.")
        if concept_drift:
            reasons.append(f"Model PR AUC ({concept_drift_result['current_pr_auc']:.4f}) fell below threshold ({self.pr_auc_threshold:.4f}).")

        if retraining_required:
            logger.error(
                f"[RETRAIN] AUTOMATED RETRAINING TRIGGERED! Action required for model deployment. Reasons: {'; '.join(reasons)}"
            )
        else:
            logger.info("Retraining Check Passed. Model is operating within expected stability parameters.")

        return {
            "retraining_required": retraining_required,
            "feature_drift_flag": feature_drift,
            "concept_drift_flag": concept_drift,
            "trigger_reasons": reasons,
            "recommended_action": "Trigger Automated Re-training Pipeline" if retraining_required else "Maintain Current Champion Model",
        }
