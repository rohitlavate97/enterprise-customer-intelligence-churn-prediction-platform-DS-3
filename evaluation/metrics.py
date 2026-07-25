"""Comprehensive model evaluation metrics and calibration scoring suite."""

from typing import Any
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from utils.logger import get_logger

logger = get_logger("evaluation.metrics")


class ModelEvaluator:
    """Evaluates binary classification predictions, probabilities, and calibration."""

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.50,
    ) -> dict[str, Any]:
        """Compute complete metric suite: PR AUC, ROC AUC, F1, Precision, Recall, Log Loss, Brier Score, and Confusion Matrix."""
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Basic metrics
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # ROC AUC
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))

        # Precision-Recall Curve & PR AUC
        precisions, recalls, _ = precision_recall_curve(y_true, y_pred_proba)
        # Note: precision_recall_curve returns decreasing recalls, auc handles x=recalls
        pr_auc = float(auc(recalls, precisions))

        # Probability quality
        loss = float(log_loss(y_true, y_pred_proba))
        brier = float(brier_score_loss(y_true, y_pred_proba))

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "log_loss": loss,
            "brier_score": brier,
            "threshold_used": threshold,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

        return metrics

    @staticmethod
    def compute_calibration_curve(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> dict[str, list[float]]:
        """Compute calibration curve probabilities and true bin fractions."""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy="uniform")
        return {
            "prob_true": [float(p) for p in prob_true],
            "prob_pred": [float(p) for p in prob_pred],
        }
