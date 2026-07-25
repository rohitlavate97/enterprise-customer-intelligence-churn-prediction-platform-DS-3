"""Partial Dependence Plot (PDP) Analysis Module."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from utils.logger import get_logger

logger = get_logger("explainability.pdp_analysis")


class PartialDependenceAnalyzer:
    """Computes Partial Dependence grid values across numerical customer features."""

    @staticmethod
    def compute_1d_partial_dependence(
        estimator: Any,
        X_trans: np.ndarray,
        feature_index: int,
        feature_name: str,
        grid_resolution: int = 20,
    ) -> dict[str, Any]:
        """Compute 1D partial dependence grid values and average response for a feature."""
        logger.debug(f"Computing partial dependence for feature '{feature_name}' (index {feature_index})...")

        pdp_result = partial_dependence(
            estimator,
            X_trans,
            features=[feature_index],
            grid_resolution=grid_resolution,
            kind="average",
        )

        grid_values = pdp_result["grid_values"][0]
        avg_predictions = pdp_result["average"][0]

        return {
            "feature_name": feature_name,
            "feature_index": feature_index,
            "grid_values": [float(v) for v in grid_values],
            "average_predicted_probability": [float(p) for p in avg_predictions],
        }
