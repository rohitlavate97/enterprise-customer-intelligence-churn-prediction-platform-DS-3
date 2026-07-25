"""Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) Feature Drift Detector."""

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from utils.logger import get_logger

logger = get_logger("monitoring.drift_detector")


class DriftDetector:
    """Calculates PSI and KS statistics to detect feature distribution drift between reference and serving data."""

    @staticmethod
    def calculate_psi(
        reference: np.ndarray | pd.Series,
        current: np.ndarray | pd.Series,
        num_bins: int = 10,
        epsilon: float = 1e-4,
    ) -> float:
        """Calculate Population Stability Index (PSI) between reference and current distribution arrays."""
        ref_arr = np.asarray(reference, dtype=float)
        cur_arr = np.asarray(current, dtype=float)

        # Drop NaNs
        ref_arr = ref_arr[~np.isnan(ref_arr)]
        cur_arr = cur_arr[~np.isnan(cur_arr)]

        if len(ref_arr) == 0 or len(cur_arr) == 0:
            return 0.0

        # Define quantile bin edges based on reference distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref_arr, percentiles)
        bins = np.unique(bins)  # Handle identical percentiles

        if len(bins) < 2:
            return 0.0

        ref_counts, _ = np.histogram(ref_arr, bins=bins)
        cur_counts, _ = np.histogram(cur_arr, bins=bins)

        ref_pct = ref_counts / len(ref_arr)
        cur_pct = cur_counts / len(cur_arr)

        # Apply epsilon smoothing to prevent div by zero / log of zero
        ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
        cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

        psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(np.round(psi_val, 4))

    @staticmethod
    def calculate_ks_test(
        reference: np.ndarray | pd.Series,
        current: np.ndarray | pd.Series,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Perform 2-sample Kolmogorov-Smirnov test for distribution drift."""
        ref_arr = np.asarray(reference, dtype=float)
        cur_arr = np.asarray(current, dtype=float)

        ref_arr = ref_arr[~np.isnan(ref_arr)]
        cur_arr = cur_arr[~np.isnan(cur_arr)]

        if len(ref_arr) == 0 or len(cur_arr) == 0:
            return {"ks_statistic": 0.0, "p_value": 1.0, "drift_detected": False}

        ks_res = stats.ks_2samp(ref_arr, cur_arr)
        drift_detected = bool(ks_res.pvalue < alpha)

        return {
            "ks_statistic": float(np.round(ks_res.statistic, 4)),
            "p_value": float(np.round(ks_res.pvalue, 6)),
            "drift_detected": drift_detected,
        }

    @classmethod
    def audit_dataset_drift(
        cls,
        df_reference: pd.DataFrame,
        df_current: pd.DataFrame,
        numerical_features: list[str] | None = None,
        psi_threshold: float = 0.25,
    ) -> dict[str, Any]:
        """Audit dataset drift across all numerical features and categorize risk levels."""
        logger.info(f"Auditing dataset drift across {len(df_reference)} reference and {len(df_current)} current records...")

        num_cols = numerical_features or list(df_reference.select_dtypes(include=[np.number]).columns)
        valid_cols = [c for c in num_cols if c in df_reference.columns and c in df_current.columns]

        feature_reports = {}
        high_drift_features = []

        for col in valid_cols:
            psi = cls.calculate_psi(df_reference[col], df_current[col])
            ks_res = cls.calculate_ks_test(df_reference[col], df_current[col])

            if psi >= 0.25:
                status = "High Drift (Retrain Required)"
                high_drift_features.append(col)
            elif psi >= 0.10:
                status = "Moderate Drift (Monitor)"
            else:
                status = "No Drift"

            feature_reports[col] = {
                "psi": psi,
                "ks_statistic": ks_res["ks_statistic"],
                "p_value": ks_res["p_value"],
                "ks_drift_detected": ks_res["drift_detected"],
                "drift_status": status,
            }

        dataset_drift_detected = len(high_drift_features) > 0

        if dataset_drift_detected:
            logger.warning(f"FEATURE DRIFT WARNING: High drift detected in {len(high_drift_features)} features: {high_drift_features}")
        else:
            logger.info("Dataset Drift Audit Complete. No significant distribution drift detected.")

        return {
            "total_features_audited": len(valid_cols),
            "high_drift_feature_count": len(high_drift_features),
            "high_drift_features": high_drift_features,
            "dataset_drift_detected": dataset_drift_detected,
            "feature_reports": feature_reports,
        }
