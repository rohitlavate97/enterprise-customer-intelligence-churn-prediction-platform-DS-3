"""Automated Data Quality & Profiling Reporter."""

import json
from pathlib import Path
import pandas as pd
from data.schema import CATEGORICAL_FEATURES, LEAKAGE_FIELDS, NUMERICAL_FEATURES, TARGET_COL
from utils.logger import get_logger

logger = get_logger("data.quality_report")


class DataQualityReporter:
    """Computes comprehensive data quality statistics, missing values, duplicates, and anomaly alerts."""

    @staticmethod
    def generate_report(df: pd.DataFrame) -> dict:
        """Analyze DataFrame and return a structured quality report dictionary."""
        n_rows, n_cols = df.shape
        duplicates = int(df.duplicated().sum())

        missing_counts = df.isna().sum().to_dict()
        missing_pcts = (df.isna().mean() * 100).round(2).to_dict()

        col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

        num_stats = {}
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                series = df[col]
                num_stats[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "p25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "p75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                }

        cat_stats = {}
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                cat_stats[col] = df[col].value_counts().to_dict()

        target_stats = {}
        if TARGET_COL in df.columns:
            target_stats = {
                "total_count": len(df),
                "churn_count": int(df[TARGET_COL].sum()),
                "churn_rate": float(df[TARGET_COL].mean()),
            }

        # Check for potential anomalies or data quality issues
        alerts = []
        if duplicates > 0:
            alerts.append(f"Found {duplicates} duplicate customer rows.")

        for col, missing in missing_counts.items():
            if col not in LEAKAGE_FIELDS and missing > 0:
                alerts.append(f"Predictor column '{col}' has {missing} ({missing_pcts[col]}%) missing values.")

        report = {
            "summary": {
                "total_rows": n_rows,
                "total_columns": n_cols,
                "duplicate_rows": duplicates,
                "alert_count": len(alerts),
            },
            "alerts": alerts,
            "missing_values": {
                "counts": missing_counts,
                "percentages": missing_pcts,
            },
            "target_distribution": target_stats,
            "numerical_summary": num_stats,
            "categorical_summary": cat_stats,
            "column_dtypes": col_types,
        }

        return report

    @classmethod
    def save_report(cls, df: pd.DataFrame, output_path: Path) -> Path:
        """Generate and save report JSON to disk."""
        report = cls.generate_report(df)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved Data Quality Report to {output_path}")
        return output_path
