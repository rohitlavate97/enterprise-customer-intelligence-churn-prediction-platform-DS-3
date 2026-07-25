"""Target Leakage Guard module to explicitly enforce zero leakage in feature matrices."""

import pandas as pd
from data.schema import LEAKAGE_FIELDS
from utils.logger import get_logger

logger = get_logger("features.leakage_guard")


class LeakageGuard:
    """Hard-stop assertion guard preventing target leakage fields from entering modeling pipelines."""

    @staticmethod
    def audit_columns(columns: list[str] | pd.Index) -> list[str]:
        """Check column list for forbidden target leakage fields.

        Returns detected leakage fields, if any.
        """
        col_set = set(columns)
        detected = [field for field in LEAKAGE_FIELDS if field in col_set]
        return detected

    @classmethod
    def assert_no_leakage(cls, df_or_cols: pd.DataFrame | list[str] | pd.Index) -> None:
        """Assert that no target leakage fields exist in the provided dataframe or column list.

        Raises:
            ValueError: If one or more target leakage fields are detected.
        """
        cols = df_or_cols.columns if isinstance(df_or_cols, pd.DataFrame) else df_or_cols
        detected = cls.audit_columns(cols)

        if detected:
            msg = (
                f"CRITICAL DATA LEAKAGE VIOLATION! Detected forbidden leakage fields in feature matrix: {detected}. "
                f"These fields exist only post-churn and MUST be stripped before any feature engineering or model training."
            )
            logger.error(msg)
            raise ValueError(msg)

        logger.debug("LeakageGuard check passed: zero target leakage fields detected.")

    @classmethod
    def filter_leakage_fields(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Explicitly strip all target leakage fields from a DataFrame."""
        detected = cls.audit_columns(df.columns)
        if detected:
            logger.info(f"LeakageGuard stripping forbidden fields: {detected}")
            return df.drop(columns=detected)
        return df
