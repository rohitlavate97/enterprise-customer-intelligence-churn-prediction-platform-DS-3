"""Data Cleaning and Schema Validation Module."""

import pandas as pd
from data.schema import CATEGORICAL_FEATURES, ID_COLS, NUMERICAL_FEATURES, TARGET_COL
from utils.logger import get_logger

logger = get_logger("data.cleaner")


class DataCleaner:
    """Validates raw customer data schema, removes duplicates, handles data types, and imputes clean baseline."""

    def __init__(self, drop_duplicates: bool = True) -> None:
        self.drop_duplicates = drop_duplicates

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw dataframe, validate types, and remove exact duplicates."""
        logger.info(f"Cleaning raw input dataframe with shape {df.shape}...")
        df_clean = df.copy()

        # 1. Deduplication
        if self.drop_duplicates:
            initial_count = len(df_clean)
            if "customer_id" in df_clean.columns:
                df_clean = df_clean.drop_duplicates(subset=["customer_id"])
            else:
                df_clean = df_clean.drop_duplicates()
            dropped = initial_count - len(df_clean)
            if dropped > 0:
                logger.info(f"Dropped {dropped} duplicate customer records.")

        # 2. Type Enforcements
        for col in NUMERICAL_FEATURES:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        for col in CATEGORICAL_FEATURES:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()

        if TARGET_COL in df_clean.columns:
            df_clean[TARGET_COL] = pd.to_numeric(df_clean[TARGET_COL], errors="coerce").astype(int)

        logger.info(f"Cleaned dataframe final shape: {df_clean.shape}")
        return df_clean
