"""Data management package initialization."""

from data.generator import CustomerDataGenerator, generate_and_save_dataset
from data.schema import (
    ALL_PREDICTOR_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLS,
    LEAKAGE_FIELDS,
    NUMERICAL_FEATURES,
    TARGET_COL,
)

__all__ = [
    "CustomerDataGenerator",
    "generate_and_save_dataset",
    "TARGET_COL",
    "ID_COLS",
    "LEAKAGE_FIELDS",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "ALL_PREDICTOR_FEATURES",
]
