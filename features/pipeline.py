"""Scikit-learn preprocessing pipeline builder ensuring strict fit/transform separation."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from config.settings import settings
from data.schema import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from features.builder import FeatureBuilder
from features.leakage_guard import LeakageGuard
from utils.logger import get_logger

logger = get_logger("features.pipeline")


class PreprocessingPipelineBuilder:
    """Constructs leak-free scikit-learn ColumnTransformer and Pipeline objects."""

    def __init__(
        self,
        scaler_type: str = "robust",
        impute_strategy: str = "median",
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.scaler_type = scaler_type
        self.impute_strategy = impute_strategy
        self.numerical_features = numerical_features or NUMERICAL_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES

    def build_column_transformer(self) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer for numerical and categorical features."""
        # Scaler selection
        scaler = RobustScaler() if self.scaler_type == "robust" else StandardScaler()

        # Numerical pipeline
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.impute_strategy)),
                ("scaler", scaler),
            ]
        )

        # Categorical pipeline
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.numerical_features),
                ("cat", cat_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )

        return preprocessor

    def build_pipeline(self) -> Pipeline:
        """Construct unified end-to-end preprocessing pipeline."""
        column_transformer = self.build_column_transformer()

        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", column_transformer),
            ]
        )
        return full_pipeline


def save_preprocessing_pipeline(
    pipeline: Pipeline,
    output_path: Path | None = None,
) -> Path:
    """Serialize fitted preprocessing pipeline artifact to disk."""
    out_path = output_path or (settings.artifacts_dir / "preprocessing_pipeline.joblib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    logger.info(f"Saved preprocessing pipeline artifact to {out_path}")
    return out_path


def load_preprocessing_pipeline(input_path: Path | None = None) -> Pipeline:
    """Load serialized preprocessing pipeline artifact from disk."""
    in_path = input_path or (settings.artifacts_dir / "preprocessing_pipeline.joblib")
    if not in_path.exists():
        raise FileNotFoundError(f"Preprocessing pipeline artifact not found at {in_path}")
    pipeline = joblib.load(in_path)
    logger.info(f"Loaded preprocessing pipeline artifact from {in_path}")
    return pipeline
