"""Custom scikit-learn compatible feature transformers."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from features.builder import FeatureBuilder
from features.leakage_guard import LeakageGuard
from utils.logger import get_logger

logger = get_logger("features.transformers")


class DomainFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn wrapper for domain feature builder."""

    def __init__(self, enforce_leakage_guard: bool = True) -> None:
        self.enforce_leakage_guard = enforce_leakage_guard
        self.builder = FeatureBuilder(enforce_leakage_guard=enforce_leakage_guard)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DomainFeatureTransformer":
        """Stateless fit method."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input dataframe by computing domain features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self.builder.transform(X)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """IQR-based outlier clipper fit exclusively on training fold distributions."""

    def __init__(self, factor: float = 1.5, num_cols: list[str] | None = None) -> None:
        self.factor = factor
        self.num_cols = num_cols
        self.lower_bounds_: dict[str, float] = {}
        self.upper_bounds_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "OutlierClipper":
        """Learn IQR lower and upper bounds for numerical columns on training data."""
        df = X.copy()
        cols = self.num_cols or list(df.select_dtypes(include=[np.number]).columns)

        for col in cols:
            if col in df.columns:
                q25 = df[col].quantile(0.25)
                q75 = df[col].quantile(0.75)
                iqr = q75 - q25
                self.lower_bounds_[col] = float(q25 - self.factor * iqr)
                self.upper_bounds_[col] = float(q75 + self.factor * iqr)

        logger.debug(f"OutlierClipper fit learned bounds for {len(self.lower_bounds_)} columns.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip values outside learned training bounds."""
        df = X.copy()
        for col, lower in self.lower_bounds_.items():
            if col in df.columns:
                upper = self.upper_bounds_[col]
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df
