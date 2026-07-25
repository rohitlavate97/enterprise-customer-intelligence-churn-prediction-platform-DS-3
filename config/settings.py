"""Global configuration settings for Enterprise Churn Prediction Platform."""

import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Platform configuration settings loaded from environment or yaml."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="CHURN_",
    )

    # Core Project Settings
    project_name: str = "Enterprise Customer Intelligence & Churn Prediction Platform"
    environment: str = "development"
    seed: int = 42

    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "models"
    artifacts_dir: Path = BASE_DIR / "models" / "artifacts"
    logs_dir: Path = BASE_DIR / "logs"

    # MLflow Settings
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="enterprise-churn-prediction")

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    latency_budget_ms: float = 200.0

    # Streamlit Settings
    dashboard_port: int = 8501

    # Real-Time Event Stream Simulation Settings
    event_stream_rate_sec: float = 1.0

    def ensure_directories(self) -> None:
        """Ensure all required runtime directories exist."""
        for path in [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.artifacts_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Instantiate global singleton settings
settings = Settings()
settings.ensure_directories()
