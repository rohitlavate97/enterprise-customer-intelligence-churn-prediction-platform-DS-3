"""Test project setup, directory creation, configuration, and logging."""

from pathlib import Path
from config.settings import settings, Settings
from utils.logger import get_logger


def test_settings_initialization():
    """Assert settings load correctly with expected defaults."""
    assert settings.project_name == "Enterprise Customer Intelligence & Churn Prediction Platform"
    assert settings.seed == 42
    assert isinstance(settings.base_dir, Path)


def test_directory_structure_existence():
    """Assert runtime directories are created."""
    settings.ensure_directories()
    assert settings.data_dir.exists()
    assert settings.raw_data_dir.exists()
    assert settings.processed_data_dir.exists()
    assert settings.models_dir.exists()
    assert settings.artifacts_dir.exists()
    assert settings.logs_dir.exists()


def test_logger_creation():
    """Assert logger initializes and logs messages without errors."""
    logger = get_logger("test_logger")
    assert logger.name == "test_logger"
    logger.info("Setup test logging verification successful.")
