"""Unit tests verifying final master documentation and production handbook completeness."""

from pathlib import Path
from config.settings import settings


def test_production_readiness_handbook_existence():
    """Assert PRODUCTION_READINESS_HANDBOOK.md exists and contains required sections."""
    handbook_path = settings.base_dir / "docs" / "PRODUCTION_READINESS_HANDBOOK.md"
    assert handbook_path.exists(), f"Handbook missing at {handbook_path}"

    content = handbook_path.read_text(encoding="utf-8")
    assert "System Architecture Deep-Dive" in content
    assert "Target Leakage Prevention" in content
    assert "Model Benchmark Matrix" in content
    assert "Financial ROI" in content
    assert "Disaster Recovery" in content


def test_readme_master_documentation_completeness():
    """Assert README.md exists and contains QuickStart, Architecture, and CLI reference."""
    readme_path = settings.base_dir / "README.md"
    assert readme_path.exists(), f"README.md missing at {readme_path}"

    content = readme_path.read_text(encoding="utf-8")
    assert "QuickStart Guide" in content
    assert "Complete CLI Script Reference Guide" in content
    assert "System Architecture" in content
