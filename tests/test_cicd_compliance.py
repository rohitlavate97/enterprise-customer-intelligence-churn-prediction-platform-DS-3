"""Unit tests for CI/CD workflow YAML structure and integrity."""

from pathlib import Path
import yaml
from config.settings import settings


def test_ci_workflow_yaml_existence_and_validity():
    """Assert .github/workflows/ci.yml exists and contains valid YAML with required jobs."""
    workflow_path = settings.base_dir / ".github" / "workflows" / "ci.yml"
    assert workflow_path.exists(), f"Workflow file missing at {workflow_path}"

    with open(workflow_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "name" in config
    assert "on" in config or True in config
    assert "jobs" in config
    assert "lint-and-test" in config["jobs"]
    assert "e2e-workflow-audit" in config["jobs"]
