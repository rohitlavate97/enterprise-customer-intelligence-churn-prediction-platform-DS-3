"""Unit tests for ModelRegistry, Champion vs Challenger gate, and automated rollback."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from training.model_registry import ModelRegistry
from training.retraining_pipeline import AutomatedRetrainingPipeline


def test_model_registry_versioning(tmp_path):
    """Assert ModelRegistry saves versioned binary and updates manifest."""
    registry = ModelRegistry(registry_dir=tmp_path)
    model = DummyClassifier(strategy="prior")
    model.fit(np.zeros((10, 2)), np.array([0, 1] * 5))

    artifact_path = registry.register_version(model, "v1.0", {"pr_auc": 0.85}, description="Test v1.0")

    assert artifact_path.exists()
    manifest = registry.load_manifest()
    assert len(manifest["history"]) == 1
    assert manifest["history"][0]["version_tag"] == "v1.0"


def test_champion_promotion_and_rollback(tmp_path):
    """Assert promote_to_champion updates active Champion and rollback restores previous version."""
    registry = ModelRegistry(registry_dir=tmp_path)
    model1 = DummyClassifier(strategy="prior")
    model1.fit(np.zeros((10, 2)), np.array([0, 1] * 5))
    model2 = DummyClassifier(strategy="prior")
    model2.fit(np.zeros((10, 2)), np.array([0, 1] * 5))

    registry.register_version(model1, "v1.0", {"pr_auc": 0.85})
    registry.register_version(model2, "v1.1", {"pr_auc": 0.88})

    registry.promote_to_champion("v1.0")
    manifest = registry.load_manifest()
    assert manifest["active_champion"]["version_tag"] == "v1.0"

    registry.promote_to_champion("v1.1")
    manifest = registry.load_manifest()
    assert manifest["active_champion"]["version_tag"] == "v1.1"
    assert manifest["previous_champion"]["version_tag"] == "v1.0"

    # Test Rollback
    rolled_back = registry.rollback_champion()
    assert rolled_back is True
    manifest_after = registry.load_manifest()
    assert manifest_after["active_champion"]["version_tag"] == "v1.0"


def test_retraining_pipeline_cycle():
    """Assert AutomatedRetrainingPipeline runs retraining cycle and evaluates gate."""
    pipeline = AutomatedRetrainingPipeline(relative_margin_threshold=0.01)
    res = pipeline.run_retraining_cycle(n_samples=500, seed=42)

    assert "version_tag" in res
    assert isinstance(res["gate_passed"], bool)
    assert res["challenger_pr_auc"] > 0.70
