"""Unit tests for DriftDetector, PSI calculation, KS testing, and automated retraining trigger."""

import numpy as np
import pandas as pd
import pytest
from monitoring.concept_drift import ConceptDriftMonitor
from monitoring.drift_detector import DriftDetector


def test_calculate_psi_identical():
    """Assert PSI is 0.0 for identical distributions."""
    np.random.seed(42)
    ref = np.random.normal(100, 15, 1000)
    cur = ref.copy()

    psi = DriftDetector.calculate_psi(ref, cur)
    assert psi == 0.0


def test_calculate_psi_shifted():
    """Assert PSI is > 0.25 for a significantly shifted distribution."""
    np.random.seed(42)
    ref = np.random.normal(100, 15, 1000)
    cur = np.random.normal(150, 25, 1000)  # Significant shift

    psi = DriftDetector.calculate_psi(ref, cur)
    assert psi >= 0.25


def test_ks_test_drift():
    """Assert KS test detects distribution drift on shifted data."""
    np.random.seed(42)
    ref = np.random.normal(50, 5, 500)
    cur_shifted = np.random.normal(70, 5, 500)

    res = DriftDetector.calculate_ks_test(ref, cur_shifted)
    assert res["drift_detected"] is True
    assert res["p_value"] < 0.05


def test_concept_drift_retraining_trigger():
    """Assert retraining trigger condition is set when PSI >= 0.25 or PR AUC < 0.80."""
    monitor = ConceptDriftMonitor(pr_auc_threshold=0.80, psi_threshold=0.25)

    drift_res_ok = {"dataset_drift_detected": False, "high_drift_feature_count": 0}
    concept_res_ok = {"concept_drift_detected": False, "current_pr_auc": 0.88}

    decision_ok = monitor.check_retraining_trigger(drift_res_ok, concept_res_ok)
    assert decision_ok["retraining_required"] is False

    # Trigger via feature drift
    drift_res_bad = {"dataset_drift_detected": True, "high_drift_feature_count": 2}
    decision_bad = monitor.check_retraining_trigger(drift_res_bad, concept_res_ok)
    assert decision_bad["retraining_required"] is True
