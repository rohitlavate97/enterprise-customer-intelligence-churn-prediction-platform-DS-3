"""Unit tests for model comparison report generator and champion model selection."""

import json
import pandas as pd
import pytest
from evaluation.comparison_report import ModelComparisonReporter


def test_comparison_matrix_generation():
    """Assert ModelComparisonReporter generates structured matrix sorted by PR AUC."""
    mock_evals = [
        {"model_name": "Logistic Regression", "pr_auc": 0.55, "roc_auc": 0.81, "f1_score": 0.52},
        {"model_name": "CatBoost", "pr_auc": 0.90, "roc_auc": 0.97, "f1_score": 0.81},
        {"model_name": "XGBoost", "pr_auc": 0.87, "roc_auc": 0.96, "f1_score": 0.77},
    ]

    reporter = ModelComparisonReporter(primary_metric="pr_auc")
    df_matrix = reporter.generate_comparison_matrix(mock_evals)

    assert len(df_matrix) == 3
    assert df_matrix.iloc[0]["model_name"] == "CatBoost"
    assert df_matrix.iloc[0]["pr_auc"] == 0.90
    assert df_matrix.iloc[1]["model_name"] == "XGBoost"


def test_champion_model_selection():
    """Assert select_champion_model picks the highest PR AUC model."""
    mock_evals = [
        {"model_name": "LightGBM", "pr_auc": 0.88, "roc_auc": 0.97, "f1_score": 0.80},
        {"model_name": "CatBoost", "pr_auc": 0.91, "roc_auc": 0.98, "f1_score": 0.82},
    ]

    reporter = ModelComparisonReporter(primary_metric="pr_auc")
    df_matrix = reporter.generate_comparison_matrix(mock_evals)
    champion = reporter.select_champion_model(df_matrix)

    assert champion["model_name"] == "CatBoost"
    assert champion["pr_auc"] == 0.91


def test_report_file_generation(tmp_path):
    """Assert markdown report and JSON summary artifacts are written to disk."""
    mock_evals = [
        {"model_name": "CatBoost", "pr_auc": 0.91, "roc_auc": 0.98, "f1_score": 0.82, "training_time_sec": 1.2, "inference_latency_ms": 0.001},
        {"model_name": "XGBoost", "pr_auc": 0.87, "roc_auc": 0.96, "f1_score": 0.77, "training_time_sec": 1.5, "inference_latency_ms": 0.002},
    ]

    reporter = ModelComparisonReporter(primary_metric="pr_auc")
    df_matrix = reporter.generate_comparison_matrix(mock_evals)
    md_path, json_path = reporter.save_reports(df_matrix, output_dir=tmp_path)

    assert md_path.exists()
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["champion_model"]["model_name"] == "CatBoost"
