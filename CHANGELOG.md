# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 09: Model Comparison & Benchmarking Report
- Model comparison reporter (`evaluation/comparison_report.py`, `ModelComparisonReporter`) aggregating metrics, training times, and inference latencies across all candidate models.
- Automated selection of Champion Model based on PR AUC.
- Generated markdown benchmark documentation (`docs/MODEL_COMPARISON_REPORT.md`) and JSON summary payload (`models/artifacts/model_comparison_summary.json`).
- Comparative benchmark execution CLI script (`scripts/run_model_comparison.py`).
- Automated unit tests (`tests/test_model_comparison.py`) verifying matrix generation, champion selection, and artifact creation.

### Added - Phase 08: CatBoost Classifier Suite & Triple Benchmark
- CatBoost training pipeline (`models/catboost_suite.py`, `CatBoostTrainer`) featuring ordered target encoding and early stopping.
- CLI script (`scripts/train_catboost.py`) running a triple gradient boosting benchmark.
- Automated unit tests (`tests/test_catboost_suite.py`).

### Added - Phase 07: LightGBM Classifier Suite & Benchmark
- LightGBM training pipeline (`models/lightgbm_suite.py`, `LightGBMTrainer`).
- CLI script (`scripts/train_lightgbm.py`).

### Added - Phase 06: XGBoost Classifier Suite & SHAP Explainability
- XGBoost training pipeline (`models/xgboost_suite.py`, `XGBoostTrainer`).
- CLI script (`scripts/train_xgboost.py`).

### Added - Phase 05: Baseline Models & Stratified Cross-Validation
- Comprehensive model evaluation metrics suite (`evaluation/metrics.py`, `ModelEvaluator`).

### Added - Phase 04: Preprocessing Pipeline & Serving Parity
- Scikit-learn custom transformers (`features/transformers.py`) & `PreprocessingPipelineBuilder` (`features/pipeline.py`).

### Added - Phase 03: Data Pipeline & Target Leakage Guard
- Data cleaning, versioning, quality reporter, and `LeakageGuard`.

### Added - Phase 02: Synthetic Customer Dataset Generator
- Seeded, reproducible synthetic dataset generator (`data/generator.py`).

### Added - Phase 01: Project Setup
- Multi-package directory structure and Pydantic Settings configuration (`config/settings.py`).
