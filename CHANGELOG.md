# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 07: LightGBM Classifier Suite & Benchmark
- LightGBM training pipeline (`models/lightgbm_suite.py`, `LightGBMTrainer`) featuring:
  - Histogram-based feature binning and leaf-wise tree growth.
  - Early stopping callbacks (`binary_logloss`).
  - Hyperparameter optimization via `RandomizedSearchCV` (`num_leaves`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`).
- CLI script (`scripts/train_lightgbm.py`) benchmarking LightGBM vs XGBoost on training speed, inference latency, PR AUC, ROC AUC, and Log Loss, saving model artifact (`models/artifacts/lightgbm_model.joblib`).
- Automated unit tests (`tests/test_lightgbm_suite.py`) verifying early stopping, probability calibration, and benchmarking against XGBoost.

### Added - Phase 06: XGBoost Classifier Suite & SHAP Explainability
- XGBoost training pipeline (`models/xgboost_suite.py`, `XGBoostTrainer`).
- CLI script (`scripts/train_xgboost.py`).
- Automated unit tests (`tests/test_xgboost_suite.py`).

### Added - Phase 05: Baseline Models & Stratified Cross-Validation
- Comprehensive model evaluation metrics suite (`evaluation/metrics.py`, `ModelEvaluator`).
- 8 baseline estimator definitions (`models/baselines.py`, `get_baseline_models`).

### Added - Phase 04: Preprocessing Pipeline & Serving Parity
- Scikit-learn custom transformers (`features/transformers.py`) & `PreprocessingPipelineBuilder` (`features/pipeline.py`).

### Added - Phase 03: Data Pipeline & Target Leakage Guard
- Data cleaning (`data/cleaner.py`), versioning (`data/versioning.py`), quality reporter (`data/quality_report.py`), and `LeakageGuard` (`features/leakage_guard.py`).

### Added - Phase 02: Synthetic Customer Dataset Generator
- Seeded, reproducible synthetic dataset generator (`data/generator.py`, `CustomerDataGenerator`).

### Added - Phase 01: Project Setup
- Multi-package directory structure and Pydantic Settings configuration (`config/settings.py`).
