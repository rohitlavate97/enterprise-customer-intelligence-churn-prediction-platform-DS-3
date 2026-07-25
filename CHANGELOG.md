# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 08: CatBoost Classifier Suite & Triple Gradient Boosting Benchmark
- CatBoost training pipeline (`models/catboost_suite.py`, `CatBoostTrainer`) featuring:
  - Ordered target encoding to prevent categorical target leakage.
  - Early stopping on a dedicated validation fold.
  - Hyperparameter optimization via `RandomizedSearchCV` (`depth`, `l2_leaf_reg`, `learning_rate`, `scale_pos_weight`).
- CLI script (`scripts/train_catboost.py`) running a triple gradient boosting benchmark (CatBoost vs LightGBM vs XGBoost) evaluating PR AUC, ROC AUC, F1, Log Loss, Training Time, and Inference Latency, saving artifact (`models/artifacts/catboost_model.joblib`).
- Automated unit tests (`tests/test_catboost_suite.py`) verifying early stopping, probability calibration, and triple model benchmark (CatBoost PR AUC: 0.9099).

### Added - Phase 07: LightGBM Classifier Suite & Benchmark
- LightGBM training pipeline (`models/lightgbm_suite.py`, `LightGBMTrainer`).
- CLI script (`scripts/train_lightgbm.py`).
- Automated unit tests (`tests/test_lightgbm_suite.py`).

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
