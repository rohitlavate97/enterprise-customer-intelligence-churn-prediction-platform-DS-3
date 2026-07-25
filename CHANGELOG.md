# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 06: XGBoost Classifier Suite & SHAP Explainability
- XGBoost training pipeline (`models/xgboost_suite.py`, `XGBoostTrainer`) featuring:
  - Early stopping on a dedicated validation fold.
  - Hyperparameter optimization via `RandomizedSearchCV`.
  - TreeExplainer SHAP & feature importance extraction.
- CLI script (`scripts/train_xgboost.py`) to tune, train, evaluate, compute feature importances, and save model artifacts (`models/artifacts/xgboost_model.joblib`).
- Automated unit tests (`tests/test_xgboost_suite.py`) verifying early stopping, probability calibration, and SHAP/feature importance matrix calculation.

### Added - Phase 05: Baseline Models & Stratified Cross-Validation
- Comprehensive model evaluation metrics suite (`evaluation/metrics.py`, `ModelEvaluator`) computing PR AUC, ROC AUC, F1, Precision, Recall, Log Loss, Brier Score, and Confusion Matrix.
- 8 baseline estimator definitions (`models/baselines.py`, `get_baseline_models`).
- `BaselineEvaluator` executing `StratifiedKFold` cross-validation with leakage-isolated preprocessing.
- Baseline model benchmark CLI runner script (`scripts/train_baselines.py`).

### Added - Phase 04: Preprocessing Pipeline & Serving Parity
- Scikit-learn custom transformers (`features/transformers.py`, `DomainFeatureTransformer`, `OutlierClipper`).
- Reusable `PreprocessingPipelineBuilder` (`features/pipeline.py`) building `ColumnTransformer`.
- Pipeline serialization (`save_preprocessing_pipeline`, `load_preprocessing_pipeline`).

### Added - Phase 03: Data Pipeline & Target Leakage Guard
- Data cleaning and schema validation (`data/cleaner.py`, `DataCleaner`).
- SHA256 dataset versioning and provenance tracking (`data/versioning.py`).
- Automated data quality and profiling reporter (`data/quality_report.py`).
- Hard-stop target leakage assertion guard (`features/leakage_guard.py`, `LeakageGuard`).
- Leakage-safe domain feature engineering (`features/builder.py`, `FeatureBuilder`).

### Added - Phase 02: Synthetic Customer Dataset Generator
- Seeded, reproducible synthetic dataset generator (`data/generator.py`, `CustomerDataGenerator`).

### Added - Phase 01: Project Setup
- Multi-package directory structure and Pydantic Settings configuration (`config/settings.py`).
