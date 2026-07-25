# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 05: Baseline Models & Stratified Cross-Validation
- Comprehensive model evaluation metrics suite (`evaluation/metrics.py`, `ModelEvaluator`) computing PR AUC, ROC AUC, F1, Precision, Recall, Log Loss, Brier Score, and Confusion Matrix.
- 8 baseline estimator definitions (`models/baselines.py`, `get_baseline_models`: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVM, KNN, Naive Bayes).
- `BaselineEvaluator` executing `StratifiedKFold` cross-validation with leakage-isolated preprocessing fitted strictly on training folds.
- Baseline model benchmark CLI runner script (`scripts/train_baselines.py`).
- Automated unit tests (`tests/test_baseline_models.py`) verifying metrics, 8 baseline definitions, cross-validation execution, and ground-truth signal recovery (PR AUC > 0.40).

### Added - Phase 04: Preprocessing Pipeline & Serving Parity
- Scikit-learn custom transformers (`features/transformers.py`, `DomainFeatureTransformer`, `OutlierClipper`).
- Reusable `PreprocessingPipelineBuilder` (`features/pipeline.py`) building `ColumnTransformer` with `SimpleImputer`, `RobustScaler`/`StandardScaler`, and `OneHotEncoder`.
- Pipeline serialization (`save_preprocessing_pipeline`, `load_preprocessing_pipeline`).
- Automated unit tests (`tests/test_preprocessing_pipeline.py`) verifying fit/transform separation and train/serve parity.

### Added - Phase 03: Data Pipeline & Target Leakage Guard
- Data cleaning and schema validation module (`data/cleaner.py`, `DataCleaner`).
- SHA256 dataset versioning and provenance tracking (`data/versioning.py`, `DatasetVersionManager`).
- Automated data quality and profiling reporter (`data/quality_report.py`, `DataQualityReporter`).
- Hard-stop target leakage assertion guard (`features/leakage_guard.py`, `LeakageGuard`).
- Leakage-safe domain feature engineering (`features/builder.py`, `FeatureBuilder`).
- End-to-end data pipeline runner script (`scripts/run_pipeline.py`).

### Added - Phase 02: Synthetic Customer Dataset Generator
- Seeded, reproducible synthetic dataset generator (`data/generator.py`, `CustomerDataGenerator`).
- Feature schema registry (`data/schema.py`).
- Ground-truth log-odds data generating process with controlled signal.

### Added - Phase 01: Project Setup
- Multi-package directory structure and Pydantic Settings configuration (`config/settings.py`).
