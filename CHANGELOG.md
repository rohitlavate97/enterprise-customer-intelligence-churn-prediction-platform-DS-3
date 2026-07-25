# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 04: Preprocessing Pipeline & Serving Parity
- Scikit-learn custom transformers (`features/transformers.py`, `DomainFeatureTransformer`, `OutlierClipper`).
- Reusable `PreprocessingPipelineBuilder` (`features/pipeline.py`) building `ColumnTransformer` with `SimpleImputer`, `RobustScaler`/`StandardScaler`, and `OneHotEncoder`.
- Pipeline serialization (`save_preprocessing_pipeline`, `load_preprocessing_pipeline`).
- Automated unit tests (`tests/test_preprocessing_pipeline.py`) verifying:
  - Provable fit/transform separation (asserting fit statistics are locked on training folds).
  - Train/serve parity (asserting single-row request transformation matches batch output).
  - Serialization roundtrip fidelity.

### Added - Phase 03: Data Pipeline & Target Leakage Guard
- Data cleaning and schema validation module (`data/cleaner.py`, `DataCleaner`).
- SHA256 dataset versioning and provenance tracking (`data/versioning.py`, `DatasetVersionManager`).
- Automated data quality and profiling reporter (`data/quality_report.py`, `DataQualityReporter`).
- Hard-stop target leakage assertion guard (`features/leakage_guard.py`, `LeakageGuard`).
- Leakage-safe domain feature engineering (`features/builder.py`, `FeatureBuilder`).
- End-to-end data pipeline runner script (`scripts/run_pipeline.py`).
- Automated unit tests (`tests/test_data_pipeline.py`) asserting deduplication, versioning checksums, quality reports, leakage assertions, and feature engineering.

### Added - Phase 02: Synthetic Customer Dataset Generator
- Seeded, reproducible synthetic dataset generator (`data/generator.py`, `CustomerDataGenerator`).
- Feature schema registry (`data/schema.py`) categorizing ID, Numerical, Categorical, Target, and explicit Leakage fields (`LEAKAGE_FIELDS`).
- Ground-truth log-odds data generating process with controlled signal (support tickets, contract type, payment failures, price increases) + noise.
- Target leakage field generation (`cancellation_processed_date`, `final_invoice_flag`, `account_status_deactivated`, `churn_reason_recorded`) populated exclusively for churners to audit leakage prevention.
- Data generation CLI script (`scripts/generate_data.py`).
- Automated tests (`tests/test_data_generator.py`) asserting reproducibility, schema compliance, target leakage isolation, and ground-truth signal recovery.

### Added - Phase 01: Project Setup
- Multi-package directory structure (`config`, `data`, `features`, `models`, `training`, `evaluation`, `explainability`, `deployment`, `api`, `dashboard`, `streaming`, `monitoring`, `utils`, `tests`).
- Centralized Pydantic Settings configuration (`config/settings.py`, `config/default_config.yaml`).
- Structured logging module with file and stdout handlers (`utils/logger.py`).
- Docker execution environment (`docker/Dockerfile`, `docker/Dockerfile.mlflow`, `docker-compose.yml`).
- Pre-commit hooks (`.pre-commit-config.yaml`) and GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- Initial unit test verifying setup, paths, and logger (`tests/test_setup.py`).
