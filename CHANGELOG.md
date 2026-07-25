# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 10: Explainability Layer & Segment Fairness Audit
- Global and local per-customer explanation generator (`explainability/shap_explainer.py`, `ModelExplainer`).
- Plain-language business narrative translation (`translate_feature_to_business_reason`) mapping model contribution scores to actionable retention team recommendations.
- Partial Dependence Plot computation engine (`explainability/pdp_analysis.py`, `PartialDependenceAnalyzer`).
- Segment fairness and error rate disparity auditor (`explainability/segment_fairness.py`, `SegmentFairnessAuditor`) auditing performance across tenure bands, contract types, plan tiers, and geographies.
- Explainability and fairness CLI runner script (`scripts/run_explainability.py`).
- Automated unit tests (`tests/test_explainability.py`) verifying business narrative translations, local traces, PDP grid calculations, and fairness disparity alerts.

### Added - Phase 09: Model Comparison & Benchmarking Report
- Model comparison reporter (`evaluation/comparison_report.py`, `ModelComparisonReporter`).
- Generated markdown benchmark documentation (`docs/MODEL_COMPARISON_REPORT.md`).

### Added - Phase 08: CatBoost Classifier Suite & Triple Benchmark
- CatBoost training pipeline (`models/catboost_suite.py`, `CatBoostTrainer`).

### Added - Phase 07: LightGBM Classifier Suite & Benchmark
- LightGBM training pipeline (`models/lightgbm_suite.py`, `LightGBMTrainer`).

### Added - Phase 06: XGBoost Classifier Suite & SHAP Explainability
- XGBoost training pipeline (`models/xgboost_suite.py`, `XGBoostTrainer`).

### Added - Phase 05: Baseline Models & Stratified Cross-Validation
- Comprehensive model evaluation metrics suite (`evaluation/metrics.py`).

### Added - Phase 04: Preprocessing Pipeline & Serving Parity
- Custom transformers (`features/transformers.py`) & `PreprocessingPipelineBuilder` (`features/pipeline.py`).

### Added - Phase 03: Data Pipeline & Target Leakage Guard
- Data cleaning, versioning, quality reporter, and `LeakageGuard`.

### Added - Phase 02: Synthetic Customer Dataset Generator
- Seeded, reproducible synthetic dataset generator (`data/generator.py`).

### Added - Phase 01: Project Setup
- Multi-package directory structure and Pydantic Settings configuration (`config/settings.py`).
