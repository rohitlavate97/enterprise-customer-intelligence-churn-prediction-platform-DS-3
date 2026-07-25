# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 11: Business Analytics Engine & Retention Intervention ROI
- Customer Lifetime Value (CLV) calculation engine (`evaluation/business_roi.py`, `BusinessROIAnalyzer`).
- Total revenue at risk estimation and retention intervention campaign ROI modeling.
- High-risk customer prioritized retention call list generator (`generate_high_risk_call_list`) outputting CSV artifacts with actionable business recommendations.
- Business analytics CLI runner (`scripts/run_business_analytics.py`) exporting `models/artifacts/business_analytics_report.json` and `models/artifacts/high_risk_call_list.csv`.
- Automated unit tests (`tests/test_business_roi.py`) verifying CLV formulas, campaign net profit calculation, and call list sorting.

### Added - Phase 10: Explainability Layer & Segment Fairness Audit
- Global and local per-customer explanation generator (`explainability/shap_explainer.py`, `ModelExplainer`).
- Plain-language business narrative translation mapping model contribution scores to actionable recommendations.
- Partial Dependence Plot computation engine (`explainability/pdp_analysis.py`, `PartialDependenceAnalyzer`).
- Segment fairness and error rate disparity auditor (`explainability/segment_fairness.py`, `SegmentFairnessAuditor`).

### Added - Phase 09: Model Comparison & Benchmarking Report
- Model comparison reporter (`evaluation/comparison_report.py`, `ModelComparisonReporter`).

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
