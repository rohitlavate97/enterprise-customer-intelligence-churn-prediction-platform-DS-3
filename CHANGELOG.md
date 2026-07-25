# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 12: Production FastAPI Inference Server & Swagger / OpenAPI Documentation
- Custom Swagger UI (`/docs`) and ReDoc (`/redoc`) documentation setup with tagged endpoint categories ("Inference", "Explainability", "System & Health", "Monitoring").
- Pydantic v2 schemas (`api/schemas.py`) enriched with `json_schema_extra` examples and field descriptions.
- Added `/explain/{customer_id}` endpoint returning detailed SHAP feature contribution breakdowns and plain-language retention summary narratives.
- Added `/metrics` endpoint serving Prometheus-compatible operational metrics (`churn_api_requests_total`, `churn_predictions_total`, `churn_high_risk_predictions_total`, `churn_api_uptime_seconds`).
- Added raw OpenAPI 3.0 specification export route `/api/v1/openapi.json`.
- Automated unit tests (`tests/test_api_docs.py`) verifying Swagger UI, ReDoc, OpenAPI schema compliance, `/explain`, and `/metrics`.

### Added - Phase 11: Business Analytics Engine & Retention Intervention ROI
- Customer Lifetime Value (CLV) calculation engine (`evaluation/business_roi.py`).
- Total revenue at risk estimation and retention intervention campaign ROI modeling.
- High-risk customer prioritized retention call list generator.

### Added - Phase 10: Explainability Layer & Segment Fairness Audit
- Global and local per-customer explanation generator (`explainability/shap_explainer.py`).
- Plain-language business narrative translation.
- Partial Dependence Plot computation engine (`explainability/pdp_analysis.py`).
- Segment fairness and error rate disparity auditor (`explainability/segment_fairness.py`).

### Added - Phase 09: Model Comparison & Benchmarking Report
- Model comparison reporter (`evaluation/comparison_report.py`).

### Added - Phase 08: CatBoost Classifier Suite & Triple Benchmark
- CatBoost training pipeline (`models/catboost_suite.py`).

### Added - Phase 07: LightGBM Classifier Suite & Benchmark
- LightGBM training pipeline (`models/lightgbm_suite.py`).

### Added - Phase 06: XGBoost Classifier Suite & SHAP Explainability
- XGBoost training pipeline (`models/xgboost_suite.py`).

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
