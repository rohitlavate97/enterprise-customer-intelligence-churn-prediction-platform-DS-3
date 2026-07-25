# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 17: End-to-End System Integration Tests & Master Workflow Orchestrator
- Master CLI Workflow Orchestrator (`scripts/run_full_pipeline.py`) executing all platform stages sequentially in 4.29s:
  1. Synthetic Data Generation with Ground-Truth Signal.
  2. Data Cleaning & Data Quality Profiling.
  3. Domain Feature Engineering & Leakage Guard.
  4. Preprocessing Pipeline & CatBoost Champion Training with Early Stopping.
  5. Explainability Layer & Segment Fairness Audit.
  6. Financial ROI & High-Risk Customer Retention Call List.
  7. Real-Time Customer Event Streaming Simulation & High-Risk Alerts.
  8. Feature Distribution Drift Audit (PSI & KS tests).
  9. Automated Retraining Cycle & Champion vs Challenger Gate.
- End-to-End System Integration Test Suite (`tests/test_e2e_pipeline.py`) validating the complete lifecycle.

### Added - Phase 16: Automated Retraining Pipeline & Model Registry
- Versioned model binary registry (`training/model_registry.py`) saving SHA256 checksums and JSON manifest logs.
- Champion vs. Challenger validation gate (`training/retraining_pipeline.py`) enforcing $\ge 1.0\%$ relative PR AUC gain threshold.
- Automated rollback mechanism (`rollback_champion`).

### Added - Phase 15: Model Monitoring & Data/Concept Drift Engine
- Feature distribution Population Stability Index (PSI) calculator (`monitoring/drift_detector.py`).
- Kolmogorov-Smirnov (KS) test for feature drift.
- Concept drift monitor and automated retraining trigger.

### Added - Phase 14: Real-Time Streaming Engine & High-Risk Alert Dispatcher
- High-velocity customer activity event producer (`streaming/producer.py`).
- Real-time stream processor consumer (`streaming/consumer.py`).

### Added - Phase 13: Interactive Streamlit Executive Dashboard & ROI Simulator
- Multi-tab Streamlit dashboard application (`dashboard/app.py`).

### Added - Phase 12: Production FastAPI Inference Server & Swagger / OpenAPI Documentation
- Custom Swagger UI (`/docs`), ReDoc (`/redoc`), OpenAPI 3.0 export (`/openapi.json`), `/explain`, and Prometheus `/metrics` endpoints.

### Added - Phase 11: Business Analytics Engine & Retention Intervention ROI
- Customer Lifetime Value (CLV) calculation engine (`evaluation/business_roi.py`).

### Added - Phase 10: Explainability Layer & Segment Fairness Audit
- Global and local per-customer explanation generator (`explainability/shap_explainer.py`).

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
