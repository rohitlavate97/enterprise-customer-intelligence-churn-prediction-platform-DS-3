# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-25

### Added - Phase 19: Final Master Documentation, System Architecture Diagrams & Production Release
- Production Readiness Handbook (`docs/PRODUCTION_READINESS_HANDBOOK.md`) featuring:
  - Complete End-to-End System Architecture Deep-Dive.
  - Target Leakage Prevention & Audit Proof.
  - Model Performance Benchmark Summary Table (CatBoost Champion **0.9099 PR AUC** vs LightGBM **0.8888** vs XGBoost **0.8710**).
  - Business Analytics & Financial Impact ($360K Revenue at Risk, 304.2% Campaign ROI).
  - Data & Concept Drift Strategy (PSI thresholds, KS testing, Retraining triggers).
  - Disaster Recovery & Fast Rollback Protocols.
- Comprehensive Master README (`README.md`) with Mermaid architecture diagrams, QuickStart guide, and full CLI Reference Guide.
- Final release unit tests (`tests/test_final_release.py`).
- Final Production Release tag `v2.0-production-release`.

### Added - Phase 18: CI/CD Pipeline & GitHub Actions Automation
- Complete GitHub Actions CI/CD workflow (`.github/workflows/ci.yml`).

### Added - Phase 17: End-to-End System Integration Tests & Master Workflow Orchestrator
- Master CLI Workflow Orchestrator (`scripts/run_full_pipeline.py`).

### Added - Phase 16: Automated Retraining Pipeline & Model Registry
- Versioned model binary registry (`training/model_registry.py`).
- Champion vs. Challenger validation gate (`training/retraining_pipeline.py`).
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
