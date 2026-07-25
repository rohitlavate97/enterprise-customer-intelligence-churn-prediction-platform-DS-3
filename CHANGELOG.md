# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 14: Real-Time Streaming Engine & High-Risk Alert Dispatcher
- High-velocity customer activity event producer (`streaming/producer.py`, `CustomerEventProducer`) generating telemetry events (`support_ticket_opened`, `usage_drop_detected`, `payment_failed`, `competitor_viewed`).
- Real-time stream processor consumer (`streaming/consumer.py`, `StreamProcessorConsumer`) featuring 5-minute sliding window state tracking per customer.
- Real-time Slack/Webhook style JSON alert dispatcher (`AlertDispatcher`) emitting critical churn alerts when predicted churn probability exceeds threshold ($\ge 0.80$).
- Real-time streaming pipeline CLI script (`scripts/run_streaming_pipeline.py`) exporting `models/artifacts/streaming_alerts.json`.
- Automated unit tests (`tests/test_streaming.py`) verifying producer event schema generation, sliding window aggregation, and Slack alert payload formatting.

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
