# Changelog

All notable changes to the Enterprise Customer Intelligence & Churn Prediction Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Phase 01: Project Setup
- Multi-package directory structure (`config`, `data`, `features`, `models`, `training`, `evaluation`, `explainability`, `deployment`, `api`, `dashboard`, `streaming`, `monitoring`, `utils`, `tests`).
- Centralized Pydantic Settings configuration (`config/settings.py`, `config/default_config.yaml`).
- Structured logging module with file and stdout handlers (`utils/logger.py`).
- Docker execution environment (`docker/Dockerfile`, `docker/Dockerfile.mlflow`, `docker-compose.yml`).
- Pre-commit hooks (`.pre-commit-config.yaml`) and GitHub Actions CI workflow (`.github/workflows/ci.yml`).
- Initial unit test verifying setup, paths, and logger (`tests/test_setup.py`).
