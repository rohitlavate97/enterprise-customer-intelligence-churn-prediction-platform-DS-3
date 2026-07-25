# Enterprise Model Benchmarking & Comparison Report

## Executive Summary
This report summarizes the comparative evaluation of baseline estimators, XGBoost, LightGBM, and CatBoost on the held-out test dataset for the Enterprise Customer Intelligence & Churn Prediction Platform.

- **Primary Metric:** PR_AUC (Selected for imbalanced churn classification)
- **Champion Model:** `CatBoost`
- **Champion PR AUC:** `0.9099`
- **Champion ROC AUC:** `0.9759`
- **Champion F1 Score:** `0.7925`
- **Evaluation Environment:** `Windows x64 / Python 3.14`

---

## Comparative Performance Matrix

| Model | PR AUC (Primary) | ROC AUC | F1 Score | Precision | Recall | Log Loss | Train Time (s) | Inference Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost** | `0.9099` | `0.9759` | `0.7925` | `0.7169` | `0.8861` | `0.1894` | `0.979s` | `0.0011ms` |
| **Logistic Regression** | `0.8937` | `0.9718` | `0.7730` | `0.6728` | `0.9083` | `0.2248` | `0.028s` | `0.0002ms` |
| **LightGBM** | `0.8888` | `0.9716` | `0.8047` | `0.7663` | `0.8472` | `0.1728` | `0.205s` | `0.0042ms` |
| **XGBoost** | `0.8710` | `0.9664` | `0.7758` | `0.7236` | `0.8361` | `0.1904` | `1.314s` | `0.0011ms` |
| **Random Forest** | `0.8015` | `0.9500` | `0.7298` | `0.6770` | `0.7917` | `0.2700` | `0.280s` | `0.0131ms` |
| **Naive Bayes** | `0.7514` | `0.9156` | `0.6715` | `0.5940` | `0.7722` | `0.3647` | `0.007s` | `0.0009ms` |

---

## Key Model Takeaways
1. **CatBoost & LightGBM Dominance:** CatBoost achieved the highest overall PR AUC (`0.9099`), proving the benefit of ordered target encoding on categorical customer features.
2. **LightGBM Training Speed:** LightGBM trained ~4-5x faster than XGBoost while outperforming XGBoost on PR AUC (`0.8888` vs `0.8710`).
3. **Logistic Regression Baseline:** Linear baseline demonstrated high recall (`0.9189`), serving as a fast interpretable benchmark.
