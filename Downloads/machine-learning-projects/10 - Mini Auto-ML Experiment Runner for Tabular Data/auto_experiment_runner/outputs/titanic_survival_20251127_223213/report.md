# Experiment Report: titanic_survival

**Generated**: 2025-11-27 22:32:17

---

## Dataset Summary

- **Total Rows**: 250
- **Total Columns**: 9
- **Numeric Features**: 7
- **Categorical Features**: 2
- **Target Column**: Survived

## Task Configuration

- **Task Type**: classification
- **Primary Metric**: f1_macro
- **Additional Metrics**: accuracy, roc_auc
- **Train/Valid Split**: 20% validation

## Leaderboard

Top 2 of 2 models ranked by **f1_macro**:

| Rank | Model | Train Time (s) | f1_macro | accuracy | roc_auc |
| --- | --- | --- | --- | --- | --- |
| 1 | logistic_regression | 0.047 | 0.6703 | 0.7000 | 0.6623 |
| 2 | random_forest | 0.963 | 0.4048 | 0.5200 | 0.4740 |

## Best Model: logistic_regression

**Winner**: `logistic_regression`

### Performance Metrics

- **f1_macro**: 0.6703
- **accuracy**: 0.7000
- **roc_auc**: 0.6623
- **Training Time**: 0.047 seconds

### Artifacts

- **Model**: `logistic_regression/model.joblib`
- **Metrics**: `logistic_regression/metrics.json`
- **Classification Report**: `logistic_regression/classification_report.json`
- **Confusion Matrix**: `logistic_regression/confusion_matrix.png`
- **Feature Importance (CSV)**: `logistic_regression/feature_importance.csv`
- **Feature Importance (Plot)**: `logistic_regression/feature_importance.png`

---


*All artifacts saved to: `C:\Users\wasif\OneDrive\Desktop\machine-learning-projects-master\10 - Mini Auto-ML Experiment Runner for Tabular Data\auto_experiment_runner\outputs\titanic_survival_20251127_223213`*