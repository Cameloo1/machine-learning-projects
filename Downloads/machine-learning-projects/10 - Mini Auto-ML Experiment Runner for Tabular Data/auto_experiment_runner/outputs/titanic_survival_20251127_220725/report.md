# Experiment Report: titanic_survival

**Generated**: 2025-11-27 22:07:28

---

## Dataset Summary

- **Total Rows**: 10
- **Total Columns**: 10
- **Numeric Features**: 5
- **Categorical Features**: 5
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
| 1 | random_forest | 0.547 | 1.0000 | 1.0000 | 1.0000 |
| 2 | logistic_regression | 0.068 | 0.3333 | 0.5000 | 0.0000 |

## Best Model: random_forest

**Winner**: `random_forest`

### Performance Metrics

- **f1_macro**: 1.0000
- **accuracy**: 1.0000
- **roc_auc**: 1.0000
- **Training Time**: 0.547 seconds

### Artifacts

- **Model**: `random_forest/model.joblib`
- **Metrics**: `random_forest/metrics.json`
- **Classification Report**: `random_forest/classification_report.json`
- **Confusion Matrix**: `random_forest/confusion_matrix.png`
- **Feature Importance (CSV)**: `random_forest/feature_importance.csv`
- **Feature Importance (Plot)**: `random_forest/feature_importance.png`

---


*All artifacts saved to: `C:\Users\wasif\OneDrive\Desktop\machine-learning-projects-master\10 - Mini Auto-ML Experiment Runner for Tabular Data\auto_experiment_runner\outputs\titanic_survival_20251127_220725`*