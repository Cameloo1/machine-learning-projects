# Experiment Report: housing_prices

**Generated**: 2025-11-27 22:32:42

---

## Dataset Summary

- **Total Rows**: 400
- **Total Columns**: 9
- **Numeric Features**: 8
- **Categorical Features**: 1
- **Target Column**: SalePrice

## Task Configuration

- **Task Type**: regression
- **Primary Metric**: r2
- **Additional Metrics**: rmse, mae
- **Train/Valid Split**: 20% validation

## Leaderboard

Top 2 of 2 models ranked by **r2**:

| Rank | Model | Train Time (s) | r2 | rmse | mae |
| --- | --- | --- | --- | --- | --- |
| 1 | linear_regression | 0.066 | 0.8838 | 31706.3848 | 26541.7146 |
| 2 | random_forest | 1.009 | 0.7936 | 42269.1204 | 34102.4253 |

## Best Model: linear_regression

**Winner**: `linear_regression`

### Performance Metrics

- **r2**: 0.8838
- **rmse**: 31706.3848
- **mae**: 26541.7146
- **Training Time**: 0.066 seconds

### Artifacts

- **Model**: `linear_regression/model.joblib`
- **Metrics**: `linear_regression/metrics.json`
- **Residual Plot**: `linear_regression/residuals.png`
- **Feature Importance (CSV)**: `linear_regression/feature_importance.csv`
- **Feature Importance (Plot)**: `linear_regression/feature_importance.png`

---


*All artifacts saved to: `C:\Users\wasif\OneDrive\Desktop\machine-learning-projects-master\10 - Mini Auto-ML Experiment Runner for Tabular Data\auto_experiment_runner\outputs\housing_prices_20251127_223238`*