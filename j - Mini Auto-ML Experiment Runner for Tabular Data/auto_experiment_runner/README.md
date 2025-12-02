# Mini Auto-ML Experiment Runner for Tabular Data

Mini AutoTab runs a curated collection of preprocessing steps and scikit-learn models against tabular CSV datasets. It builds per-model pipelines, tracks metrics, saves artifacts (plots, feature importance, joblib dumps), and emits a Markdown report so you can review experiments like a lightweight AutoML system.

## Quickstart

```bash
python -m venv .venv && .\.venv\Scripts\activate  # PowerShell on Windows
pip install -e .
# (Optional) pip install pytest
autotab --config configs/demo_classification.yaml
autotab --config configs/demo_regression.yaml
```

### Example CLI Output

```
$ autotab --config configs/demo_classification.yaml
Successfully loaded config!
Problem Name: titanic_survival
Task Type: classification

Dataset Metadata:
  Rows: 250
  Columns: 10

Experiment Configuration:
  Models to train: 2
  Primary metric: f1_macro

Running experiment...
[OK] Experiment complete! Trained 2 models.
[OK] All artifacts saved to: outputs\titanic_survival_YYYYMMDD_HHMMSS

Leaderboard (sorted by f1_macro):
Rank  Model               Train Time (s)  f1_macro  accuracy  roc_auc
1     random_forest       0.612           0.86      0.90      0.92
2     logistic_regression 0.108           0.81      0.85      0.87
```

### Output Folder Preview

```
outputs/
  titanic_survival_20250101_123045/
    config.yaml
    metadata.json
    leaderboard.csv
    leaderboard.json
    report.md
    logistic_regression/
      model.joblib
      metrics.json
      classification_report.json
      confusion_matrix.png
    random_forest/
      model.joblib
      metrics.json
      feature_importance.csv
      feature_importance.png
```

## Demo Configs & Data

| Config | Dataset | Task | Notes |
| --- | --- | --- | --- |
| `configs/demo_classification.yaml` | `examples/data/titanic.csv` (synthetic Titanic-style manifest with 250 passengers) | Classification | Saves metrics, confusion matrix, feature importance plots |
| `configs/demo_regression.yaml` | `examples/data/housing.csv` (400 synthetic housing sales) | Regression | Includes residual plots and regression metrics |

## Running Tests

```
pip install pytest
pytest
```

The tests cover YAML config loading, dataset splitting, and an end-to-end smoke test that runs `run_experiment` on a synthetic mini dataset to ensure leaderboard generation stays healthy.
