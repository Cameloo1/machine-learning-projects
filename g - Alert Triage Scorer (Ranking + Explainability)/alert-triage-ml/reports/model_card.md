# Alert Triage Model Card

## Purpose

Classify synthetic SOC alerts into `Low`, `Medium`, or `High` priority for triage workflow experiments.

## Data

- Source: generated synthetic alert data.
- Current raw input: `data/raw/alerts_synthetic.csv`.
- Splits: `data/processed/train.csv`, `val.csv`, and `test.csv`.

This is not production telemetry.

## Models

- XGBoost pipeline: `models/xgb_pipeline.pkl`
- LightGBM pipeline: `models/lgbm_pipeline.pkl`

Both models use the project preprocessing pipeline and are evaluated on the held-out test split.

## Current Saved Metrics

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| XGBoost | 0.901 | 0.882 |
| LightGBM | 0.873 | 0.852 |

Metric files:

- `artifacts/metrics/xgb_metrics.json`
- `artifacts/metrics/lgbm_metrics.json`

## Intended Use

- Portfolio/demo evidence for SOC triage modeling.
- Local testing of explainability and inference utilities.
- Baseline for future workflow simulations.

## Not Intended For

- Production alert prioritization.
- Compliance decisions.
- Real incident response without validation on real local telemetry.

## Main Limitations

- Synthetic data may not capture real SOC drift, label noise, or adversarial behavior.
- Metrics are only meaningful for the generated dataset.
- SHAP explanations describe model behavior, not ground truth causality.

## Reproduce

```bash
python -m src.data_generation --n_samples 6000
python -m src.train
python -m src.evaluate
python -m src.explain
```
