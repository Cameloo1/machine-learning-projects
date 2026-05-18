# Alert Triage Scorer

Security ML project that ranks synthetic SOC alerts into Low, Medium, and High priority classes with gradient boosting and SHAP explanations.

## Models

- XGBoost pipeline
- LightGBM pipeline

## Run

```bash
pip install -r requirements.txt
python -m src.data_generation --n_samples 6000
python -m src.train
python -m src.evaluate
python -m src.explain
python -m src.inference --mode csv --input data/raw/alerts_synthetic.csv --output artifacts/scored_alerts.csv --model_path models/xgb_pipeline.pkl
```

Fast local training check:

```bash
python -m src.data_generation --n_samples 1500
python -m src.train --search-iterations 5 --cv-folds 3
```

## Outputs

- raw and processed data under `data/`
- trained pipelines under `models/`
- metrics under `artifacts/metrics/`
- plots under `artifacts/plots/`
- SHAP outputs under `artifacts/shap/`
- supporting reports under `reports/`

## Current Snapshot

Saved metrics show XGBoost ahead of LightGBM on the committed synthetic dataset. See `reports/model_card.md` for the compact result summary.

## Reproducibility Notes

Synthetic data generation is deterministic when the configured seed is fixed. Model metrics can still vary slightly by library/platform version.
