# Market Regime Sequence Model

Sequence-modeling project for market-regime forecasting using engineered OHLCV features.

## Models

- Random Forest baseline
- LSTM
- 1D-CNN

## Data

- Market data is downloaded at runtime.
- Regime labels are expected at `data/regimes/regime_labels.csv`.
- If labels are missing, the pipeline can generate demonstration labels.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Outputs

- processed arrays under `data/processed/`
- model/scaler artifacts under `models/`
- metrics under `results/metrics/`
- plots under `results/plots/`
- explainability outputs under `results/explainability/`

## Reproducibility Notes

This is a heavier project than the tabular examples. Full reproduction depends on data availability, label quality, and deep-learning runtime. Use the Random Forest baseline as the first sanity check before relying on sequence-model results.
