# Short-Term Price Move Classifier

Supervised learning project that predicts whether SPY's next trading-day return is positive or negative.

## Models

- Logistic Regression baseline
- Random Forest
- Gradient Boosting
- majority-class baseline for comparison

## Method

- Pulls SPY OHLCV data from Yahoo Finance.
- Builds lagged returns, moving-average ratios, rolling volatility, relative volume, and RSI.
- Uses a time-based train/test split with no shuffling.
- Reports accuracy, ROC AUC, and comparison plots.

## Run

```bash
pip install -r requirements.txt
python -m short_term_price_classifier.run_experiment
```

Optional notebook:

```bash
jupyter notebook notebooks/short_term_price_classifier_demo.ipynb
```

## Outputs

- plots under `artifacts/plots/`
- metrics printed to the console

## Reproducibility Notes

This project depends on live market data, so exact metrics can drift. Treat it as a modeling demonstration, not a trading signal.
