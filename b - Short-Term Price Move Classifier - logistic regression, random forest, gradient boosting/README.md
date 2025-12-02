# Short-Term Price Move Classifier

Predict whether the next trading day's return for a liquid asset (default: SPY) will be positive or negative using compact technical features and lightweight supervised models.

## Problem Statement

- **Objective:** Classify next-day up/down moves for SPY using only past information.
- **Label:** `1` if `Close[t+1] > Close[t]`, else `0`.
- **Data Source:** Yahoo Finance via `yfinance`, auto-adjusted OHLCV starting `2015-01-01`.

## Methodology

- **Features:** Lagged returns (`ret_lag_1..5`), moving-average ratios (`ma_5_ratio`, `ma_10_ratio`), rolling volatility (`vol_10d`, `vol_20d`), relative volume (`vol_rel_20d`), and RSI (`rsi_14`). All features rely solely on information available at time *t*.
- **Models:** Logistic Regression (baseline), Random Forest, and Gradient Boosting from `scikit-learn`.
- **Baseline:** Always predict the majority class observed in the training window.
- **Validation:** Strict time-based split (`70%` train / `30%` test) with no shuffling to avoid leakage.

## Results (sample)

Model metrics will vary with the evaluation window, but expect accuracy and ROC AUC to be slightly above the majority baseline (which typically lands near 0.52–0.54 accuracy, ~0.50 ROC AUC). Update this section after running the experiment on your machine.

## Limitations

- Single asset (SPY) and single-step horizon.
- Ignores frictions such as transaction costs and slippage.
- Does not include position sizing, risk management, or portfolio logic.
- Demonstration-only — **not** financial advice or an investible trading strategy.

## Getting Started

```bash
pip install -r requirements.txt
python -m short_term_price_classifier.run_experiment
```

The script will download data, engineer features, fit all models, print metrics, and display ROC / importance plots.

### Notebook

Launch the guided walkthrough notebook:

```bash
jupyter notebook notebooks/short_term_price_classifier_demo.ipynb
```

## Repository Layout

- `short_term_price_classifier/`: Reusable package with configuration, data loading, feature engineering, modeling, evaluation helpers, and the `run_experiment` pipeline.
- `notebooks/short_term_price_classifier_demo.ipynb`: Narrative exploration that reuses the package functions.
- `requirements.txt`: Minimal dependency list.
- `README.md`: You are here.

