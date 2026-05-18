# Next-Day Realized Volatility Forecasting

Regression project that forecasts next-day realized volatility for SPY and compares ML models against a strong naive baseline.

## Models

- same-as-yesterday volatility baseline
- Linear Regression
- Random Forest Regressor

## Method

- Downloads SPY daily OHLCV data.
- Computes rolling realized volatility from log returns.
- Builds lagged return, magnitude, volume, and volatility features.
- Uses a strict time-series split with no shuffling.
- Compares RMSE and MAE against the baseline.

## Run

```bash
pip install numpy pandas yfinance matplotlib scikit-learn
python vol_forecasting.py
```

## Output

- `SPY_volatility_forecast_comparison.png`
- console metrics for baseline and models

## Reproducibility Notes

The script is self-contained but network-backed. Results can change as market data updates.
