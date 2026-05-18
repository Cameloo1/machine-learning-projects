# SPY Market Regime Clustering

KMeans clustering project that labels recent SPY market regimes using daily returns, rolling volatility, and relative volume.

## What It Does

- Downloads SPY OHLCV data with `yfinance` and a Stooq fallback.
- Builds return, volatility, and volume features.
- Fits KMeans clusters.
- Labels clusters into readable market-regime buckets.
- Writes two PNG plots for review.

## Run

```bash
pip install -r requirements.txt
python model.py
```

## Outputs

- `spy_price_regimesSPY_last_2y.png`
- `spy_vol_scatterSPY_last_2y.png`
- console summary of recent regime labels and cluster statistics

## Reproducibility Notes

This project depends on live market data, so exact outputs can change over time. Use the root verifier for structure checks:

```bash
python scripts\verify_projects.py --project a --level quick
```
