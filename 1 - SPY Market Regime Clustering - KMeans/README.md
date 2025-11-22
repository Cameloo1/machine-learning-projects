# SPY Market Regime Clustering - KMeans

A machine learning project that identifies distinct market regimes in the SPY (S&P 500 ETF) using K-Means clustering. The model classifies market conditions into regimes such as Calm, Trending, and Volatile periods based on price returns and volatility patterns.

## Overview

This project analyzes historical SPY price data to automatically detect and label different market regimes. By clustering market behavior based on returns and volatility features, it provides insights into market conditions that can be useful for trading strategies, risk management, and market analysis.

## Methodology

1. **Data Download**: Automatically fetches SPY price data using yfinance with fallback to Stooq API
2. **Data Preprocessing**: Converts the data into a pandas DataFrame and sorts the data by date
3. **Scaling**: Scales the data to have a mean of 0 and a standard deviation of 1
4. **Feature Engineering**: Calculates 1-day returns, 5-day and 20-day rolling volatility, and relative volume
5. **K-Means Clustering**: Identifies distinct market regimes using unsupervised learning
6. **Regime Labeling**: Automatically labels clusters as:
   - **Calm**: Low volatility periods
   - **Trending Up/Down**: Moderate volatility with directional movement
   - **High Volatile Up/Down**: High volatility periods with positive/negative returns
   - **Sideways**: Moderate volatility with minimal directional movement
7. **Visualizations**: Generates two plots:
   - Price chart with regime-colored points
   - Volatility scatter plot (5-day vs 20-day volatility)

## Installation

1. Clone this repository or navigate to the project directory
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the model with default parameters (SPY, 2 years, 4 clusters):

```bash
python model.py
```

### Customization

You can modify the parameters in the `build_and_run()` function:

```python
df_regimes, scaler, kmeans_model, stats = build_and_run(
    ticker="SPY",      # Stock ticker symbol
    period="2y",       # Time period (e.g., "1y", "6mo", "30d")
    n_clusters=4,      # Number of clusters for K-Means
)
```

### Optional: Custom Yahoo Finance Session

If you encounter rate limiting issues with yfinance, you can set environment variables:

```bash
export YF_USER_AGENT="your-user-agent"
export YF_COOKIE="your-cookie-string"
```

## Output

The script generates:

1. **Console Output**:
   - Cluster/regime summary statistics
   - Latest 10 days with regime labels

2. **Visualizations** (saved as PNG files):
   - `spy_price_regimesSPY_last_2y.png`: Price chart with regime-colored points
   - `spy_vol_scatterSPY_last_2y.png`: Volatility scatter plot showing regime clusters

## Technical Details

### Features Used for Clustering

- `ret_1d`: 1-day return (percentage change)
- `vol_5d`: 5-day rolling standard deviation of returns
- `vol_20d`: 20-day rolling standard deviation of returns
- `vol_rel`: Relative volume (current volume / 20-day average volume)

## Requirements

- Python 3.7+
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
- pandas-datareader
- requests

## Notes

- The model uses a non-interactive matplotlib backend to prevent blocking during execution
- Data download includes automatic retry logic with exponential backoff
- The script saves plots instead of displaying them interactively

