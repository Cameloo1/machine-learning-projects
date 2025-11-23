# Next-Day Realized Volatility Forecasting

This machine learning project attempts to forecast the **next-day realized volatility** of an asset (e.g., SPY) using historical price and volume data. This project demonstrates a strict time-series modeling discipline, avoiding common pitfalls like look-ahead bias, and compares ML models against a strong naive baseline.

## Project Overview

Volatility is a key metric in financial risk management. This project answers the question: **"Can we predict tomorrow's market turbulence better than simply assuming it will be the same as today?"**

The pipeline:
1.  Downloads daily data from **Yahoo Finance** (with an automatic **Stooq** fallback).
2.  Computes **Realized Volatility** using a rolling window of daily returns.
3.  Engineers lagged features (past returns, squared/abs returns, volume, and volatility).
4.  Trains **Linear Regression** and **Random Forest** models using a time-aware split.
5.  Evaluates performance against a "Same-As-Yesterday" baseline.

## Quick Start

### Prerequisites
Ensure you have Python installed along with the required libraries:

```bash
pip install numpy pandas yfinance matplotlib scikit-learn
```

### Running the Model
Simply execute the main script. It is self-contained and requires no arguments.

```bash
python vol_forecasting.py
```

The script will:
1.  Download data for `SPY` (S&P 500 ETF).
2.  Train models on data from 2015 to present.
3.  Output performance metrics (RMSE, MAE) to the console.
4.  Display a plot comparing predictions on the test set.

## Methodology

### 1. Data Pipeline
-   **Source**: Daily OHLCV data from Yahoo Finance (`yfinance`).
-   **Fallback**: If Yahoo fails, the system automatically attempts to pull CSV data from Stooq.
-   **Cleaning**: Data is sorted by date, and missing values are handled to ensure a continuous time series.

### 2. Feature Engineering
We predict **Tomorrow's Realized Volatility ($t+1$)** using information available at **Today ($t$)**.

-   **Target**: Rolling standard deviation of log-returns (window = 10 days), shifted by -1 day.
-   **Features**:
    -   **Lagged Returns**: Past 5 days of log returns (Directional).
    -   **Squared/Absolute Returns**: Past 5 days of magnitude proxies ($r^2$ and $|r|$).
    -   **Lagged Volume**: Past 5 days of trading volume.
    -   **Lagged Volatility**: Past 3 days of realized volatility (capturing *volatility clustering*).

### 3. Modeling Strategy
-   **Split**: Strict time-series split (First 80% = Train, Last 20% = Test). **No shuffling** to prevent data leakage.
-   **Baseline**: "Tomorrow's volatility = Today's volatility". This is the standard benchmark in financial time series.
-   **Models**:
    -   **Linear Regression**: Uses scaled features (`StandardScaler`).
    -   **Random Forest**: Non-linear model using 300 trees. Regularized with `min_samples_leaf=20` to prevent overfitting to market noise.

## Interpreting Results

Financial volatility is highly **autocorrelated** (it "clusters"). High volatility tends to be followed by high volatility, and low by low.

-   **Feature Importance**: You will likely see that `rv_lag_1` (yesterday's volatility) accounts for >90% of the model's predictive power.
-   **Success Metric**: A successful model must achieve a **Test RMSE** lower than the Baseline.
-   **Regularization**: We intentionally constrained the Random Forest model to ensure it generalizes well to unseen market regimes, rather than memorizing past noise. This results in a healthier Train/Test error balance, even if it struggles to beat the naive baseline in highly volatile periods.

## Model Performance Analysis

### Key Findings

**1. Volatility Clustering Dominates**
The Random Forest model reveals that **`rv_lag_1` (yesterday's volatility) accounts for >90% of predictive power**. This confirms the well-known financial phenomenon of volatility clustering: high volatility periods tend to persist, making the naive baseline ("tomorrow = today") an extremely strong benchmark.

**2. Simplicity vs. Complexity**
While the Random Forest captures non-linear relationships, the **Baseline ("Tomorrow = Today") remains a tough competitor**. In many test periods, the naive baseline outperforms complex models, demonstrating that simple heuristics can be more robust than ML in high-noise environments.

**3. Feature Insights**
- **Squared and Absolute Returns** can capture "shock" days (large outsized moves), contributing to the model's ability to forecast spikes in volatility even when directionality alone is insufficient.
- **Volatility Lags** are by far the most critical predictors.
- **Volume and Returns** provide marginal signal, often contributing <5% to the total importance after strong regularization.

## License
This project is open-source and available for educational purposes.
