import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import logging
import sys

# --------------------------------------------------
# 1) CONFIGURATION
# --------------------------------------------------

TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = None          # None = up to today
N_RETURNS = 5            # number of past daily returns used as features
N_VOL_WINDOW = 10        # rolling window length (in days) to compute realized volatility
N_VOL_LAGS = 3           # number of past realized vol values to use as features
TEST_SIZE_FRACTION = 0.2 # last 20% of samples used as test set
RANDOM_STATE = 42        # for reproducibility

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------
# 2) DATA DOWNLOAD (YFINANCE + STOOQ FALLBACK)
# --------------------------------------------------

def download_price_data(ticker: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    """
    Downloads daily price data from Yahoo Finance, falling back to Stooq if necessary.
    Returns a cleaned DataFrame with DatetimeIndex and at least 'Adj Close' and 'Volume'.
    """
    logging.info(f"Attempting to download data for {ticker} from Yahoo Finance...")
    
    df = pd.DataFrame()
    used_source = "yfinance"

    # --- Primary: Yahoo Finance ---
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Basic validation
        if df.empty:
            raise ValueError("Yahoo Finance returned empty DataFrame.")
        
        # Handle MultiIndex columns if present (common in newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Check required columns
        required_yf = ['Adj Close', 'Volume']
        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            logging.info("'Adj Close' not found, using 'Close' as proxy.")
            df['Adj Close'] = df['Close']
        
        missing_cols = [c for c in required_yf if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in Yahoo data: {missing_cols}")

        # Check row count sanity (arbitrary low threshold to detect failure)
        if len(df) < 100:
            raise ValueError(f"Yahoo Finance returned too few rows ({len(df)}).")

        logging.info(f"Successfully downloaded {len(df)} rows from Yahoo Finance.")

    except Exception as e:
        logging.warning(f"Yahoo Finance download failed or unusable: {e}")
        logging.info("Attempting fallback to Stooq...")
        used_source = "stooq"

        # --- Fallback: Stooq ---
        try:
            stooq_url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
            df = pd.read_csv(stooq_url)
            
            # Stooq Validation & Cleaning
            if df.empty:
                raise ValueError("Stooq returned empty DataFrame.")
            
            # Stooq columns are usually: Date, Open, High, Low, Close, Volume
            # Ensure 'Date' exists
            if 'Date' not in df.columns:
                raise ValueError("Stooq data missing 'Date' column.")
                
            # Preprocessing
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter Date Range
            if start_date:
                df = df.loc[df.index >= start_date]
            if end_date:
                df = df.loc[df.index <= end_date]
            
            # Clean NaNs in critical columns
            if 'Close' not in df.columns or 'Volume' not in df.columns:
                raise ValueError("Stooq data missing 'Close' or 'Volume'.")
                
            df.dropna(subset=['Close', 'Volume'], inplace=True)
            
            # Create Adj Close
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # Final check
            if df.empty:
                raise ValueError("Stooq DataFrame empty after filtering.")
                
            logging.info(f"Successfully downloaded {len(df)} rows from Stooq.")

        except Exception as stooq_e:
            logging.error(f"Stooq fallback also failed: {stooq_e}")
            logging.critical("Both data sources failed. Exiting.")
            sys.exit(1)

    # --- Final Clean-up (Common) ---
    # Keep only needed columns to avoid confusion
    df = df[['Adj Close', 'Volume']].copy()
    
    # Ensure index is DatetimeIndex and sorted
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Drop any remaining NaNs
    df.dropna(inplace=True)
    
    return df

def main():
    # --------------------------------------------------
    # 2) EXECUTE DATA DOWNLOAD
    # --------------------------------------------------
    data = download_price_data(TICKER, START_DATE, END_DATE)

    # --------------------------------------------------
    # 3) COMPUTE RETURNS & REALIZED VOLATILITY
    # --------------------------------------------------
    logging.info("Computing returns and realized volatility...")
    
    # Daily log returns
    data["ret"] = np.log(data["Adj Close"]).diff()
    
    # Additional return features (squared and absolute returns)
    data["ret_sq"] = data["ret"] ** 2
    data["ret_abs"] = data["ret"].abs()

    # Realized volatility (rolling std dev of returns)
    data["rv"] = data["ret"].rolling(N_VOL_WINDOW).std()
    
    # Drop initial NaNs from diff and rolling
    data.dropna(inplace=True)

    # --------------------------------------------------
    # 4) TARGET DEFINITION & FEATURE ENGINEERING
    # --------------------------------------------------
    logging.info("Building feature matrix...")
    
    # Target: Next-day realized volatility
    data["target_rv_next"] = data["rv"].shift(-1)
    
    # Features
    # a) Past returns
    for lag in range(1, N_RETURNS + 1):
        data[f"ret_lag_{lag}"] = data["ret"].shift(lag)

    # a2) Past squared returns
    for lag in range(1, N_RETURNS + 1):
        data[f"ret_sq_lag_{lag}"] = data["ret_sq"].shift(lag)

    # a3) Past absolute returns
    for lag in range(1, N_RETURNS + 1):
        data[f"ret_abs_lag_{lag}"] = data["ret_abs"].shift(lag)
        
    # b) Past volumes
    for lag in range(1, N_RETURNS + 1):
        data[f"vol_lag_{lag}"] = data["Volume"].shift(lag)
        
    # c) Past realized vol
    for lag in range(1, N_VOL_LAGS + 1):
        data[f"rv_lag_{lag}"] = data["rv"].shift(lag)
        
    # Drop rows with NaNs (due to shifting for features or target at the end)
    data.dropna(inplace=True)
    
    # Define X and y
    feature_cols = (
        [f"ret_lag_{lag}" for lag in range(1, N_RETURNS + 1)] +
        [f"ret_sq_lag_{lag}" for lag in range(1, N_RETURNS + 1)] +
        [f"ret_abs_lag_{lag}" for lag in range(1, N_RETURNS + 1)] +
        [f"vol_lag_{lag}" for lag in range(1, N_RETURNS + 1)] +
        [f"rv_lag_{lag}" for lag in range(1, N_VOL_LAGS + 1)]
    )
    
    X = data[feature_cols].values
    y = data["target_rv_next"].values
    dates = data.index
    
    # --------------------------------------------------
    # 5) TIME-AWARE TRAIN/TEST SPLIT
    # --------------------------------------------------
    n_samples = len(data)
    test_size = int(n_samples * TEST_SIZE_FRACTION)
    train_size = n_samples - test_size
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_train, dates_test = dates[:train_size], dates[train_size:]
    
    print("-" * 30)
    print(f"Total samples: {n_samples}")
    print(f"Train samples: {train_size}")
    print(f"Test samples:  {test_size}")
    print("-" * 30)

    # --------------------------------------------------
    # 6) BASELINE MODEL: "Tomorrow's vol = Today's vol"
    # --------------------------------------------------
    # rv_today corresponds to data["rv"] at time t
    # We are predicting target_rv_next (time t+1)
    rv_today = data["rv"].values
    
    baseline_pred_train = rv_today[:train_size]
    baseline_pred_test = rv_today[train_size:]
    
    # Evaluation metrics
    def evaluate_model(name, y_true_train, y_pred_train, y_true_test, y_pred_test):
        train_rmse = mean_squared_error(y_true_train, y_pred_train, squared=False)
        train_mae = mean_absolute_error(y_true_train, y_pred_train)
        test_rmse = mean_squared_error(y_true_test, y_pred_test, squared=False)
        test_mae = mean_absolute_error(y_true_test, y_pred_test)
        
        print(f"\n=== {name} ===")
        print(f"Train RMSE: {train_rmse:.6f}")
        print(f"Train MAE:  {train_mae:.6f}")
        print(f"Test  RMSE: {test_rmse:.6f}")
        print(f"Test  MAE:  {test_mae:.6f}")
        
        return test_rmse, test_mae

    evaluate_model("Baseline: 'tomorrow vol = today vol'", 
                   y_train, baseline_pred_train, 
                   y_test, baseline_pred_test)

    # --------------------------------------------------
    # 7) FEATURE SCALING
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------------------------------
    # 8) MODEL 1: LINEAR REGRESSION
    # --------------------------------------------------
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    
    lin_train_pred = lin_reg.predict(X_train_scaled)
    lin_test_pred = lin_reg.predict(X_test_scaled)
    
    evaluate_model("Linear Regression", 
                   y_train, lin_train_pred, 
                   y_test, lin_test_pred)

    # --------------------------------------------------
    # 9) MODEL 2: RANDOM FOREST REGRESSOR
    # --------------------------------------------------
    # Note: Tree-based models don't strictly need scaling, but using unscaled features here as requested
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    rf_train_pred = rf.predict(X_train)
    rf_test_pred = rf.predict(X_test)
    
    evaluate_model("Random Forest Regressor", 
                   y_train, rf_train_pred, 
                   y_test, rf_test_pred)

    # --------------------------------------------------
    # 10) VISUAL COMPARISON PLOT (TEST SET)
    # --------------------------------------------------
    logging.info("Generating comparison plot...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label="True next-day vol", color='black', linewidth=1.5, alpha=0.8)
    plt.plot(dates_test, baseline_pred_test, label="Baseline (today vol)", color='gray', linestyle='--', alpha=0.6)
    plt.plot(dates_test, lin_test_pred, label="LinearReg pred", color='blue', alpha=0.6)
    plt.plot(dates_test, rf_test_pred, label="RandomForest pred", color='green', alpha=0.6)
    
    plt.title(f"{TICKER} - Next-day Realized Vol Forecast (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Realized Volatility")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot as PNG
    plot_filename = f"{TICKER}_volatility_forecast_comparison.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved as {plot_filename}")
    
    plt.show()

    # --------------------------------------------------
    # 11) FEATURE IMPORTANCE (RANDOM FOREST)
    # --------------------------------------------------
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_k = 10

    print("\nTop feature importances (Random Forest):")
    for i in sorted_idx[:top_k]:
        print(f"{feature_cols[i]:<15} : {importances[i]:.4f}")

if __name__ == "__main__":
    main()

