import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("market_regime.features")

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Use exponential moving average for better accuracy (standard RSI)
    # gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    # loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    # Using simple rolling mean as per "implement manually... simple helper" prompt, 
    # but Wilder's smoothing is standard. Let's stick to simple rolling if not specified, 
    # or Wilder's. The prompt just says "RSI(14)". 
    # I'll use the Wilder's smoothing (ewm) as it's more correct for "RSI".
    
    gain = delta.where(delta > 0, 0).ewm(com=period - 1, min_periods=period).mean()
    loss = -delta.where(delta < 0, 0).ewm(com=period - 1, min_periods=period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series) -> pd.DataFrame:
    """Compute MACD line, Signal line, and MACD Histogram."""
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    return pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'macd_hist': macd_hist
    })

def engineer_features(input_path: str = "data/raw/spy_merged.csv", output_path: str = "data/features/features.csv") -> pd.DataFrame:
    """
    Generate technical indicators and financial features.
    """
    logger.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        logger.error(f"File not found: {input_path}")
        raise

    # Sort just in case
    df.sort_index(inplace=True)
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    logger.info("Computing features...")
    
    # Returns & Volatility
    df['ret_1d'] = close.pct_change()
    df['ret_5d'] = close.pct_change(5)
    
    df['vol_5d'] = df['ret_1d'].rolling(5).std()
    df['vol_20d'] = df['ret_1d'].rolling(20).std()
    df['vol_acc'] = df['vol_5d'] / df['vol_20d']
    df['range_vol'] = (high - low) / close
    
    # Volume features
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    
    df['vol_rel'] = volume / vol_mean_20
    df['vol_z'] = (volume - vol_mean_20) / vol_std_20
    
    # Trend / Momentum
    df['sma_10'] = close.rolling(10).mean()
    df['sma_20'] = close.rolling(20).mean()
    df['sma_50'] = close.rolling(50).mean()
    
    df['slope_10'] = df['sma_10'].pct_change()
    
    # Momentum burst (handle division by zero safely)
    # Replace 0 vol with NaN or small epsilon? Prompt says "safely".
    # We'll use a small epsilon or replace infs later.
    df['mom_burst'] = df['ret_1d'] / (df['vol_5d'] + 1e-9)
    
    # Market Structure
    df['rsi_14'] = compute_rsi(close, 14)
    
    macd_df = compute_macd(close)
    df = pd.concat([df, macd_df], axis=1)
    
    # Bollinger Bands
    std_20 = close.rolling(20).std()
    bb_upper = df['sma_20'] + 2 * std_20
    bb_lower = df['sma_20'] - 2 * std_20
    
    # BB Width
    # bb_width = (bb_upper - bb_lower) / sma_20
    # (sma + 2std - (sma - 2std)) / sma = 4std / sma
    df['bb_width'] = (bb_upper - bb_lower) / df['sma_20']
    
    # Data Cleaning
    # Drop initial rows where rolling is NaN
    # Max rolling window is 50 (sma_50), so we drop first 50
    logger.info("Cleaning data (dropping NaNs and Infs)...")
    df.dropna(inplace=True)
    
    # Ensure no infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Keep Date (index), Close, Volume for context if needed, but features are main goal
    # Prompt says "Keep the Date column and Close, Volume for context."
    # They are already in df.
    
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    df.to_csv(output_path)
    logger.info(f"Saved features to {output_path}")
    
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engineer_features()

