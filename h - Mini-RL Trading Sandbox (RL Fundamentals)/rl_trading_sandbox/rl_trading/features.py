"""Feature engineering pipeline for SPY intraday data.

This module provides functions for computing technical indicators and features
from OHLCV data, as well as utilities for feature normalization.
"""

import numpy as np
import pandas as pd


# Feature columns when using log returns (default)
FEATURE_COLUMNS_LOG = [
    # Raw / normalized price & volume context
    "ret_log_1",
    "price_norm",
    "vol_rel",
    # Volatility & regime risk
    "vol_20",
    "vol_100",
    "drawdown_pct",
    # Trend and slope information
    "ma_fast",
    "ma_slow",
    "ma_200_slope",
    "regime_fast",
    # Oscillator
    "rsi_14",
]

# Feature columns when using simple returns
FEATURE_COLUMNS_SIMPLE = [
    "ret_1",
    "price_norm",
    "vol_rel",
    "vol_20",
    "vol_100",
    "drawdown_pct",
    "ma_fast",
    "ma_slow",
    "ma_200_slope",
    "regime_fast",
    "rsi_14",
]


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) manually.
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over the period.
    
    Args:
        series: Price series (typically close prices).
        period: Lookback period for RSI calculation.
    
    Returns:
        RSI values as a pandas Series.
    """
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    # Use exponential moving average (Wilder's smoothing)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Handle edge case where avg_loss is 0
    rsi = rsi.replace([np.inf, -np.inf], 100.0)
    
    return rsi


def add_basic_features(
    df: pd.DataFrame, 
    use_log_returns: bool = True
) -> pd.DataFrame:
    """Add technical features to OHLCV DataFrame.
    
    Computes the following features:
    - ret_1 / ret_log_1: Simple & log returns
    - price_norm: Close divided by a 200-bar rolling mean
    - vol_rel: Volume divided by a 50-bar rolling mean
    - vol_20: 20-bar rolling volatility of returns
    - vol_100: 100-bar rolling volatility (regime risk)
    - drawdown_pct: Distance from 200-bar rolling max
    - ma_fast / ma_slow: 10- & 50-bar simple moving averages
    - ma_200_slope: Normalized slope of the 200-bar MA
    - regime_fast: Sign of (ma_fast - ma_slow)
    - rsi_14: 14-bar RSI oscillator
    
    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume.
        use_log_returns: If True, use log returns for volatility calc;
            otherwise use simple returns.
    
    Returns:
        New DataFrame with all original columns plus feature columns.
        Initial rows with NaN values from rolling windows are dropped.
    
    Raises:
        ValueError: If required columns are missing.
    
    Example:
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2020-01-01', periods=100, freq='30min'),
        ...     'open': np.random.randn(100).cumsum() + 300,
        ...     'high': np.random.randn(100).cumsum() + 301,
        ...     'low': np.random.randn(100).cumsum() + 299,
        ...     'close': np.random.randn(100).cumsum() + 300,
        ...     'volume': np.random.randint(1000, 10000, 100)
        ... })
        >>> df_feat = add_basic_features(df)
        >>> print(df_feat.columns.tolist())
    """
    required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    
    # Work on a copy and ensure sorted by timestamp
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Returns
    df["ret_1"] = df["close"].pct_change()
    df["ret_log_1"] = np.log(df["close"]).diff()
    
    # Normalized price (rescale to local mean over 200 bars)
    price_roll_mean = df["close"].rolling(window=200, min_periods=1).mean()
    df["price_norm"] = df["close"] / price_roll_mean
    
    # Relative volume (scale by 50-bar mean with min_periods=1 to avoid NaNs)
    vol_ma = df["volume"].rolling(window=50, min_periods=1).mean()
    df["vol_rel"] = df["volume"] / vol_ma
    
    # Volatility features
    return_col = "ret_log_1" if use_log_returns else "ret_1"
    df["vol_20"] = df[return_col].rolling(window=20, min_periods=20).std()
    df["vol_100"] = df[return_col].rolling(window=100, min_periods=20).std()
    
    # Moving averages
    df["ma_fast"] = df["close"].rolling(window=10, min_periods=10).mean()
    df["ma_slow"] = df["close"].rolling(window=50, min_periods=50).mean()
    df["ma_200"] = df["close"].rolling(window=200, min_periods=50).mean()
    
    # Simple regime / trend indicator: sign of fast MA minus slow MA
    # +1 when fast > slow (bullish), -1 when fast < slow (bearish), 0 when equal
    df["regime_fast"] = np.sign(df["ma_fast"] - df["ma_slow"]).astype(float)
    
    # Slow MA slope over last 20 bars, normalized by price to keep scale stable
    slope_window = 20
    df["ma_200_slope"] = (
        df["ma_200"] - df["ma_200"].shift(slope_window)
    ) / (slope_window * df["close"])
    
    # RSI
    df["rsi_14"] = _compute_rsi(df["close"], period=14)
    
    # Regime/drawdown context
    rolling_max = df["close"].rolling(window=200, min_periods=1).max()
    df["drawdown_pct"] = (df["close"] / rolling_max) - 1.0
    
    # Drop rows with NaN values (from rolling windows)
    df = df.dropna().reset_index(drop=True)
    
    return df


def get_feature_columns(use_log_returns: bool = True) -> list[str]:
    """Get the list of feature column names.
    
    Args:
        use_log_returns: If True, return columns using log returns;
            otherwise return columns using simple returns.
    
    Returns:
        List of feature column names in canonical order.
    
    Example:
        >>> cols = get_feature_columns(use_log_returns=True)
        >>> print(cols)
        ['ret_log_1', 'price_norm', 'vol_rel', 'vol_20', 'vol_100',
         'drawdown_pct', 'ma_fast', 'ma_slow', 'ma_200_slope', 'regime_fast',
         'rsi_14']
    """
    if use_log_returns:
        return FEATURE_COLUMNS_LOG.copy()
    return FEATURE_COLUMNS_SIMPLE.copy()


class FeatureScaler:
    """Z-score normalization for feature columns.
    
    Stores per-feature mean and standard deviation from training data,
    then applies the same transformation to validation/test data.
    This prevents data leakage from future data into past observations.
    
    Attributes:
        means: Dictionary of feature name to mean value.
        stds: Dictionary of feature name to standard deviation.
        fitted: Whether the scaler has been fit to data.
    
    Example:
        >>> scaler = FeatureScaler()
        >>> scaler.fit(train_df, ['ret_log_1', 'vol_20', 'rsi_14'])
        >>> train_scaled = scaler.transform(train_df, ['ret_log_1', 'vol_20', 'rsi_14'])
        >>> test_scaled = scaler.transform(test_df, ['ret_log_1', 'vol_20', 'rsi_14'])
    """
    
    def __init__(self) -> None:
        """Initialize the scaler with empty state."""
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.fitted: bool = False
    
    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Compute mean and std for each feature column.
        
        Args:
            df: DataFrame containing the feature columns.
            feature_cols: List of column names to fit.
        
        Raises:
            ValueError: If any feature column is missing from df.
        """
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing feature columns: {missing}")
        
        self.means = {}
        self.stds = {}
        
        for col in feature_cols:
            values = df[col].values.astype(np.float64)
            self.means[col] = float(np.mean(values))
            std = float(np.std(values))
            # Avoid division by zero
            self.stds[col] = std if std > 1e-8 else 1.0
        
        self.fitted = True
    
    def transform(
        self, 
        df: pd.DataFrame, 
        feature_cols: list[str]
    ) -> pd.DataFrame:
        """Apply z-score normalization using stored mean/std.
        
        Args:
            df: DataFrame containing the feature columns.
            feature_cols: List of column names to transform.
        
        Returns:
            New DataFrame with normalized feature columns.
            Non-feature columns are preserved unchanged.
        
        Raises:
            RuntimeError: If scaler has not been fitted.
            ValueError: If any feature column is missing.
        """
        if not self.fitted:
            raise RuntimeError(
                "FeatureScaler has not been fitted. Call fit() first."
            )
        
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing feature columns: {missing}")
        
        missing_fit = set(feature_cols) - set(self.means.keys())
        if missing_fit:
            raise ValueError(
                f"Feature columns not fitted: {missing_fit}. "
                f"Fitted columns: {list(self.means.keys())}"
            )
        
        df = df.copy()
        
        for col in feature_cols:
            df[col] = (df[col] - self.means[col]) / self.stds[col]
        
        return df
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        feature_cols: list[str]
    ) -> pd.DataFrame:
        """Fit the scaler and transform the data in one step.
        
        Args:
            df: DataFrame containing the feature columns.
            feature_cols: List of column names to fit and transform.
        
        Returns:
            New DataFrame with normalized feature columns.
        """
        self.fit(df, feature_cols)
        return self.transform(df, feature_cols)
    
    def get_params(self) -> dict[str, dict[str, float]]:
        """Get the fitted parameters.
        
        Returns:
            Dictionary with 'means' and 'stds' sub-dictionaries.
        
        Raises:
            RuntimeError: If scaler has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("FeatureScaler has not been fitted.")
        return {"means": self.means.copy(), "stds": self.stds.copy()}
    
    def __repr__(self) -> str:
        if self.fitted:
            return f"FeatureScaler(fitted=True, n_features={len(self.means)})"
        return "FeatureScaler(fitted=False)"


def prepare_features_for_split(
    df: pd.DataFrame,
    feature_scaler: FeatureScaler,
    feature_cols: list[str],
    fit_scaler: bool
) -> pd.DataFrame:
    """Prepare features for a data split with proper normalization.
    
    This helper ensures correct handling of train vs val/test splits:
    - For training data (fit_scaler=True): fit the scaler and transform
    - For val/test data (fit_scaler=False): only transform using fitted params
    
    Args:
        df: DataFrame with feature columns already computed.
        feature_scaler: FeatureScaler instance to use.
        feature_cols: List of feature column names to normalize.
        fit_scaler: If True, fit the scaler on this data; otherwise just transform.
    
    Returns:
        DataFrame with normalized feature columns.
    
    Example:
        >>> scaler = FeatureScaler()
        >>> feat_cols = get_feature_columns()
        >>> train_scaled = prepare_features_for_split(
        ...     train_df, scaler, feat_cols, fit_scaler=True
        ... )
        >>> val_scaled = prepare_features_for_split(
        ...     val_df, scaler, feat_cols, fit_scaler=False
        ... )
    """
    if fit_scaler:
        return feature_scaler.fit_transform(df, feature_cols)
    return feature_scaler.transform(df, feature_cols)

