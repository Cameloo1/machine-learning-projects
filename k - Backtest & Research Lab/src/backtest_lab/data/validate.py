from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

logger = logging.getLogger(__name__)

DEFAULT_REQUIRED_COLS = ["ts", "symbol", "open", "high", "low", "close", "volume"]
SCHEMA_VERSION = "prices_validator_v1"


def _ts_to_str(value: Any) -> str:
    """
    Convert a timestamp value to a string representation.
    
    Handles various timestamp formats and edge cases, converting them to
    ISO format strings when possible. Returns "NaT" for null/missing values.
    
    Args:
        value: A timestamp value (can be pd.Timestamp, datetime, or other)
    
    Returns:
        String representation of the timestamp in ISO format, or "NaT" for null values
    """
    if value is None or pd.isna(value):
        return "NaT"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        return pd.to_datetime(value).isoformat()
    except Exception:
        return str(value)


def _sample_ts_symbol(
    ts_series: pd.Series, symbol_series: pd.Series, mask: pd.Series, limit: int = 5
) -> List[Dict[str, str]]:
    """
    Extract sample rows with timestamp and symbol information based on a boolean mask.
    
    Used to provide concrete examples of problematic rows in error messages.
    Applies the mask to both series and returns up to 'limit' samples.
    
    Args:
        ts_series: Series containing timestamp values
        symbol_series: Series containing symbol values
        mask: Boolean mask indicating which rows to sample
        limit: Maximum number of samples to return (default: 5)
    
    Returns:
        List of dictionaries, each containing 'ts' and 'symbol' keys with string values
    """
    samples: List[Dict[str, str]] = []
    for ts_val, sym_val in zip(ts_series[mask].head(limit), symbol_series[mask].head(limit)):
        samples.append({"ts": _ts_to_str(ts_val), "symbol": str(sym_val)})
    return samples


def _sample_monotonic_violations(
    symbol: str, ts_values: pd.Series, limit: int, out: List[Dict[str, str]]
) -> None:
    """
    Detect and sample timestamp monotonicity violations for a single symbol.
    
    Checks that timestamps are strictly increasing. When a violation is found
    (current timestamp <= previous timestamp), adds a sample to the output list.
    Stops early if the limit is reached.
    
    Args:
        symbol: The symbol being checked
        ts_values: Series of timestamp values, assumed to be sorted
        limit: Maximum number of violation samples to collect
        out: Output list to append violation samples to (modified in-place)
    """
    if len(out) >= limit:
        return
    values = ts_values.to_numpy()
    for idx in range(1, len(values)):
        if values[idx] <= values[idx - 1]:
            out.append(
                {
                    "symbol": str(symbol),
                    "prev_ts": _ts_to_str(values[idx - 1]),
                    "curr_ts": _ts_to_str(values[idx]),
                }
            )
            if len(out) >= limit:
                return


def validate_prices(
    prices: pd.DataFrame,
    *,
    required_cols: List[str] | None = None,
    duplicate_policy: str = "keep_first_after_sort",
    drop_rows_with_null_close: bool = True,
    strict: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and clean price data for backtesting.
    
    Performs comprehensive validation and cleaning of market price data:
    - Validates required columns exist
    - Converts and validates timestamps
    - Cleans and validates symbol names
    - Sorts data by symbol and timestamp
    - Removes duplicate (ts, symbol) pairs
    - Checks timestamp monotonicity within each symbol
    - Handles null close prices
    - Generates detailed diagnostics about data quality
    
    Args:
        prices: DataFrame containing price data with columns like ts, symbol, open, high, low, close, volume
        required_cols: List of column names that must be present (default: DEFAULT_REQUIRED_COLS)
        duplicate_policy: How to handle duplicate (ts, symbol) pairs (default: "keep_first_after_sort")
        drop_rows_with_null_close: Whether to drop rows with null close values (default: True)
        strict: If True, raises errors for data quality issues; if False, attempts to clean (default: True)
    
    Returns:
        Tuple containing:
        - Cleaned DataFrame sorted by symbol and timestamp
        - Dictionary of diagnostics with validation statistics and data quality metrics
    
    Raises:
        ValueError: If required columns are missing, timestamps are invalid, symbols are empty,
                   monotonicity violations are found, or other data quality issues are detected
    
    Example:
        >>> df = pd.DataFrame({
        ...     'ts': ['2024-01-01', '2024-01-02'],
        ...     'symbol': ['AAPL', 'AAPL'],
        ...     'open': [100, 101],
        ...     'high': [102, 103],
        ...     'low': [99, 100],
        ...     'close': [101, 102],
        ...     'volume': [1000, 1100]
        ... })
        >>> clean_df, diagnostics = validate_prices(df)
    """
    required = list(required_cols) if required_cols is not None else list(DEFAULT_REQUIRED_COLS)

    df = prices.copy()
    n_rows_in = int(len(df))
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. Columns found: {list(df.columns)}"
        )

    raw_ts = df["ts"].copy()
    if not is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    n_bad_ts = int(df["ts"].isna().sum())
    if n_bad_ts:
        samples = _sample_ts_symbol(raw_ts, df["symbol"], df["ts"].isna())
        raise ValueError(f"Found {n_bad_ts} invalid ts values. Samples: {samples}")

    symbol_raw = df["symbol"]
    symbol_clean = symbol_raw.astype(str).str.strip()
    empty_symbol_mask = symbol_raw.isna() | symbol_clean.eq("")
    n_empty_symbol = int(empty_symbol_mask.sum())
    if n_empty_symbol:
        samples = _sample_ts_symbol(df["ts"], symbol_raw, empty_symbol_mask)
        raise ValueError(f"Found {n_empty_symbol} empty symbols. Samples: {samples}")
    df["symbol"] = symbol_clean

    n_symbols_in = int(df["symbol"].nunique())

    df = df.sort_values(["symbol", "ts"], ascending=True, kind="mergesort").reset_index(drop=True)
    logger.info("Sorted prices by symbol and ts")

    if duplicate_policy != "keep_first_after_sort":
        raise ValueError(f"Unsupported duplicate_policy: {duplicate_policy}")

    key_counts = df.groupby(["ts", "symbol"], sort=False).size()
    dup_keys = key_counts[key_counts > 1]
    n_dup_keys = int(len(dup_keys))
    n_dup_rows_removed = int((dup_keys - 1).sum())
    dup_key_samples = [
        f"({_ts_to_str(ts)}, {symbol})" for ts, symbol in list(dup_keys.index)[:10]
    ]

    if n_dup_rows_removed:
        df = df.drop_duplicates(subset=["ts", "symbol"], keep="first").reset_index(drop=True)
    logger.info("Dropped duplicate rows: %s", n_dup_rows_removed)

    monotonic_violations: List[Dict[str, str]] = []
    for symbol, group in df.groupby("symbol", sort=False):
        _sample_monotonic_violations(symbol, group["ts"], limit=5, out=monotonic_violations)
        if len(monotonic_violations) >= 5:
            break
    if monotonic_violations:
        raise ValueError(
            f"Found {len(monotonic_violations)} monotonicity violations. Samples: {monotonic_violations}"
        )

    null_close_mask = df["close"].isna()
    n_null_close = int(null_close_mask.sum())
    n_rows_dropped_null_close = 0
    if n_null_close:
        if drop_rows_with_null_close:
            df = df.loc[~null_close_mask].reset_index(drop=True)
            n_rows_dropped_null_close = n_null_close
        elif strict:
            samples = _sample_ts_symbol(df["ts"], df["symbol"], null_close_mask)
            raise ValueError(
                f"Found {n_null_close} rows with null close values. Samples: {samples}"
            )
        else:
            df = df.loc[~null_close_mask].reset_index(drop=True)
            n_rows_dropped_null_close = n_null_close
    logger.info("Dropped rows with null close: %s", n_rows_dropped_null_close)

    missingness_global = {
        col: int(df[col].isna().sum()) for col in required if col in df.columns
    }
    missingness_by_symbol: List[Dict[str, Any]] = []
    for symbol, group in df.groupby("symbol", sort=True):
        missingness_by_symbol.append(
            {
                "symbol": str(symbol),
                "n_rows": int(len(group)),
                "null_open": int(group["open"].isna().sum()),
                "null_high": int(group["high"].isna().sum()),
                "null_low": int(group["low"].isna().sum()),
                "null_close": int(group["close"].isna().sum()),
                "null_volume": int(group["volume"].isna().sum()),
            }
        )

    ts_min = _ts_to_str(df["ts"].min()) if len(df) else None
    ts_max = _ts_to_str(df["ts"].max()) if len(df) else None

    n_rows_out = int(len(df))
    n_symbols_out = int(df["symbol"].nunique())

    diagnostics: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_rows_in": n_rows_in,
        "n_rows_out": n_rows_out,
        "n_symbols_in": n_symbols_in,
        "n_symbols_out": n_symbols_out,
        "n_bad_ts": n_bad_ts,
        "n_empty_symbol": n_empty_symbol,
        "duplicate_policy": duplicate_policy,
        "n_dup_keys": n_dup_keys,
        "n_dup_rows_removed": n_dup_rows_removed,
        "dup_key_samples": dup_key_samples,
        "n_null_close": n_null_close,
        "n_rows_dropped_null_close": n_rows_dropped_null_close,
        "missingness_global": missingness_global,
        "missingness_by_symbol": missingness_by_symbol,
        "ts_min": ts_min,
        "ts_max": ts_max,
    }

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. Columns found: {list(df.columns)}"
        )
    if df["ts"].isna().any():
        raise ValueError("Found NaT values in ts after validation")
    if df["symbol"].isna().any() or df["symbol"].astype(str).str.strip().eq("").any():
        raise ValueError("Found empty symbols after validation")
    if df["close"].isna().any():
        raise ValueError("Found null close values after validation")
    if df.duplicated(subset=["ts", "symbol"]).any():
        raise ValueError("Found duplicate (ts, symbol) after validation")
    sorted_df = df.sort_values(["symbol", "ts"], ascending=True, kind="mergesort")
    if not sorted_df[["symbol", "ts"]].reset_index(drop=True).equals(
        df[["symbol", "ts"]].reset_index(drop=True)
    ):
        raise ValueError("Output is not sorted by symbol then ts")

    return df, diagnostics
