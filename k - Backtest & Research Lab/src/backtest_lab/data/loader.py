from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]
CANONICAL_COLS = ["ts", "symbol", "open", "high", "low", "close", "volume"]
DATE_COLS = ["date", "datetime", "timestamp", "ts"]


def _normalize_column_name(name: str) -> str:
    return name.strip().lower()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {col: _normalize_column_name(col) for col in df.columns}
    normalized = list(mapping.values())
    duplicates = sorted({col for col in normalized if normalized.count(col) > 1})
    if duplicates:
        raise ValueError(f"Duplicate columns after normalization: {duplicates}")
    return df.rename(columns=mapping)


def _find_date_column(columns: Sequence[str]) -> str:
    matches = [col for col in DATE_COLS if col in columns]
    if not matches:
        raise ValueError(
            f"Missing date column. Expected one of {DATE_COLS}. Columns found: {list(columns)}"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple date columns found: {matches}. Expected a single column.")
    return matches[0]


def _infer_symbol(path: Path) -> str:
    symbol = path.stem
    if not symbol:
        raise ValueError(f"Unable to infer symbol from path: {path}")
    return symbol


def _csv_has_required_cols(path: Path) -> bool:
    try:
        sample = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    cols = {_normalize_column_name(col) for col in sample.columns}
    return set(REQUIRED_COLS).issubset(cols) and any(col in cols for col in DATE_COLS)


def _standardize_prices_df(
    raw: pd.DataFrame, symbol: str | None, source: Path | None = None
) -> pd.DataFrame:
    df = _normalize_columns(raw)
    date_col = _find_date_column(df.columns)
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Columns found: {list(df.columns)}"
        )

    try:
        ts = pd.to_datetime(df[date_col], errors="raise")
    except Exception as exc:
        raise ValueError(f"Failed to parse timestamps in column '{date_col}': {exc}") from exc

    if symbol is None:
        if "symbol" in df.columns:
            symbol_series = df["symbol"].astype(str).str.strip()
        else:
            symbol = _infer_symbol(source) if source is not None else None
            if symbol is None:
                raise ValueError("Symbol column missing and no symbol provided")
            symbol_series = str(symbol)
    else:
        symbol_series = str(symbol)

    standardized = pd.DataFrame(
        {
            "ts": ts,
            "symbol": symbol_series,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    standardized = standardized[CANONICAL_COLS]

    source_label = str(source) if source is not None else "<in-memory>"
    if symbol is None:
        symbols = sorted({str(val) for val in standardized["symbol"].unique()})
        symbol_label = ",".join(symbols[:5]) + ("..." if len(symbols) > 5 else "")
    else:
        symbol_label = str(symbol)
    logger.info("Loaded source: %s symbol=%s rows=%s", source_label, symbol_label, len(standardized))
    if len(standardized) > 0:
        min_ts = standardized["ts"].min()
        max_ts = standardized["ts"].max()
    else:
        min_ts = None
        max_ts = None
    logger.info("Date range for %s: %s -> %s", symbol, min_ts, max_ts)
    missingness = standardized[REQUIRED_COLS].isna().sum().to_dict()
    missingness = {key: int(value) for key, value in missingness.items()}
    logger.info("Missing values for %s: %s", symbol, missingness)

    standardized = standardized.sort_values(["ts", "symbol"]).reset_index(drop=True)
    dup_mask = standardized.duplicated(subset=["ts", "symbol"])
    dup_count = int(dup_mask.sum())
    if dup_count:
        standardized = standardized.loc[~dup_mask].copy()
    logger.info("Dropped duplicate rows for %s: %s", symbol, dup_count)
    return standardized


def load_prices_from_csv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    """Load a single CSV file and return canonical long-format prices."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    return _standardize_prices_df(raw, symbol, source=csv_path)


def load_prices_from_csv_dir(dir_path: str | Path) -> pd.DataFrame:
    """Load all CSV files in a directory and return a concatenated long-format dataset."""
    csv_dir = Path(dir_path)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {csv_dir}")

    csv_files = sorted(
        [path for path in csv_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv"]
    )
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {csv_dir}")

    frames: List[pd.DataFrame] = []
    for csv_file in csv_files:
        if not _csv_has_required_cols(csv_file):
            logger.warning("Skipping non-price CSV file: %s", csv_file)
            continue
        frames.append(load_prices_from_csv(csv_file, symbol=_infer_symbol(csv_file)))

    if not frames:
        raise ValueError(f"No valid price CSV files found in directory: {csv_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ts", "symbol"]).reset_index(drop=True)
    dup_mask = combined.duplicated(subset=["ts", "symbol"])
    dup_count = int(dup_mask.sum())
    if dup_count:
        combined = combined.loc[~dup_mask].copy()
    logger.info("Dropped duplicate rows after concat: %s", dup_count)
    return combined[CANONICAL_COLS]


def load_prices_remote_yfinance(
    symbols: List[str],
    start: str,
    end: str,
    cache_dir: str | Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch prices from yfinance, cache them to CSV, then load via CSV loaders."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for remote loading (pip install yfinance).") from exc

    if not symbols:
        raise ValueError("symbols must be a non-empty list")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        symbol = str(symbol)
        cache_file = cache_path / f"{symbol}.csv"
        if cache_file.exists() and not refresh:
            logger.info("Using cached yfinance data for %s: %s", symbol, cache_file)
            continue

        data = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for symbol: {symbol}")
        data.reset_index(inplace=True)
        data.to_csv(cache_file, index=False)
        logger.info("Fetched yfinance data for %s rows=%s -> %s", symbol, len(data), cache_file)

    return load_prices_from_csv_dir(cache_path)


def load_prices(cfg: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = cfg["data"] if isinstance(cfg, dict) else getattr(cfg, "data", None)
    if data_cfg is None:
        raise ValueError("cfg.data is required to load prices")

    mode = data_cfg.get("mode") if isinstance(data_cfg, dict) else getattr(data_cfg, "mode", "csv")
    if mode == "csv":
        prices_path = data_cfg.get("prices_path") if isinstance(data_cfg, dict) else getattr(data_cfg, "prices_path", None)
        if prices_path is None:
            prices_path = data_cfg.get("path") if isinstance(data_cfg, dict) else getattr(data_cfg, "path", None)
        if prices_path is None:
            raise ValueError("data.prices_path is required when data.mode == 'csv'")
        path = Path(prices_path)
        if path.is_dir():
            return load_prices_from_csv_dir(path)
        return load_prices_from_csv(path)

    if mode == "yfinance":
        universe_cfg = cfg.get("universe") if isinstance(cfg, dict) else getattr(cfg, "universe", None)
        symbols = None
        if universe_cfg is not None:
            symbols = universe_cfg.get("symbols") if isinstance(universe_cfg, dict) else getattr(universe_cfg, "symbols", None)
        start = data_cfg.get("start") if isinstance(data_cfg, dict) else getattr(data_cfg, "start", None)
        end = data_cfg.get("end") if isinstance(data_cfg, dict) else getattr(data_cfg, "end", None)
        cache_dir = data_cfg.get("cache_dir") if isinstance(data_cfg, dict) else getattr(data_cfg, "cache_dir", "data/raw")
        if not symbols:
            raise ValueError("universe.symbols is required when data.mode == 'yfinance'")
        if start is None or end is None:
            raise ValueError("data.start and data.end are required when data.mode == 'yfinance'")
        return load_prices_remote_yfinance(list(symbols), start=start, end=end, cache_dir=cache_dir)

    if mode == "stooq":
        raise NotImplementedError("stooq loading is not implemented yet")

    raise ValueError(f"Unsupported data.mode '{mode}'")
