#!/usr/bin/env python
"""
Debug variant of the downloader with extensive logging for Yahoo issues.

Project integration:
- Point backtest configs to the canonical output file:
  data/processed/prices_long.csv (or .parquet when --out-format parquet).
- Schema: ts, symbol, open, high, low, close, volume.
- Timing assumption: daily bars (close-based); corporate actions are stored separately
  for Yahoo in data/raw/yahoo/{symbol}_actions.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    import pandas_datareader.data as pdr  # type: ignore
except Exception:
    pdr = None

try:
    import requests  # type: ignore
except Exception:
    requests = None


REQUIRED_COLUMNS = ["ts", "symbol", "open", "high", "low", "close", "volume"]
PRICE_VIOLATION_THRESHOLD = 0.001
RET_OUTLIER_THRESHOLD = 0.5


class DownloadError(RuntimeError):
    pass


class ValidationError(RuntimeError):
    pass


@dataclass
class SourceResult:
    source: str
    source_symbol: str
    raw_path: Path
    actions_path: Optional[Path]
    cached: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug download daily market data with extra logging."
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated tickers, e.g. SPY,QQQ,IWM",
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--out-format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format for canonical prices file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download raw files even if cached",
    )
    parser.add_argument("--min-rows", type=int, default=252)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between network calls",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--no-log-http",
        action="store_true",
        help="Disable HTTP response logging",
    )
    return parser.parse_args()


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("download_market_data_debug")
    for name in ["yfinance", "urllib3", "requests"]:
        logging.getLogger(name).setLevel(logging.DEBUG)
    return logger


def log_environment(logger: logging.Logger, session: Optional["requests.Session"]) -> None:
    yf_version = getattr(yf, "__version__", "not-installed") if yf is not None else "none"
    pdr_version = getattr(pdr, "__version__", "not-installed") if pdr is not None else "none"
    req_version = getattr(requests, "__version__", "not-installed") if requests is not None else "none"
    logger.info(
        "versions python=%s pandas=%s yfinance=%s pandas_datareader=%s requests=%s",
        sys.version.split()[0],
        pd.__version__,
        yf_version,
        pdr_version,
        req_version,
    )
    if session is not None:
        logger.debug("session headers=%s", dict(session.headers))


def build_http_session(
    logger: logging.Logger, log_http: bool
) -> Optional["requests.Session"]:
    if requests is None:
        logger.warning("requests not available; proceeding without custom headers")
        return None
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/json, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    if log_http:
        def _log_response(resp, *args, **kwargs):  # type: ignore[no-untyped-def]
            req = resp.request
            logger.debug(
                "http request method=%s url=%s headers=%s",
                req.method,
                req.url,
                dict(req.headers),
            )
            logger.debug(
                "http response url=%s status=%s headers=%s",
                resp.url,
                resp.status_code,
                dict(resp.headers),
            )
            content_type = resp.headers.get("Content-Type", "")
            if "json" in content_type or "text" in content_type:
                snippet = resp.text[:500].replace("\n", " ")
                logger.debug("http response snippet=%s", snippet)
            else:
                logger.debug("http response bytes=%s", len(resp.content))
            return resp

        session.hooks["response"] = [_log_response]
    return session


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def retry_call(
    func: Callable[[], Any],
    logger: logging.Logger,
    label: str,
    tries: int = 4,
    base_delay: float = 1.5,
) -> Any:
    last_exc: Optional[Exception] = None
    for attempt in range(tries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            delay = base_delay * (2**attempt)
            logger.warning(
                "retrying after error label=%s attempt=%s delay=%.2f err=%s",
                label,
                attempt + 1,
                delay,
                exc,
            )
            time.sleep(delay)
    raise DownloadError(f"failed after retries label={label} err={last_exc}")


def ensure_dirs() -> None:
    Path("data/raw/yahoo").mkdir(parents=True, exist_ok=True)
    Path("data/raw/stooq").mkdir(parents=True, exist_ok=True)
    Path("data/raw/alphavantage").mkdir(parents=True, exist_ok=True)
    Path("data/raw/tiingo").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/manifests").mkdir(parents=True, exist_ok=True)


def normalize_dates(df: pd.DataFrame, ts_col: str) -> pd.Series:
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    ts = ts.dt.tz_localize(None)
    ts = ts.dt.normalize()
    return ts


def validate_prices(
    df: pd.DataFrame,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_rows: int,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"missing required columns: {missing_cols}")

    df = df.copy()
    df["ts"] = normalize_dates(df, "ts")
    df = df.dropna(subset=["ts"])
    df = df[(df["ts"] >= start) & (df["ts"] <= end)]
    df = df.sort_values(["ts", "symbol"])

    dup_mask = df.duplicated(subset=["ts", "symbol"])
    dup_count = int(dup_mask.sum())
    if dup_count:
        df = df[~dup_mask]
        logger.warning("duplicates removed ticker=%s count=%s", symbol, dup_count)

    if not df["ts"].is_monotonic_increasing:
        raise ValidationError("timestamps not monotonic after sorting")

    if len(df) < min_rows:
        raise ValidationError(f"too few rows: {len(df)} < {min_rows}")

    missingness = df[REQUIRED_COLUMNS].isna().sum().to_dict()
    missingness = {k: int(v) for k, v in missingness.items()}

    price_cols = ["open", "high", "low", "close"]
    numeric = df[price_cols].apply(pd.to_numeric, errors="coerce")
    pos_violations = (numeric <= 0).sum().sum()

    high_violations = (
        (numeric["high"] < numeric[["open", "close"]].max(axis=1)).sum()
    )
    low_violations = (
        (numeric["low"] > numeric[["open", "close"]].min(axis=1)).sum()
    )

    total_violations = int(pos_violations + high_violations + low_violations)
    violation_pct = total_violations / max(len(df), 1)
    if violation_pct > PRICE_VIOLATION_THRESHOLD:
        raise ValidationError(
            f"price violations exceed threshold pct={violation_pct:.4f}"
        )

    expected_days = len(pd.bdate_range(start, end))
    observed_days = df["ts"].nunique()
    missing_days_pct = (
        float((expected_days - observed_days) / expected_days)
        if expected_days > 0
        else 0.0
    )

    ret = pd.to_numeric(df["close"], errors="coerce").pct_change()
    outliers = df.loc[ret.abs() > RET_OUTLIER_THRESHOLD, ["ts"]].copy()
    outliers["ret"] = ret[ret.abs() > RET_OUTLIER_THRESHOLD].values
    outliers["abs_ret"] = outliers["ret"].abs()
    outliers = outliers.sort_values("abs_ret", ascending=False).head(10)
    outlier_list = [
        {"ts": row.ts.strftime("%Y-%m-%d"), "ret": float(row.ret)}
        for row in outliers.itertuples(index=False)
    ]

    logger.info(
        "validation stats ticker=%s rows=%s missing_days_pct=%.4f missingness=%s",
        symbol,
        len(df),
        missing_days_pct,
        missingness,
    )
    if outlier_list:
        logger.info("outlier returns ticker=%s outliers=%s", symbol, outlier_list)

    stats = {
        "rows": int(len(df)),
        "duplicate_timestamps_removed": dup_count,
        "missingness": missingness,
        "missing_days_pct": missing_days_pct,
        "outlier_returns": outlier_list,
        "price_violation_pct": violation_pct,
    }
    return df, stats


def _resolve_ts_col(data: pd.DataFrame, preferred: str) -> str:
    if preferred in data.columns:
        return preferred
    for candidate in ["Date", "Datetime", "date", "datetime", "index", "ts", "Unnamed: 0"]:
        if candidate in data.columns:
            return candidate
    raise ValidationError("timestamp column missing")


def normalize_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    ts_col: str,
    col_map: Dict[str, str],
) -> pd.DataFrame:
    data = df.copy()
    if ts_col not in data.columns:
        data = data.reset_index()

    data = data.rename(columns=col_map)
    ts_source = _resolve_ts_col(data, ts_col)
    data["ts"] = normalize_dates(data, ts_source)
    data["symbol"] = symbol

    for col in ["open", "high", "low", "close", "volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "volume" in data.columns:
        vol = data["volume"]
        if vol.notna().all() and ((vol % 1) == 0).all():
            data["volume"] = vol.astype("int64")
        else:
            data["volume"] = vol.astype("float")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_cols:
        raise ValidationError(f"missing columns after normalization: {missing_cols}")

    data = data[REQUIRED_COLUMNS]
    data = data.sort_values(["ts", "symbol"])
    return data


def fetch_csv_url(
    url: str,
    session: Optional["requests.Session"],
    logger: logging.Logger,
    label: str,
    tries: int = 4,
    base_delay: float = 1.5,
) -> pd.DataFrame:
    if session is None:
        return pd.read_csv(url)

    def _fetch() -> pd.DataFrame:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

    return retry_call(_fetch, logger, label, tries=tries, base_delay=base_delay)


def debug_yahoo_chart_endpoint(
    symbol: str,
    start: str,
    end: str,
    session: Optional["requests.Session"],
    logger: logging.Logger,
) -> None:
    if session is None:
        logger.warning("no session for yahoo chart debug")
        return
    try:
        period1 = int(pd.Timestamp(start).timestamp())
        period2 = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?period1={period1}&period2={period2}"
            "&interval=1d&events=div,splits"
        )
        resp = session.get(url, timeout=30)
        logger.debug("yahoo chart status=%s url=%s", resp.status_code, resp.url)
        content_type = resp.headers.get("Content-Type", "")
        snippet = resp.text[:1000].replace("\n", " ")
        logger.debug("yahoo chart content-type=%s snippet=%s", content_type, snippet)
        try:
            payload = resp.json()
            chart = payload.get("chart", {})
            if chart.get("error"):
                logger.debug("yahoo chart error=%s", chart.get("error"))
            result = chart.get("result")
            if result:
                meta = result[0].get("meta", {})
                logger.debug("yahoo chart meta=%s", meta)
                indicators = result[0].get("indicators", {})
                quote = indicators.get("quote", [])
                if quote:
                    logger.debug("yahoo chart quote keys=%s", list(quote[0].keys()))
        except Exception as exc:
            logger.debug("yahoo chart json parse failed err=%s", exc)
    except Exception as exc:
        logger.warning("yahoo chart debug failed err=%s", exc)


def fetch_yahoo_chart_direct(
    symbol: str,
    start: str,
    end: str,
    session: Optional["requests.Session"],
    logger: logging.Logger,
) -> pd.DataFrame:
    if session is None:
        raise DownloadError("requests session unavailable for direct yahoo fetch")

    period1 = int(pd.Timestamp(start).timestamp())
    period2 = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits",
    }

    def _fetch() -> Dict[str, Any]:
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            raise DownloadError("yahoo chart endpoint rate limited (429)")
        resp.raise_for_status()
        return resp.json()

    payload = retry_call(_fetch, logger, f"yahoo-chart:{symbol}", tries=4, base_delay=2.0)
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise DownloadError(f"yahoo chart error: {chart.get('error')}")
    result = chart.get("result")
    if not result:
        raise DownloadError("yahoo chart empty result")

    res = result[0]
    timestamps = res.get("timestamp")
    indicators = res.get("indicators", {})
    quote = indicators.get("quote", [])
    if not timestamps or not quote:
        raise DownloadError("yahoo chart missing timestamps/quote")

    quote0 = quote[0]
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s"),
            "Open": quote0.get("open"),
            "High": quote0.get("high"),
            "Low": quote0.get("low"),
            "Close": quote0.get("close"),
            "Volume": quote0.get("volume"),
        }
    )
    return df


def download_yahoo(
    symbol: str,
    start: str,
    end: str,
    force: bool,
    sleep_s: float,
    logger: logging.Logger,
    session: Optional["requests.Session"],
) -> Tuple[pd.DataFrame, SourceResult]:
    raw_path = Path("data/raw/yahoo") / f"{symbol}.csv"
    actions_path = Path("data/raw/yahoo") / f"{symbol}_actions.csv"

    if raw_path.exists() and not force:
        logger.info("using cached yahoo data ticker=%s path=%s", symbol, raw_path)
        raw_df = pd.read_csv(raw_path)
        result = SourceResult("yahoo", symbol, raw_path, actions_path, True)
        return raw_df, result

    if yf is None:
        logger.warning("yfinance not installed, using direct chart fallback")
        raw_df = fetch_yahoo_chart_direct(symbol, start, end, session, logger)
        raw_df.to_csv(raw_path, index=False)
        return raw_df, SourceResult("yahoo", symbol, raw_path, actions_path, False)

    def _fetch() -> pd.DataFrame:
        return yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            actions=True,
            session=session,
        )

    logger.debug(
        "yfinance download params symbol=%s start=%s end=%s interval=1d auto_adjust=False",
        symbol,
        start,
        end,
    )
    try:
        started = time.time()
        raw_df = retry_call(_fetch, logger, f"yahoo:{symbol}")
        logger.debug("yfinance download elapsed=%.2fs", time.time() - started)
    except Exception as exc:
        logger.error("yfinance download failed symbol=%s err=%s", symbol, exc)
        debug_yahoo_chart_endpoint(symbol, start, end, session, logger)
        logger.warning("attempting direct yahoo chart fallback symbol=%s", symbol)
        raw_df = fetch_yahoo_chart_direct(symbol, start, end, session, logger)
        raw_df.to_csv(raw_path, index=False)
        return raw_df, SourceResult("yahoo", symbol, raw_path, actions_path, False)

    if raw_df is None or raw_df.empty:
        logger.error("yfinance returned empty frame symbol=%s", symbol)
        debug_yahoo_chart_endpoint(symbol, start, end, session, logger)
        logger.warning("attempting direct yahoo chart fallback symbol=%s", symbol)
        raw_df = fetch_yahoo_chart_direct(symbol, start, end, session, logger)
        raw_df.to_csv(raw_path, index=False)
        return raw_df, SourceResult("yahoo", symbol, raw_path, actions_path, False)
    logger.debug(
        "yfinance frame shape=%s index_type=%s columns=%s",
        raw_df.shape,
        type(raw_df.index).__name__,
        list(raw_df.columns),
    )
    logger.debug("yfinance head:\n%s", raw_df.head(5).to_string())
    if isinstance(raw_df.columns, pd.MultiIndex):
        if symbol in raw_df.columns.get_level_values(-1):
            raw_df = raw_df.xs(symbol, axis=1, level=-1)
        else:
            raw_df = raw_df.droplevel(-1, axis=1)
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(raw_df.columns)):
        logger.error("yfinance missing columns symbol=%s cols=%s", symbol, list(raw_df.columns))
        debug_yahoo_chart_endpoint(symbol, start, end, session, logger)
        raise DownloadError("missing expected yahoo columns")
    if raw_df[["Open", "High", "Low", "Close"]].dropna(how="all").empty:
        logger.error("yfinance all-NaN prices symbol=%s", symbol)
        debug_yahoo_chart_endpoint(symbol, start, end, session, logger)
        raise DownloadError("yahoo data all NaN")
    raw_df.to_csv(raw_path)

    if not actions_path.exists() or force:
        try:
            actions = retry_call(
                lambda: yf.Ticker(symbol, session=session).actions,
                logger,
                f"yahoo-actions:{symbol}",
            )
            if actions is not None and not actions.empty:
                logger.debug("yfinance actions rows=%s", len(actions))
                actions.to_csv(actions_path)
        except Exception as exc:
            logger.warning(
                "failed to download actions ticker=%s err=%s", symbol, exc
            )

    time.sleep(sleep_s)
    return raw_df, SourceResult("yahoo", symbol, raw_path, actions_path, False)


def normalize_yahoo(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw_df.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    return normalize_ohlcv(df, symbol, "Date", col_map)


def map_stooq_symbol(symbol: str) -> List[str]:
    if "." in symbol:
        return [symbol]
    return [f"{symbol}.US", symbol]


def download_stooq(
    symbol: str,
    start: str,
    end: str,
    force: bool,
    sleep_s: float,
    logger: logging.Logger,
    session: Optional["requests.Session"],
) -> Tuple[pd.DataFrame, SourceResult]:
    candidates = map_stooq_symbol(symbol)
    last_exc: Optional[Exception] = None
    for stooq_symbol in candidates:
        raw_path = Path("data/raw/stooq") / f"{stooq_symbol}.csv"
        if raw_path.exists() and not force:
            logger.info("using cached stooq data ticker=%s path=%s", symbol, raw_path)
            raw_df = pd.read_csv(raw_path)
            return raw_df, SourceResult("stooq", stooq_symbol, raw_path, None, True)

        try:
            if pdr is not None:
                def _fetch() -> pd.DataFrame:
                    try:
                        return pdr.DataReader(
                            stooq_symbol, "stooq", start, end, session=session
                        )
                    except TypeError:
                        return pdr.DataReader(stooq_symbol, "stooq", start, end)
                raw_df = retry_call(_fetch, logger, f"stooq:{stooq_symbol}")
                if raw_df is None or raw_df.empty:
                    raise DownloadError("empty stooq frame")
                raw_df.to_csv(raw_path)
            else:
                url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
                raw_df = fetch_csv_url(url, session, logger, f"stooq:{stooq_symbol}")
                if raw_df is None or raw_df.empty:
                    raise DownloadError("empty stooq frame")
                raw_df.to_csv(raw_path, index=False)

            time.sleep(sleep_s)
            return raw_df, SourceResult("stooq", stooq_symbol, raw_path, None, False)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "stooq attempt failed ticker=%s stooq_symbol=%s err=%s",
                symbol,
                stooq_symbol,
                exc,
            )
            continue

    raise DownloadError(f"stooq failed after candidates err={last_exc}")


def normalize_stooq(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw_df.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    return normalize_ohlcv(df, symbol, "Date", col_map)


def download_alphavantage(
    symbol: str,
    start: str,
    end: str,
    force: bool,
    sleep_s: float,
    logger: logging.Logger,
    session: Optional["requests.Session"],
) -> Tuple[pd.DataFrame, SourceResult]:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise DownloadError("alphavantage key not set")

    raw_path = Path("data/raw/alphavantage") / f"{symbol}.csv"
    if raw_path.exists() and not force:
        logger.info("using cached alphavantage data ticker=%s path=%s", symbol, raw_path)
        raw_df = pd.read_csv(raw_path)
        return raw_df, SourceResult("alphavantage", symbol, raw_path, None, True)

    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}&outputsize=full&apikey={api_key}&datatype=csv"
    )

    raw_df = fetch_csv_url(
        url, session, logger, f"alphavantage:{symbol}", tries=4, base_delay=2.0
    )
    if raw_df is None or raw_df.empty:
        raise DownloadError("empty alphavantage frame")
    raw_df.to_csv(raw_path, index=False)
    time.sleep(sleep_s)
    return raw_df, SourceResult("alphavantage", symbol, raw_path, None, False)


def normalize_alphavantage(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw_df.copy()
    col_map = {
        "timestamp": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=col_map)
    if "ts" not in df.columns:
        raise ValidationError("alphavantage timestamp column missing")
    return normalize_ohlcv(df, symbol, "ts", {})


def download_tiingo(
    symbol: str,
    start: str,
    end: str,
    force: bool,
    sleep_s: float,
    logger: logging.Logger,
    session: Optional["requests.Session"],
) -> Tuple[pd.DataFrame, SourceResult]:
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        raise DownloadError("tiingo key not set")

    raw_path = Path("data/raw/tiingo") / f"{symbol}.csv"
    if raw_path.exists() and not force:
        logger.info("using cached tiingo data ticker=%s path=%s", symbol, raw_path)
        raw_df = pd.read_csv(raw_path)
        return raw_df, SourceResult("tiingo", symbol, raw_path, None, True)

    url = (
        f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?"
        f"startDate={start}&endDate={end}&format=csv&token={api_key}"
    )

    raw_df = fetch_csv_url(
        url, session, logger, f"tiingo:{symbol}", tries=4, base_delay=2.0
    )
    if raw_df is None or raw_df.empty:
        raise DownloadError("empty tiingo frame")
    raw_df.to_csv(raw_path, index=False)
    time.sleep(sleep_s)
    return raw_df, SourceResult("tiingo", symbol, raw_path, None, False)


def normalize_tiingo(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw_df.copy()
    col_map = {
        "date": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=col_map)
    if "ts" not in df.columns:
        raise ValidationError("tiingo timestamp column missing")
    return normalize_ohlcv(df, symbol, "ts", {})


def process_ticker(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_rows: int,
    force: bool,
    sleep_s: float,
    logger: logging.Logger,
    session: Optional["requests.Session"],
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    attempts: List[Tuple[str, Callable[..., Tuple[pd.DataFrame, SourceResult]]]] = [
        ("yahoo", download_yahoo),
        ("stooq", download_stooq),
        ("alphavantage", download_alphavantage),
        ("tiingo", download_tiingo),
    ]

    for name, func in attempts:
        if name in {"alphavantage", "tiingo"}:
            env_key = "ALPHAVANTAGE_API_KEY" if name == "alphavantage" else "TIINGO_API_KEY"
            if not os.getenv(env_key):
                logger.info("skipping key-based fallback source=%s", name)
                continue
        try:
            raw_df, source_result = func(
                symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                force, sleep_s, logger, session
            )

            if source_result.source == "yahoo":
                norm = normalize_yahoo(raw_df, symbol)
            elif source_result.source == "stooq":
                norm = normalize_stooq(raw_df, symbol)
            elif source_result.source == "alphavantage":
                norm = normalize_alphavantage(raw_df, symbol)
            elif source_result.source == "tiingo":
                norm = normalize_tiingo(raw_df, symbol)
            else:
                raise ValidationError(f"unknown source {source_result.source}")

            cleaned, stats = validate_prices(
                norm, symbol, start, end, min_rows, logger
            )

            file_hash = sha256_file(source_result.raw_path)
            meta = {
                "status": "success",
                "source": source_result.source,
                "source_symbol": source_result.source_symbol,
                "raw_path": str(source_result.raw_path),
                "actions_path": str(source_result.actions_path)
                if source_result.actions_path
                else None,
                "cached": source_result.cached,
                "row_count": int(len(cleaned)),
                "date_coverage": {
                    "start": cleaned["ts"].min().strftime("%Y-%m-%d"),
                    "end": cleaned["ts"].max().strftime("%Y-%m-%d"),
                },
                "missingness": stats["missingness"],
                "duplicate_timestamps_removed": stats["duplicate_timestamps_removed"],
                "missing_days_pct": stats["missing_days_pct"],
                "outlier_returns": stats["outlier_returns"],
                "price_violation_pct": stats["price_violation_pct"],
                "sha256": file_hash,
            }
            return cleaned, meta
        except Exception as exc:
            logger.warning("source failed ticker=%s source=%s err=%s", symbol, name, exc)
            continue

    return None, {"status": "failed", "error": "all sources failed"}


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.log_level)
    ensure_dirs()
    session = build_http_session(logger, log_http=not args.no_log_http)
    log_environment(logger, session)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        logger.error("no tickers provided")
        return 1

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start > end:
        logger.error("start date after end date")
        return 1

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest_dir = Path("data/manifests") / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version.split()[0],
        "tickers": tickers,
        "start": args.start,
        "end": args.end,
        "results": {},
    }

    all_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for symbol in tickers:
        logger.info("processing ticker=%s", symbol)
        df, meta = process_ticker(
            symbol, start, end, args.min_rows, args.force, args.sleep, logger, session
        )
        manifest["results"][symbol] = meta

        if meta.get("status") == "success" and df is not None:
            all_frames.append(df)
            summary_rows.append(
                {
                    "symbol": symbol,
                    "status": "success",
                    "source": meta.get("source"),
                    "rows": meta.get("row_count"),
                    "start": meta.get("date_coverage", {}).get("start"),
                    "end": meta.get("date_coverage", {}).get("end"),
                    "missing_days_pct": meta.get("missing_days_pct"),
                }
            )
        else:
            summary_rows.append(
                {
                    "symbol": symbol,
                    "status": "failed",
                    "source": None,
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "missing_days_pct": None,
                }
            )

    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("data/processed/download_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    if not all_frames:
        logger.error("no tickers succeeded; see manifest at %s", manifest_path)
        return 1

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["ts", "symbol"])
    dup_count = int(combined.duplicated(subset=["ts", "symbol"]).sum())
    if dup_count:
        logger.error("combined output has duplicate (ts, symbol) count=%s", dup_count)
        return 1

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in combined.columns]
    if missing_cols:
        logger.error("combined output missing columns=%s", missing_cols)
        return 1

    out_path = Path("data/processed") / (
        "prices_long.parquet" if args.out_format == "parquet" else "prices_long.csv"
    )
    if args.out_format == "parquet":
        combined.to_parquet(out_path, index=False)
    else:
        combined.to_csv(out_path, index=False)

    succeeded = [s for s in tickers if manifest["results"][s]["status"] == "success"]
    failed = [s for s in tickers if manifest["results"][s]["status"] != "success"]
    source_map = {s: manifest["results"][s].get("source") for s in succeeded}

    logger.info(
        "download complete succeeded=%s failed=%s sources=%s output=%s",
        succeeded,
        failed,
        source_map,
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
