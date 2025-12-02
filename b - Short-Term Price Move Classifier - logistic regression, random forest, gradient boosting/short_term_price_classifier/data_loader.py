"""Data loading utilities with hardened Yahoo Finance access."""

from __future__ import annotations

import io
import re
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus
from typing import Optional, Tuple

import pandas as pd  # type: ignore
import requests  # type: ignore
import yfinance as yf  # type: ignore
from requests import Session  # type: ignore
from requests.exceptions import RequestException  # type: ignore


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}
CRUMB_REGEX = re.compile(r'"CrumbStore":{"crumb":"(?P<crumb>[^"]+)"}')


def _decode_crumb(raw_crumb: str) -> str:
    """Convert escaped crumb strings into usable form."""

    decoded = raw_crumb.encode("utf-8").decode("unicode_escape")
    return decoded.replace("\\u002F", "/")


def _fetch_crumb(session: Session, timeout: float = 5.0) -> Optional[str]:
    """Hit Yahoo's crumb endpoint to retrieve the latest crumb token."""

    try:
        response = session.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=timeout)
        response.raise_for_status()
    except RequestException:
        return None
    crumb = response.text.strip()
    return _decode_crumb(crumb) if crumb else None


def _build_yahoo_session(ticker: str, timeout: float = 10.0) -> Tuple[Session, Optional[str]]:
    """Return a session and crumb token preloaded with Yahoo cookies and headers."""

    session = requests.Session()
    session.headers.update(YAHOO_HEADERS)

    quote_url = f"https://finance.yahoo.com/quote/{ticker}"
    response = session.get(quote_url, timeout=timeout)
    response.raise_for_status()

    crumb = _fetch_crumb(session)
    if crumb is None:
        match = CRUMB_REGEX.search(response.text)
        if match:
            crumb = _decode_crumb(match.group("crumb"))

    if crumb:
        session.cookies.set("crumb", crumb, domain=".yahoo.com")

    return session, crumb


def _start_to_unix(start: str) -> int:
    """Convert YYYY-MM-DD string to a UTC epoch timestamp."""

    dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _maybe_adjust_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Adjust OHLC columns using the Adj Close ratio to mimic auto_adjust=True."""

    if "Adj Close" not in df.columns or "Close" not in df.columns:
        return df

    with pd.option_context("mode.use_inf_as_na", True):
        adj_factor = df["Adj Close"] / df["Close"]

    for column in ["Open", "High", "Low", "Close"]:
        if column in df.columns:
            df[column] = df[column] * adj_factor

    return df.drop(columns=["Adj Close"], errors="ignore")


def _download_via_csv(
    session: Session,
    crumb: Optional[str],
    ticker: str,
    start: str,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fallback download via the Yahoo CSV endpoint using custom session headers."""

    if crumb is None:
        return pd.DataFrame()

    period1 = _start_to_unix(start)
    period2 = int(datetime.now(timezone.utc).timestamp())
    encoded_crumb = quote_plus(crumb)
    url = (
        f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history"
        f"&includeAdjustedClose=true&crumb={encoded_crumb}"
    )

    response = session.get(url, timeout=timeout, headers=YAHOO_HEADERS)
    response.raise_for_status()

    csv_buffer = io.StringIO(response.text)
    df = pd.read_csv(csv_buffer, parse_dates=["Date"], index_col="Date")
    df = _maybe_adjust_prices(df)
    return df


def _ticker_to_stooq_symbol(ticker: str) -> str:
    """Map a Yahoo-style ticker to its Stooq equivalent."""

    symbol = ticker.strip().lower()
    if "." in symbol:
        return symbol
    return f"{symbol}.us"


def _download_from_stooq(ticker: str, start: str, timeout: float = 30.0) -> pd.DataFrame:
    """Fallback downloader that fetches OHLCV data from Stooq."""

    symbol = _ticker_to_stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    response = requests.get(url, timeout=timeout, headers=YAHOO_HEADERS)
    response.raise_for_status()

    text_data = response.text.strip()
    if not text_data or "No data" in text_data:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(text_data), parse_dates=["Date"])
    if df.empty:
        return df

    df = df.set_index("Date").sort_index()
    df = df[df.index >= pd.Timestamp(start)]

    for column in REQUIRED_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df[REQUIRED_COLUMNS]


def load_ohlcv(
    ticker: str,
    start: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """Download OHLCV data for a ticker starting at a given date.

    Parameters
    ----------
    ticker:
        The ticker symbol understood by Yahoo Finance (e.g., "SPY").
    start:
        ISO formatted start date (YYYY-MM-DD).
    max_retries:
        Number of download attempts before failing.
    retry_delay:
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame
        Clean OHLCV data sorted in ascending date order.
    """

    last_error: Optional[Exception] = None
    session_ctx: Optional[Tuple[Session, Optional[str]]] = None
    df = pd.DataFrame()

    for attempt in range(max_retries):
        if session_ctx is None:
            try:
                session_ctx = _build_yahoo_session(ticker)
            except RequestException as exc:  # pragma: no cover - network errors
                last_error = exc
                session_ctx = None

        if session_ctx is None:
            time.sleep(retry_delay)
            continue

        try:
            df = yf.download(
                ticker,
                start=start,
                auto_adjust=True,
                progress=False,
                session=session_ctx[0],
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - network errors
            last_error = exc
            df = pd.DataFrame()
            session_ctx = None  # rebuild session next attempt

        if not df.empty:
            break

        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    if df.empty and session_ctx is not None:
        try:
            df = _download_via_csv(session_ctx[0], session_ctx[1], ticker, start)
        except Exception as exc:  # pragma: no cover - network errors
            last_error = exc
            df = pd.DataFrame()

    if df.empty:
        try:
            df = _download_from_stooq(ticker, start)
        except Exception as exc:  # pragma: no cover - network errors
            last_error = exc
            df = pd.DataFrame()

    if df.empty:
        msg = f"No data returned for ticker {ticker} starting {start}."
        if last_error is not None:
            msg += f" Last error: {last_error}"
        raise ValueError(msg)

    df = df.copy().sort_index()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    return df[REQUIRED_COLUMNS]
"""Data loading utilities."""


