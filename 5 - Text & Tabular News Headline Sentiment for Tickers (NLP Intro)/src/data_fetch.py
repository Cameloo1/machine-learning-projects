"""Utilities for downloading raw financial news headlines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

from .config import (
    LANGUAGE,
    MAX_ARTICLES_PER_TICKER,
    RAW_DATA_PATH,
    TICKERS,
)
from .manual_scrape import collect_manual_headlines

load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"


def _require_api_key() -> str:
    if not NEWS_API_KEY:
        raise RuntimeError(
            "NEWS_API_KEY not found. Add it to a .env file before fetching data."
        )
    return NEWS_API_KEY


def fetch_headlines_for_ticker(ticker: str) -> List[Dict[str, str]]:
    """Fetch recent headlines for a ticker from NewsAPI."""
    api_key = _require_api_key()
    page_size = min(60, MAX_ARTICLES_PER_TICKER)
    collected: List[Dict[str, str]] = []
    page = 1

    while len(collected) < MAX_ARTICLES_PER_TICKER:
        params = {
            "q": f"{ticker} stock",
            "language": LANGUAGE,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "page": page,
            "apiKey": api_key,
            "searchIn": "title,description",
        }
        response = requests.get(NEWS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
        if not articles:
            break

        for article in articles:
            headline = article.get("title") or ""
            if not headline:
                continue
            collected.append(
                {
                    "ticker": ticker,
                    "headline": headline.strip(),
                    "source": (article.get("source") or {}).get("name"),
                    "published_at": article.get("publishedAt"),
                    "url": article.get("url"),
                }
            )
            if len(collected) >= MAX_ARTICLES_PER_TICKER:
                break
        page += 1

    print(f"[data_fetch] {ticker}: fetched {len(collected)} headlines.")
    return collected


def _collect_via_api() -> pd.DataFrame:
    all_rows: List[Dict[str, str]] = []
    for ticker in TICKERS:
        rows = fetch_headlines_for_ticker(ticker)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("NewsAPI returned zero headlines.")
    df = pd.DataFrame(all_rows)
    return df


def collect_all_headlines(save_path: str | os.PathLike[str] = RAW_DATA_PATH) -> pd.DataFrame:
    """Collect headlines via API when possible, otherwise scrape manually."""
    force_manual = os.getenv("USE_MANUAL_SCRAPE", "0") == "1"
    df: pd.DataFrame | None = None

    if NEWS_API_KEY and not force_manual:
        try:
            df = _collect_via_api()
            print("[data_fetch] Using NewsAPI data.")
        except Exception as exc:  # noqa: BLE001
            print(f"[data_fetch] API collection failed: {exc}")
            df = None
    else:
        reason = "manual mode requested" if force_manual else "no NEWS_API_KEY found"
        print(f"[data_fetch] Skipping API collection ({reason}).")

    if df is None:
        print("[data_fetch] Falling back to manual scraping pipeline...")
        df = collect_manual_headlines()
        if df.empty:
            raise RuntimeError("Manual scraping returned no headlines.")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.dropna(subset=["headline"])
    df = df.drop_duplicates(subset=["headline"])
    df = df[df["headline"].str.len() >= 20]
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.sort_values("published_at", ascending=False)
    per_ticker_counts = df["ticker"].value_counts()

    df = df[["ticker", "headline", "source", "published_at", "url"]]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(
        f"[data_fetch] Saved {len(df)} cleaned headlines to {save_path} "
        f"({len(TICKERS)} tickers)."
    )
    print("[data_fetch] Per-ticker counts:")
    for ticker, count in per_ticker_counts.items():
        print(f"  - {ticker}: {count}")
    return df


