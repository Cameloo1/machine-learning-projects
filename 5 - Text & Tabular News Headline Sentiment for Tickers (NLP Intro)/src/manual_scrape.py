"""Manual scraping fallback for collecting finance headlines.

Scraping should only be executed in accordance with each site's robots.txt
and terms of service. The functions below are lightweight, respectful
scrapers that add short delays and identify themselves via User-Agent.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List
from urllib.parse import urljoin, urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import TICKERS

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NewsSentimentBot/0.1; +https://example.com/bot)",
    "Accept-Language": "en-US,en;q=0.9",
}
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
BING_NEWS_URL = "https://www.bing.com/news/search"


def safe_get(url: str, params: dict | None = None, timeout: int = 10) -> str | None:
    """Wrapper around requests.get with basic error handling."""
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        LOGGER.warning("Request failed for %s: %s", url, exc)
        return None


def scrape_google_news(ticker: str, max_headlines: int = 50) -> List[Dict[str, str]]:
    """Scrape Google News RSS search results for a ticker."""
    params = {
        "q": f"{ticker} stock",
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    html = safe_get(GOOGLE_NEWS_RSS, params=params)
    if not html:
        return []

    soup = BeautifulSoup(html, "xml")
    rows: List[Dict[str, str]] = []
    for item in soup.find_all("item"):
        title_tag = item.find("title")
        link_tag = item.find("link")
        if not title_tag or not link_tag:
            continue
        headline = " ".join(title_tag.get_text(strip=True).split())
        if len(headline) < 5:
            continue

        source_tag = item.find("source")
        time_tag = item.find("pubDate")
        rows.append(
            {
                "ticker": ticker,
                "headline": headline,
                "source": source_tag.get_text(strip=True) if source_tag else "Google News",
                "published_at": time_tag.get_text(strip=True) if time_tag else "",
                "url": link_tag.get_text(strip=True),
            }
        )
        if len(rows) >= max_headlines:
            break
    LOGGER.info("[manual_scrape] Google News returned %s rows for %s", len(rows), ticker)
    time.sleep(1)
    return rows


def scrape_bing_news(ticker: str, max_headlines: int = 50) -> List[Dict[str, str]]:
    """Scrape Bing News search results for the ticker."""
    params = {
        "q": f"{ticker} stock",
        "setlang": "en-US",
        "form": "HDRSC6",
    }
    html = safe_get(BING_NEWS_URL, params=params)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    for item in soup.select("div.news-card, div.t_s"):
        link = item.find("a")
        title_tag = item.find("a", class_="title") or link
        if not link or not title_tag:
            continue
        headline = " ".join(title_tag.get_text(strip=True).split())
        if len(headline) < 5:
            continue
        source_tag = item.find("div", class_="source") or item.find("span", class_="source")
        time_tag = item.find("span", class_="time") or item.find("span", class_="pub")
        rows.append(
            {
                "ticker": ticker,
                "headline": headline,
                "source": source_tag.get_text(strip=True) if source_tag else "Bing News",
                "published_at": time_tag.get_text(strip=True) if time_tag else "",
                "url": urljoin("https://www.bing.com", link.get("href")),
            }
        )
        if len(rows) >= max_headlines:
            break
    LOGGER.info("[manual_scrape] Bing News returned %s rows for %s", len(rows), ticker)
    time.sleep(1)
    return rows


def clean_scraped_headlines(rows: List[Dict[str, str]], min_len: int = 20) -> pd.DataFrame:
    """Convert list of headline dicts into a filtered DataFrame."""
    if not rows:
        return pd.DataFrame(columns=["ticker", "headline", "source", "published_at", "url"])

    df = pd.DataFrame(rows, columns=["ticker", "headline", "source", "published_at", "url"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df.dropna(subset=["headline"])
    df = df[df["headline"].str.len() >= min_len]
    df = df.drop_duplicates(subset=["headline"])
    df = df.reset_index(drop=True)
    return df


def collect_manual_headlines() -> pd.DataFrame:
    """Aggregate manual scraping across configured tickers."""
    all_rows: List[Dict[str, str]] = []
    for ticker in TICKERS:
        rows = []
        rows.extend(scrape_google_news(ticker))
        rows.extend(scrape_bing_news(ticker))
        LOGGER.info("[manual_scrape] Combined %s rows for %s", len(rows), ticker)
        all_rows.extend(rows)

    df = clean_scraped_headlines(all_rows)
    LOGGER.info("[manual_scrape] Final scraped dataset size: %s", len(df))
    return df


