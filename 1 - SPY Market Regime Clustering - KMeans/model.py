"""
SPY Market Regime Clustering (Calm / Trending / Volatile)

Requirements:
    pip install yfinance pandas numpy scikit-learn matplotlib
"""

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent blocking
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def _period_to_days(period: str) -> int:
    """
    Convert period strings like '2y', '6mo', '30d' into an approximate day count.
    """
    if not period:
        raise ValueError("Period string cannot be empty.")

    num_part = "".join(ch for ch in period if ch.isdigit())
    unit = "".join(ch for ch in period if ch.isalpha()).lower()

    if not num_part or unit not in {"d", "mo", "y"}:
        raise ValueError(f"Unsupported period format: {period}")

    value = int(num_part)
    if unit == "d":
        return value
    if unit == "mo":
        return value * 30
    if unit == "y":
        return value * 365

    raise ValueError(f"Unsupported period unit: {unit}")


def _parse_cookie_header(cookie_header: str) -> dict[str, str]:
    """
    Convert a standard Cookie header string into a dictionary.
    """
    cookies: dict[str, str] = {}
    for chunk in cookie_header.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, value = chunk.split("=", 1)
        cookies[name.strip()] = value.strip()
    return cookies


def _build_custom_session() -> requests.Session | None:
    """
    Build a requests.Session if env vars provide Yahoo-compatible headers/cookies.
    Set YF_USER_AGENT and/or YF_COOKIE before running to enable this.
    """
    user_agent = os.getenv("YF_USER_AGENT")
    cookie_header = os.getenv("YF_COOKIE")

    if not user_agent and not cookie_header:
        return None

    session = requests.Session()
    if user_agent:
        session.headers["User-Agent"] = user_agent
    if cookie_header:
        session.cookies.update(_parse_cookie_header(cookie_header))
    return session


def _download_with_yfinance(
    ticker: str,
    period: str,
    interval: str,
    max_retries: int = 3,
) -> pd.DataFrame | None:
    """
    Try downloading data with yfinance, returning None if it ultimately fails.
    """
    session = _build_custom_session()

    for attempt in range(max_retries):
        try:
            ticker_obj = yf.Ticker(ticker, session=session)
            df = ticker_obj.history(period=period, interval=interval, auto_adjust=True)

            if df.empty:
                raise ValueError("yfinance returned an empty DataFrame.")

            df = df.reset_index()
            if "Date" not in df.columns and "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "Date"})
            return df
        except Exception as exc:  # Broad catch so we can fallback gracefully
            wait_time = 2 ** attempt
            print(f"yfinance attempt {attempt + 1} failed: {exc}. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    return None


def _download_with_stooq(ticker: str, period: str) -> pd.DataFrame:
    """
    Fallback download using the public Stooq API through pandas-datareader.
    """
    days = _period_to_days(period)
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=days)

    df = pdr.DataReader(ticker, "stooq", start=start, end=end)
    if df.empty:
        raise RuntimeError("Stooq returned no data.")

    df = df.sort_index().reset_index()
    df = df.rename(columns={"Date": "Date"})  # Explicit for clarity
    df["Date"] = pd.to_datetime(df["Date"])

    # Stooq returns auto-adjusted prices; align column order with expectations.
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return df


def download_price_data(
    ticker: str = "SPY",
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data with a resilient fallback strategy.
    """
    df = _download_with_yfinance(ticker=ticker, period=period, interval=interval)
    if df is None:
        print("Falling back to Stooq via pandas-datareader...")
        df = _download_with_stooq(ticker=ticker, period=period)

    if df.empty:
        raise RuntimeError("No data downloaded from any source.")

    df = df.rename_axis("Date").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add returns and volatility features for clustering.
    """
    df = df.copy()
    df.sort_values("Date", inplace=True)

    # 1-day return
    df["ret_1d"] = df["Close"].pct_change()

    # Rolling volatility of returns (5d & 20d)
    df["vol_5d"] = df["ret_1d"].rolling(window=5).std()
    df["vol_20d"] = df["ret_1d"].rolling(window=20).std()

    # Relative volume (vs 20d average)
    df["vol_rel"] = df["Volume"] / df["Volume"].rolling(window=20).mean()

    feature_cols = ["ret_1d", "vol_5d", "vol_20d", "vol_rel"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    return df


def fit_regime_model(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
):
    """
    Scale features, fit KMeans, attach numeric regime labels.
    Returns: df_with_regimes, scaler, kmeans.
    """
    feature_cols = ["ret_1d", "vol_5d", "vol_20d", "vol_rel"]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    regimes = kmeans.fit_predict(X_scaled)

    df = df.copy()
    df["regime"] = regimes

    return df, scaler, kmeans


def _label_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign regime labels (Calm / Trending / Volatile) to each cluster
    based on average volatility and return.

    Adds a 'regime_label' column and returns df and cluster_stats.
    """
    # Aggregate basic stats per regime
    cluster_stats = (
        df.groupby("regime")
        .agg(
            mean_ret=("ret_1d", "mean"),
            mean_vol5=("vol_5d", "mean"),
            mean_vol20=("vol_20d", "mean"),
            mean_vol_rel=("vol_rel", "mean"),
            count=("ret_1d", "size"),
        )
        .sort_index()
    )

    # Quantiles of volatility to define calm vs volatile
    q1 = cluster_stats["mean_vol20"].quantile(0.33)
    q3 = cluster_stats["mean_vol20"].quantile(0.66)

    regime_label_map = {}
    for regime, row in cluster_stats.iterrows():
        vol = row["mean_vol20"]
        ret = row["mean_ret"]

        if vol <= q1:
            label = "Calm"
        elif vol >= q3:
            # High volatility regimes
            if ret >= 0:
                label = "High Volatile Up"
            else:
                label = "High Volatile Down"
        else:
            # Middle vol regimes: call them trending up/down
            if ret > 0:
                label = "Trending Up"
            elif ret < 0:
                label = "Trending Down"
            else:
                label = "Sideways"

        regime_label_map[regime] = label

    df = df.copy()
    df["regime_label"] = df["regime"].map(regime_label_map)

    # Attach labels into stats DataFrame for printing
    cluster_stats["regime_label"] = cluster_stats.index.map(regime_label_map)

    return df, cluster_stats


def plot_price_with_regimes(df: pd.DataFrame, title_suffix: str = ""):
    """
    Plot SPY price with points colored by regime label.
    """
    df = df.copy()

    # Encode labels to integers for consistent coloring
    labels = pd.Categorical(df["regime_label"])
    label_codes = labels.codes
    unique_labels = list(labels.categories)

    plt.figure(figsize=(14, 6))
    plt.plot(df["Date"], df["Close"], linewidth=1.0, alpha=0.7, label="Close Price")

    # Scatter colored by regime
    scatter = plt.scatter(
        df["Date"],
        df["Close"],
        c=label_codes,
        cmap="tab10",
        s=15,
        alpha=0.9,
    )

    # Build legend from unique labels
    handles = []
    for i, label in enumerate(unique_labels):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab10(i / max(len(unique_labels) - 1, 1)),
                markersize=8,
                label=label,
            )
        )

    plt.legend(
        handles=handles,
        title="Regime",
        loc="upper left",
    )

    plt.title(f"SPY Price with Market Regimes {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    
    # Save plot instead of showing to prevent blocking
    filename = f"spy_price_regimes{title_suffix.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()


def plot_vol_scatter(df: pd.DataFrame, title_suffix: str = ""):
    """
    Scatter plot of 5d vs 20d volatility, colored by regime.
    """
    df = df.copy()
    labels = pd.Categorical(df["regime_label"])
    label_codes = labels.codes
    unique_labels = list(labels.categories)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["vol_5d"],
        df["vol_20d"],
        c=label_codes,
        cmap="tab10",
        alpha=0.8,
        s=15,
    )

    plt.xlabel("5-day volatility of returns")
    plt.ylabel("20-day volatility of returns")
    plt.title(f"Volatility Regimes (SPY) {title_suffix}")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Legend
    handles = []
    for i, label in enumerate(unique_labels):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab10(i / max(len(unique_labels) - 1, 1)),
                markersize=8,
                label=label,
            )
        )
    plt.legend(handles=handles, title="Regime", loc="upper left")

    plt.tight_layout()
    
    # Save plot instead of showing to prevent blocking
    filename = f"spy_vol_scatter{title_suffix.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close()


def print_cluster_summary(cluster_stats: pd.DataFrame):
    """
    Neatly print the per-cluster statistics.
    """
    print("\n=== Cluster / Regime Summary ===")
    print(cluster_stats.to_string(float_format=lambda x: f"{x:0.5f}"))


def save_regime_labels(df: pd.DataFrame, filename: str = "regime_labels.csv"):
    """
    Save a CSV for downstream ML containing Date, OHLCV, engineered features,
    numeric regime and regime_label. Date is ISO-formatted, no index.
    """
    df_out = df.copy()

    # Ensure Date column is datetime and formatted as ISO (YYYY-MM-DD)
    if "Date" in df_out.columns and not pd.api.types.is_datetime64_any_dtype(df_out["Date"]):
        df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")

    # Select columns for the CSV; fall back if some original columns are missing
    columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ret_1d",
        "vol_5d",
        "vol_20d",
        "vol_rel",
        "regime",
        "regime_label",
    ]
    existing_columns = [c for c in columns if c in df_out.columns]

    # Format Date to ISO (no time)
    if "Date" in existing_columns:
        df_out["Date"] = df_out["Date"].dt.strftime("%Y-%m-%d")

    # Write CSV
    df_out[existing_columns].to_csv(filename, index=False)
    print(f"Saved regime labels CSV to: {filename}")


def build_and_run(
    ticker: str = "SPY",
    period: str = "2y",
    n_clusters: int = 4,
):
    print(f"Downloading data for {ticker} over period={period}...")
    df = download_price_data(ticker=ticker, period=period)

    print("Engineering features...")
    df_feat = engineer_features(df)

    print(f"Fitting KMeans with n_clusters={n_clusters}...")
    df_reg, scaler, kmeans = fit_regime_model(df_feat, n_clusters=n_clusters)

    print("Labelling regimes...")
    df_reg, cluster_stats = _label_regimes(df_reg)

    print_cluster_summary(cluster_stats)

    # Latest few rows for sanity check
    print("\n=== Latest 10 days with regimes ===")
    print(
        df_reg[["Date", "Close", "ret_1d", "vol_5d", "vol_20d", "vol_rel", "regime", "regime_label"]]
        .tail(10)
        .to_string(index=False, float_format=lambda x: f"{x:0.5f}")
    )

    title_suffix = f"({ticker}, last {period})"
    print("\nPlotting price with regimes...")
    plot_price_with_regimes(df_reg, title_suffix=title_suffix)

    print("Plotting volatility scatter...")
    plot_vol_scatter(df_reg, title_suffix=title_suffix)

    # Save CSV for downstream ML
    csv_filename = "regime_labels.csv"
    print(f"\nSaving regime labels to `{csv_filename}` for downstream ML...")
    save_regime_labels(df_reg, filename=csv_filename)

    print("\nDone.")
    return df_reg, scaler, kmeans, cluster_stats


if __name__ == "__main__":
    # You can tweak these parameters for your LinkedIn run
    df_regimes, scaler, kmeans_model, stats = build_and_run(
        ticker="SPY",
        period="2y",
        n_clusters=4,
    )
