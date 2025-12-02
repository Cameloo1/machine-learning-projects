"""Download SPY intraday data for the RL Trading Sandbox.

This script downloads REAL market data from yfinance (primary) or stooq (fallback).
It does NOT generate synthetic data - only retrieves actual market data.

Note: yfinance has limitations:
- 30m data: only last 60 days available
- 1h data: only last 730 days (2 years) available
- For longer history, consider using daily data (interval='1d')

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --interval 1h
    python scripts/download_data.py --source stooq
    python scripts/download_data.py --interval 1d  # Daily data for longer history
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time

import pandas as pd


def download_from_yfinance(
    symbol: str = "SPY",
    interval: str = "30m",
    start_date: str = "2019-01-01",
    end_date: str | None = None,
) -> pd.DataFrame | None:
    """Download REAL market data from yfinance.
    
    Note: yfinance has limitations on intraday data:
    - 30m data: only last 60 days available
    - 1h data: only last 730 days (2 years) available
    - Daily data (1d): Full history available
    
    Args:
        symbol: Ticker symbol (default: SPY)
        interval: Bar interval ('30m', '1h', '1d')
        start_date: Start date string
        end_date: End date string (None = today)
    
    Returns:
        DataFrame with OHLCV data, or None if download fails.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("    ERROR: yfinance not installed. Install with: pip install yfinance")
        return None
    
    print(f"    Downloading {symbol} from yfinance...")
    print(f"    Interval: {interval}")
    
    end_dt = datetime.now() if end_date is None else pd.to_datetime(end_date)
    start_dt = pd.to_datetime(start_date)
    
    # yfinance intraday limitations
    if interval in ["30m", "15m", "5m", "1m"]:
        # Only last 60 days for sub-hourly data
        max_lookback = timedelta(days=59)
        if (end_dt - start_dt) > max_lookback:
            print(f"    WARNING: yfinance only provides {interval} data for last 60 days")
            print(f"    Requested range: {(end_dt - start_dt).days} days")
            print(f"    Adjusting start date to fit limitation...")
            start_dt = end_dt - max_lookback
            print(f"    Adjusted range: {start_dt.date()} to {end_dt.date()}")
    elif interval == "1h":
        # Only last 730 days for hourly data
        max_lookback = timedelta(days=729)
        if (end_dt - start_dt) > max_lookback:
            print(f"    WARNING: yfinance only provides {interval} data for last 730 days")
            print(f"    Requested range: {(end_dt - start_dt).days} days")
            print(f"    Adjusting start date to fit limitation...")
            start_dt = end_dt - max_lookback
            print(f"    Adjusted range: {start_dt.date()} to {end_dt.date()}")
    # Daily data (1d) has no such limitations
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval=interval,
        )
        
        if df.empty:
            print("    ERROR: No data returned from yfinance")
            print(f"    This may be due to:")
            print(f"      - Invalid date range")
            print(f"      - Market holidays/weekends")
            print(f"      - API rate limiting")
            return None
        
        # Standardize column names
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        # Rename datetime column to timestamp
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "timestamp"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        
        # Keep only required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        # Check if we have all required columns
        missing = set(required_cols) - set(available_cols)
        if missing:
            print(f"    WARNING: Missing columns: {missing}")
            # Try to continue if we have at least timestamp and close
            if "timestamp" not in df.columns or "close" not in df.columns:
                print("    ERROR: Missing critical columns (timestamp, close)")
                return None
        
        print(f"    Successfully downloaded {len(df)} {interval} bars")
        print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
        
    except Exception as e:
        print(f"    ERROR: yfinance download failed: {e}")
        return None


def download_from_stooq(
    symbol: str = "SPY.US",
    interval: str = "d",
    start_date: str = "2019-01-01",
    end_date: str | None = None,
) -> pd.DataFrame | None:
    """Download REAL daily data from Stooq.
    
    Note: Stooq only provides daily data for US stocks.
    Intraday data is NOT available from Stooq.
    
    Args:
        symbol: Ticker symbol with exchange suffix (e.g., 'SPY.US')
        interval: Must be 'd' for daily (stooq only supports daily)
        start_date: Start date string
        end_date: End date string
    
    Returns:
        DataFrame with OHLCV data, or None if download fails.
    """
    try:
        from pandas_datareader import data as pdr
    except ImportError:
        print("    ERROR: pandas_datareader not installed.")
        print("    Install with: pip install pandas-datareader")
        return None
    
    if interval != "d":
        print(f"    ERROR: Stooq only provides daily data, not {interval}")
        print("    Use --interval 1d for Stooq downloads")
        return None
    
    print(f"    Downloading {symbol} from Stooq (daily data only)...")
    
    end_dt = datetime.now() if end_date is None else pd.to_datetime(end_date)
    start_dt = pd.to_datetime(start_date)
    
    try:
        df = pdr.DataReader(symbol, "stooq", start=start_dt, end=end_dt)
        
        if df.empty:
            print("    ERROR: No data returned from Stooq")
            return None
        
        # Stooq returns data in reverse chronological order
        df = df.sort_index()
        df = df.reset_index()
        
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        
        # Rename date column to timestamp
        if "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        elif df.index.name and "date" in str(df.index.name).lower():
            # If index is a date, use it
            df = df.reset_index()
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
        
        # Keep only required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols]
        
        # Check for missing columns
        missing = set(required_cols) - set(available_cols)
        if missing:
            print(f"    WARNING: Missing columns: {missing}")
            if "timestamp" not in df.columns or "close" not in df.columns:
                print("    ERROR: Missing critical columns")
                return None
        
        print(f"    Successfully downloaded {len(df)} daily bars")
        print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
        
    except Exception as e:
        print(f"    ERROR: Stooq download failed: {e}")
        return None




def download_spy_data(
    output_path: str = "data/spy_30m_2019_2025.csv",
    interval: str = "30m",
    start_date: str = "2019-01-01",
    end_date: str | None = None,
    source: str = "auto",
) -> bool:
    """Download REAL SPY market data and save to CSV.
    
    This function ONLY retrieves real market data. It does NOT generate
    synthetic data. If real data cannot be obtained, the function fails.
    
    Args:
        output_path: Path to save the CSV file
        interval: Bar interval ('30m', '1h', '1d')
        start_date: Start date
        end_date: End date (None = today)
        source: Data source ('yfinance', 'stooq', 'auto')
    
    Returns:
        True if successful, False otherwise
    
    Raises:
        ValueError: If real data cannot be obtained for the requested interval.
    """
    print("=" * 60)
    print("SPY Real Market Data Download Script")
    print("=" * 60)
    print(f"\nTarget: {output_path}")
    print(f"Interval: {interval}")
    print(f"Date range: {start_date} to {end_date or 'now'}")
    print(f"Source: {source}")
    print("\nNOTE: This script only downloads REAL market data.")
    print("      Synthetic data generation is disabled.")
    
    df = None
    
    # Try yfinance first
    if source in ["auto", "yfinance"]:
        print("\n[1] Trying yfinance...")
        df = download_from_yfinance("SPY", interval, start_date, end_date)
        
        if df is not None:
            # Check if we got the requested interval
            if interval != "1d":
                # For intraday, verify we have reasonable data
                if len(df) == 0:
                    df = None
                else:
                    print(f"    Successfully downloaded {len(df)} {interval} bars from yfinance")
    
    # Fallback to stooq (only for daily data)
    if df is None and source in ["auto", "stooq"]:
        print("\n[2] Trying Stooq...")
        stooq_symbol = "SPY.US"
        
        # Stooq primarily provides daily data
        if interval == "1d":
            df = download_from_stooq(stooq_symbol, "d", start_date, end_date)
        else:
            print(f"    WARNING: Stooq does not provide {interval} intraday data.")
            print(f"    Stooq only provides daily data. Use --interval 1d for Stooq.")
            print(f"    For intraday data, yfinance is required (with limitations).")
    
    if df is None:
        print("\n" + "=" * 60)
        print("ERROR: Failed to download real market data")
        print("=" * 60)
        print("\nPossible reasons:")
        print("  1. yfinance limitations:")
        print("     - 30m data: only last 60 days available")
        print("     - 1h data: only last 730 days (2 years) available")
        print("  2. Network/API issues")
        print("  3. Invalid date range")
        print("\nSuggestions:")
        print("  - For longer history, use daily data: --interval 1d")
        print("  - Adjust date range to fit yfinance limitations")
        print("  - Check internet connection and try again")
        print("\nThis script does NOT generate synthetic data.")
        print("Only real market data is retrieved.")
        return False
    
    # Validate data
    print("\n[Validation]")
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"    ERROR: Missing columns: {missing}")
        return False
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Remove any NaN rows
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"    Dropped {initial_len - len(df)} rows with NaN values")
    
    # Summary
    print(f"    Total rows: {len(df)}")
    print(f"    Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"    Columns: {list(df.columns)}")
    
    # Save to CSV
    print(f"\n[Saving]")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"    Saved to: {output_path}")
    print(f"    File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download SPY intraday data for RL Trading Sandbox"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/spy_30m_2019_2025.csv",
        help="Output CSV path (default: data/spy_30m_2019_2025.csv)"
    )
    parser.add_argument(
        "--interval", "-i",
        default="30m",
        choices=["30m", "1h", "1d"],
        help="Bar interval (default: 30m)"
    )
    parser.add_argument(
        "--start", "-s",
        default="2019-01-01",
        help="Start date (default: 2019-01-01)"
    )
    parser.add_argument(
        "--end", "-e",
        default=None,
        help="End date (default: today)"
    )
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "yfinance", "stooq"],
        help="Data source (default: auto). Note: Only real market data is retrieved."
    )
    
    args = parser.parse_args()
    
    success = download_spy_data(
        output_path=args.output,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        source=args.source,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

