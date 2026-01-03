from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sample predictions CSV from prices.")
    parser.add_argument("--prices", required=True, help="Path to prices_long.csv")
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbols to include")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--start", default=None, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prices = pd.read_csv(args.prices)
    prices["ts"] = pd.to_datetime(prices["ts"], errors="coerce")
    symbols = {sym.strip() for sym in args.symbols.split(",") if sym.strip()}
    df = prices.loc[prices["symbol"].isin(symbols), ["ts", "symbol", "close"]].copy()

    if args.start:
        df = df.loc[df["ts"] >= pd.Timestamp(args.start)]
    if args.end:
        df = df.loc[df["ts"] <= pd.Timestamp(args.end)]

    df = df.sort_values(["symbol", "ts"], kind="mergesort")
    df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change()
    df["pred"] = df["ret"].apply(lambda val: 0.6 if val > 0 else 0.4)
    df.loc[df["ret"].isna(), "pred"] = 0.5

    out = df[["ts", "symbol", "pred"]].copy()
    out.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
