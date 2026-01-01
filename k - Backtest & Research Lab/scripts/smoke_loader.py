from __future__ import annotations

import argparse
import logging

import pandas as pd

from backtest_lab.data.loader import load_prices_from_csv, load_prices_from_csv_dir


def _print_single(df: pd.DataFrame) -> None:
    print(df.head())
    print(df.dtypes)
    print(df.columns.tolist())


def _print_dir(df: pd.DataFrame) -> None:
    print(f"rows={len(df)} symbols={df['symbol'].nunique()}")
    ranges = df.groupby("symbol")["ts"].agg(["min", "max"]).head(10)
    print(ranges)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the CSV price loaders.")
    parser.add_argument("--file", dest="file_path", help="Path to a single CSV file.")
    parser.add_argument("--symbol", help="Override symbol for single-file load.")
    parser.add_argument("--dir", dest="dir_path", help="Path to a directory of CSV files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.file_path:
        df = load_prices_from_csv(args.file_path, symbol=args.symbol)
        _print_single(df)

    if args.dir_path:
        df = load_prices_from_csv_dir(args.dir_path)
        _print_dir(df)

    if not args.file_path and not args.dir_path:
        parser.error("Provide --file and/or --dir to run the smoke test.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
