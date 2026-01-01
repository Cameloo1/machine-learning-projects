from __future__ import annotations

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "returns_v1"


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute close-to-close returns per symbol.

    Contract:
    - ret[t] = close[t] / close[t-1] - 1
    - output is long format with columns: ts, symbol, ret
    - timestamps are preserved and sorted (symbol, ts)
    """
    required = {"ts", "symbol", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"compute_returns missing columns: {sorted(missing)}")

    df = prices[["ts", "symbol", "close"]].copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

    df["ret"] = df.groupby("symbol", sort=False)["close"].pct_change()

    out = df[["ts", "symbol", "ret"]].copy()
    out = out.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)

    logger.info("Computed returns rows=%s symbols=%s", len(out), out["symbol"].nunique())
    return out


def summarize_returns(returns: pd.DataFrame) -> Dict[str, int]:
    """Summarize missingness and coverage for diagnostics."""
    return {
        "n_rows": int(len(returns)),
        "n_symbols": int(returns["symbol"].nunique()),
        "n_null_ret": int(returns["ret"].isna().sum()),
    }
