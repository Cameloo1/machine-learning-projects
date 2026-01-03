from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def _ensure_numeric(series: pd.Series, label: str) -> None:
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().any():
        n_bad = int(coerced.isna().sum())
        raise ValueError(f"{label} contains non-numeric values: {n_bad}")
    if not np.isfinite(coerced.to_numpy()).all():
        raise ValueError(f"{label} contains non-finite values")


def _ensure_datetime(series: pd.Series, label: str) -> None:
    ts = pd.to_datetime(series, errors="coerce")
    if ts.isna().any():
        n_bad = int(ts.isna().sum())
        raise ValueError(f"{label} contains invalid timestamps: {n_bad}")


def validate_returns_df(returns_df: pd.DataFrame) -> None:
    required = [
        "ts",
        "decision_ts",
        "gross",
        "net",
        "exposure",
        "turnover",
        "txn_cost",
        "slippage_cost",
        "costs",
    ]
    _ensure_columns(returns_df, required, "returns_df")

    _ensure_datetime(returns_df["ts"], "returns_df.ts")
    _ensure_datetime(returns_df["decision_ts"], "returns_df.decision_ts")

    for col in ["gross", "net", "exposure", "turnover", "txn_cost", "slippage_cost", "costs"]:
        _ensure_numeric(returns_df[col], f"returns_df.{col}")
        if returns_df[col].isna().any():
            raise ValueError(f"returns_df.{col} contains NaN values")

    if "window_id" in returns_df.columns:
        _ensure_numeric(returns_df["window_id"], "returns_df.window_id")
        key_cols = ["window_id", "ts"]
        sort_cols = ["window_id", "ts"]
    else:
        key_cols = ["ts"]
        sort_cols = ["ts"]

    if returns_df.duplicated(subset=key_cols).any():
        dup_count = int(returns_df.duplicated(subset=key_cols).sum())
        raise ValueError(f"returns_df has duplicate keys: {dup_count}")

    sorted_df = returns_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if not sorted_df[sort_cols].equals(returns_df[sort_cols].reset_index(drop=True)):
        raise ValueError("returns_df is not sorted by expected keys")


def validate_output_weights_df(weights_df: pd.DataFrame) -> None:
    required = ["ts", "symbol", "weight"]
    _ensure_columns(weights_df, required, "weights_df")
    _ensure_datetime(weights_df["ts"], "weights_df.ts")
    if weights_df["symbol"].isna().any() or weights_df["symbol"].astype(str).str.strip().eq("").any():
        raise ValueError("weights_df.symbol contains empty values")
    _ensure_numeric(weights_df["weight"], "weights_df.weight")
    if weights_df["weight"].isna().any():
        raise ValueError("weights_df.weight contains NaN values")

    if "window_id" in weights_df.columns:
        _ensure_numeric(weights_df["window_id"], "weights_df.window_id")
        key_cols = ["window_id", "ts", "symbol"]
        sort_cols = ["window_id", "ts", "symbol"]
    else:
        key_cols = ["ts", "symbol"]
        sort_cols = ["ts", "symbol"]

    if weights_df.duplicated(subset=key_cols).any():
        dup_count = int(weights_df.duplicated(subset=key_cols).sum())
        raise ValueError(f"weights_df has duplicate keys: {dup_count}")

    sorted_df = weights_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if not sorted_df[sort_cols].equals(weights_df[sort_cols].reset_index(drop=True)):
        raise ValueError("weights_df is not sorted by expected keys")


def validate_trades_df(trades_df: pd.DataFrame) -> None:
    required = [
        "ts",
        "symbol",
        "weight",
        "dw",
        "abs_dw",
        "txn_cost",
        "slippage_cost",
        "cost",
    ]
    _ensure_columns(trades_df, required, "trades_df")

    _ensure_datetime(trades_df["ts"], "trades_df.ts")
    if trades_df["symbol"].isna().any() or trades_df["symbol"].astype(str).str.strip().eq("").any():
        raise ValueError("trades_df.symbol contains empty values")
    for col in ["weight", "dw", "abs_dw", "txn_cost", "slippage_cost", "cost"]:
        _ensure_numeric(trades_df[col], f"trades_df.{col}")
        if trades_df[col].isna().any():
            raise ValueError(f"trades_df.{col} contains NaN values")

    if "window_id" in trades_df.columns:
        _ensure_numeric(trades_df["window_id"], "trades_df.window_id")
        key_cols = ["window_id", "ts", "symbol"]
        sort_cols = ["window_id", "ts", "symbol"]
    else:
        key_cols = ["ts", "symbol"]
        sort_cols = ["ts", "symbol"]

    if trades_df.duplicated(subset=key_cols).any():
        dup_count = int(trades_df.duplicated(subset=key_cols).sum())
        raise ValueError(f"trades_df has duplicate keys: {dup_count}")

    sorted_df = trades_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if not sorted_df[sort_cols].equals(trades_df[sort_cols].reset_index(drop=True)):
        raise ValueError("trades_df is not sorted by expected keys")
