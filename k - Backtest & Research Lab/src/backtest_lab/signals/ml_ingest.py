from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"ts", "symbol", "pred"}


def load_predictions(
    path: str | Path,
    *,
    ts_col: str = "ts",
    symbol_col: str = "symbol",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """
    Load model predictions in long format.

    Required output columns: ts, symbol, pred.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")

    raw = pd.read_csv(csv_path)
    mapping = {ts_col: "ts", symbol_col: "symbol", pred_col: "pred"}
    missing = [col for col in [ts_col, symbol_col, pred_col] if col not in raw.columns]
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")

    df = raw.rename(columns=mapping)[["ts", "symbol", "pred"]].copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if df["ts"].isna().any():
        raise ValueError("Predictions contain invalid timestamps")
    df["symbol"] = df["symbol"].astype(str).str.strip()
    if df["symbol"].eq("").any():
        raise ValueError("Predictions contain empty symbols")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    if df["pred"].isna().any():
        raise ValueError("Predictions contain non-numeric pred values")

    df = df.sort_values(["symbol", "ts"], kind="mergesort").reset_index(drop=True)
    logger.info("Loaded predictions %s rows=%s", csv_path, len(df))
    return df
