from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def _parse_ts(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)


def _hash_assets(assets: List[str]) -> str:
    payload = "\n".join(assets).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_universe(
    prices: pd.DataFrame,
    start_ts: Any,
    end_ts: Any,
    min_history: int,
    policy: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = prices.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

    start = _parse_ts(start_ts) if start_ts is not None else df["ts"].min()
    end = _parse_ts(end_ts) if end_ts is not None else df["ts"].max()
    if start is pd.NaT or end is pd.NaT:
        raise ValueError("start_ts/end_ts could not be parsed to timestamps")
    if start > end:
        raise ValueError("start_ts must be <= end_ts")

    window = df.loc[(df["ts"] >= start) & (df["ts"] <= end)].copy()
    window = window.sort_values(["symbol", "ts"]).reset_index(drop=True)

    coverage_by_symbol: List[Dict[str, Any]] = []
    rows_dropped_missing = 0

    complete_mask = window[REQUIRED_COLS].notna().all(axis=1)
    if policy == "drop_rows":
        rows_dropped_missing = int((~complete_mask).sum())
        window_after = window.loc[complete_mask].copy()
    else:
        window_after = window

    counts_after = (
        window_after.groupby("symbol", sort=True).size().to_dict() if len(window_after) else {}
    )

    eligible_symbols: List[str] = []
    dropped_symbols: List[str] = []

    for symbol, group in window.groupby("symbol", sort=True):
        total_rows = int(len(group))
        complete_rows = int(group[REQUIRED_COLS].notna().all(axis=1).sum())
        missing_rows = total_rows - complete_rows
        rows_after = int(counts_after.get(symbol, 0))

        min_history_ok = rows_after >= int(min_history)
        missing_ok = missing_rows == 0

        eligible = False
        drop_reason = None
        if policy == "drop_symbol":
            eligible = min_history_ok and missing_ok
            if not min_history_ok:
                drop_reason = "min_history"
            elif not missing_ok:
                drop_reason = "missing_data"
        elif policy == "drop_rows":
            eligible = min_history_ok
            if not min_history_ok:
                drop_reason = "min_history"
        elif policy == "keep_gaps":
            eligible = total_rows >= int(min_history)
            if not eligible:
                drop_reason = "min_history"
        else:
            raise ValueError(f"Unsupported missing_data_policy: {policy}")

        if eligible:
            eligible_symbols.append(str(symbol))
        else:
            dropped_symbols.append(str(symbol))

        coverage_by_symbol.append(
            {
                "symbol": str(symbol),
                "n_rows": total_rows,
                "n_complete_rows": complete_rows,
                "n_missing_rows": missing_rows,
                "n_rows_after_policy": rows_after,
                "min_history_ok": bool(min_history_ok),
                "eligible": bool(eligible),
                "drop_reason": drop_reason,
            }
        )

    final_assets = sorted(eligible_symbols)
    asset_hash = _hash_assets(final_assets)

    filtered = window_after.loc[window_after["symbol"].isin(final_assets)].copy()
    filtered = filtered.sort_values(["symbol", "ts"]).reset_index(drop=True)

    diagnostics: Dict[str, Any] = {
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "min_history": int(min_history),
        "missing_data_policy": str(policy),
        "n_symbols_in": int(window["symbol"].nunique()),
        "n_symbols_out": int(len(final_assets)),
        "n_rows_in_window": int(len(window)),
        "n_rows_after_policy": int(len(window_after)),
        "rows_dropped_missing": int(rows_dropped_missing),
        "coverage_by_symbol": coverage_by_symbol,
        "dropped_symbols": dropped_symbols,
        "final_assets": final_assets,
        "asset_hash": asset_hash,
    }

    logger.info(
        "Universe selection policy=%s start=%s end=%s symbols_in=%s symbols_out=%s rows_in=%s rows_out=%s",
        policy,
        diagnostics["start_ts"],
        diagnostics["end_ts"],
        diagnostics["n_symbols_in"],
        diagnostics["n_symbols_out"],
        diagnostics["n_rows_in_window"],
        diagnostics["n_rows_after_policy"],
    )

    return filtered, diagnostics
