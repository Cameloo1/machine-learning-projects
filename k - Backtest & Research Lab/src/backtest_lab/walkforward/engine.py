from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from backtest_lab.data.universe import select_universe
from backtest_lab.execution.accounting import run_backtest
from backtest_lab.signals.features import compute_features
from backtest_lab.strategies import factory as strategy_factory
from backtest_lab.walkforward.windows import generate_walkforward_windows

logger = logging.getLogger(__name__)


def _ensure_no_leakage(window: Dict[str, Any]) -> None:
    if window["train_end"] >= window["test_start"]:
        raise ValueError("Leakage check failed: train_end must be before test_start")
    if window.get("val_end") is not None and window["val_end"] >= window["test_start"]:
        raise ValueError("Leakage check failed: val_end must be before test_start")


def run_walkforward(
    prices: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    wf_cfg = cfg.get("walkforward") or {}
    windows = generate_walkforward_windows(
        prices,
        train_days=int(wf_cfg["train_days"]),
        test_days=int(wf_cfg["test_days"]),
        step_days=int(wf_cfg["step_days"]),
        val_days=int(wf_cfg.get("val_days", 0) or 0),
    )

    if not windows:
        raise ValueError("Walk-forward produced zero windows; check date range and parameters")

    returns_list: List[pd.DataFrame] = []
    weights_list: List[pd.DataFrame] = []
    trades_list: List[pd.DataFrame] = []
    window_diags: List[Dict[str, Any]] = []

    for window in windows:
        _ensure_no_leakage(window)
        window_id = window["window_id"]

        window_prices = prices.loc[
            (prices["ts"] >= window["train_start"]) & (prices["ts"] <= window["test_end"])
        ].copy()

        universe_cfg = cfg["universe"]
        window_prices, universe_diag = select_universe(
            window_prices,
            start_ts=window["train_start"],
            end_ts=window["test_end"],
            min_history=universe_cfg["min_history_days"],
            policy=universe_cfg["missing_data_policy"],
        )

        features = compute_features(window_prices, cfg["features"])
        strategy = strategy_factory.create(cfg["strategy"])
        warmup_diag: Dict[str, Any] = {}
        weights = strategy.predict_weights(
            window_prices, features, cfg, diagnostics=warmup_diag
        )

        alignment_diag: Dict[str, Any] = {}
        returns_df, weights_df, trades_df = run_backtest(
            window_prices, weights, cfg["execution"], diagnostics=alignment_diag
        )

        returns_df = returns_df.loc[
            (returns_df["ts"] >= window["test_start"]) & (returns_df["ts"] <= window["test_end"])
        ].copy()
        weights_df = weights_df.loc[
            (weights_df["ts"] >= window["test_start"]) & (weights_df["ts"] <= window["test_end"])
        ].copy()
        trades_df = trades_df.loc[
            (trades_df["ts"] >= window["test_start"]) & (trades_df["ts"] <= window["test_end"])
        ].copy()

        returns_df["window_id"] = window_id
        weights_df["window_id"] = window_id
        trades_df["window_id"] = window_id

        returns_list.append(returns_df)
        weights_list.append(weights_df)
        trades_list.append(trades_df)

        window_diags.append(
            {
                "window": {
                    "window_id": window_id,
                    "train_start": window["train_start"].isoformat(),
                    "train_end": window["train_end"].isoformat(),
                    "val_start": window["val_start"].isoformat()
                    if window.get("val_start") is not None
                    else None,
                    "val_end": window["val_end"].isoformat()
                    if window.get("val_end") is not None
                    else None,
                    "test_start": window["test_start"].isoformat(),
                    "test_end": window["test_end"].isoformat(),
                },
                "universe": {
                    "final_assets": universe_diag.get("final_assets", []),
                    "asset_hash": universe_diag.get("asset_hash"),
                    "n_symbols_out": universe_diag.get("n_symbols_out"),
                },
                "alignment": alignment_diag,
                "warmup": warmup_diag,
            }
        )

    returns_all = pd.concat(returns_list, ignore_index=True)
    weights_all = pd.concat(weights_list, ignore_index=True)
    trades_all = pd.concat(trades_list, ignore_index=True)

    returns_all = returns_all.sort_values(["window_id", "ts"], kind="mergesort").reset_index(drop=True)
    weights_all = weights_all.sort_values(["window_id", "ts", "symbol"], kind="mergesort").reset_index(drop=True)
    trades_all = trades_all.sort_values(["window_id", "ts", "symbol"], kind="mergesort").reset_index(drop=True)

    diagnostics = {
        "window_count": len(windows),
        "windows": window_diags,
    }

    logger.info("Walk-forward complete windows=%s", len(windows))
    return returns_all, weights_all, trades_all, diagnostics
