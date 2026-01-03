from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtest_lab.data.universe import select_universe
from backtest_lab.execution.accounting import run_backtest
from backtest_lab.metrics.returns import compute_returns
from backtest_lab.signals.features import compute_features
from backtest_lab.strategies import factory as strategy_factory
from backtest_lab.walkforward.windows import generate_walkforward_windows

logger = logging.getLogger(__name__)


def _ensure_no_leakage(window: Dict[str, Any]) -> None:
    if window["train_end"] >= window["test_start"]:
        raise ValueError("Leakage check failed: train_end must be before test_start")
    if window.get("val_end") is not None and window["val_end"] >= window["test_start"]:
        raise ValueError("Leakage check failed: val_end must be before test_start")


def _slice_by_ts(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, *, ts_col: str = "ts") -> pd.DataFrame:
    return df.loc[(df[ts_col] >= start) & (df[ts_col] <= end)].copy()


def _validation_score(net_returns: pd.Series) -> Tuple[float, str]:
    n = int(len(net_returns))
    if n > 1:
        std = float(net_returns.std(ddof=0))
        if std > 0:
            sharpe = float(net_returns.mean() / std) * np.sqrt(252)
            return sharpe, "sharpe"
    equity = (1.0 + net_returns.fillna(0.0)).cumprod()
    if n == 0:
        return 0.0, "cagr"
    cagr = float(equity.iloc[-1] ** (252 / max(n, 1)) - 1.0)
    return cagr, "cagr"


def _build_sma_candidates(features_cfg: Dict[str, Any]) -> List[Dict[str, int]]:
    base_fast = int(features_cfg["sma_fast"])
    base_slow = int(features_cfg["sma_slow"])
    raw = [
        (base_fast, base_slow),
        (max(2, base_fast - 2), base_slow),
        (base_fast, base_slow + 5),
        (base_fast + 2, base_slow + 4),
        (max(2, base_fast - 1), base_slow + 2),
    ]
    seen = set()
    out = []
    for fast, slow in raw:
        if fast >= slow or fast < 2:
            continue
        key = (int(fast), int(slow))
        if key in seen:
            continue
        seen.add(key)
        out.append({"sma_fast": int(fast), "sma_slow": int(slow)})
    return out[:6]


def _build_rsi_candidates(features_cfg: Dict[str, Any]) -> List[Dict[str, float]]:
    base_low = float(features_cfg["rsi_low"])
    base_high = float(features_cfg["rsi_high"])
    lows = [base_low - 5, base_low, base_low + 5]
    highs = [base_high - 5, base_high, base_high + 5]
    seen = set()
    out = []
    for low in lows:
        for high in highs:
            if not (0 < low < high < 100):
                continue
            key = (round(low, 2), round(high, 2))
            if key in seen:
                continue
            seen.add(key)
            out.append({"rsi_low": float(low), "rsi_high": float(high)})
    return out[:6]


def _with_feature_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg_out = dict(cfg)
    cfg_out["features"] = dict(cfg.get("features", {}) or {})
    cfg_out["features"].update(overrides)
    cfg_out["strategy"] = dict(cfg.get("strategy", {}) or {})
    cfg_out["execution"] = dict(cfg.get("execution", {}) or {})
    cfg_out["universe"] = dict(cfg.get("universe", {}) or {})
    return cfg_out


def run_walkforward(
    prices: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    wf_cfg = cfg.get("walkforward") or {}
    selection_mode = str(cfg.get("universe_selection_mode", "train_only"))
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
    window_universe_diags: List[Dict[str, Any]] = []

    for window in windows:
        _ensure_no_leakage(window)
        window_id = window["window_id"]

        window_prices = prices.loc[
            (prices["ts"] >= window["train_start"]) & (prices["ts"] <= window["test_end"])
        ].copy()

        universe_cfg = cfg["universe"]
        if selection_mode == "train_only":
            selection_end = window.get("val_end") or window["train_end"]
            selection_prices = window_prices.loc[
                (window_prices["ts"] >= window["train_start"])
                & (window_prices["ts"] <= selection_end)
            ].copy()
            _selected, universe_diag = select_universe(
                selection_prices,
                start_ts=window["train_start"],
                end_ts=selection_end,
                min_history=universe_cfg["min_history_days"],
                policy=universe_cfg["missing_data_policy"],
                symbols=universe_cfg.get("symbols"),
            )
            final_assets = set(universe_diag.get("final_assets", []))
            window_prices = window_prices.loc[window_prices["symbol"].isin(final_assets)].copy()
            if universe_cfg["missing_data_policy"] == "drop_rows":
                complete_mask = window_prices[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
                window_prices = window_prices.loc[complete_mask].copy()
            window_prices = window_prices.sort_values(["symbol", "ts"]).reset_index(drop=True)
            universe_diag["selection_mode"] = selection_mode
            universe_diag["selection_end"] = pd.Timestamp(selection_end).isoformat()
        else:
            window_prices, universe_diag = select_universe(
                window_prices,
                start_ts=window["train_start"],
                end_ts=window["test_end"],
                min_history=universe_cfg["min_history_days"],
                policy=universe_cfg["missing_data_policy"],
                symbols=universe_cfg.get("symbols"),
            )
            universe_diag["selection_mode"] = selection_mode
            universe_diag["selection_end"] = pd.Timestamp(window["test_end"]).isoformat()

        window_returns = compute_returns(window_prices)
        validation_diag: Dict[str, Any] = {"enabled": False}
        selected_cfg = cfg

        val_start = window.get("val_start")
        val_end = window.get("val_end")
        if val_start is not None and val_end is not None:
            strat_name = str(cfg.get("strategy", {}).get("name"))
            if strat_name in {"sma_trend", "rsi_mr"}:
                candidates = (
                    _build_sma_candidates(cfg["features"])
                    if strat_name == "sma_trend"
                    else _build_rsi_candidates(cfg["features"])
                )
                candidate_results = []
                for overrides in candidates:
                    candidate_cfg = _with_feature_overrides(cfg, overrides)
                    features_candidate = compute_features(window_prices, candidate_cfg["features"])
                    candidate_strategy = strategy_factory.create(candidate_cfg["strategy"])
                    weights_candidate = candidate_strategy.predict_weights(
                        window_prices, features_candidate, candidate_cfg, diagnostics=None
                    )
                    val_returns_df, _, _ = run_backtest(
                        window_prices, weights_candidate, candidate_cfg["execution"], diagnostics=None
                    )
                    decision_col = "decision_ts" if "decision_ts" in val_returns_df.columns else "ts"
                    val_slice = _slice_by_ts(val_returns_df, val_start, val_end, ts_col=decision_col)
                    score, metric = _validation_score(val_slice["net"].astype(float))
                    candidate_results.append(
                        {
                            "params": overrides,
                            "score": float(score),
                            "metric": metric,
                            "n_periods": int(len(val_slice)),
                            "decision_start": val_slice[decision_col].min().isoformat()
                            if len(val_slice)
                            else None,
                            "decision_end": val_slice[decision_col].max().isoformat()
                            if len(val_slice)
                            else None,
                        }
                    )
                if candidate_results:
                    candidate_results = sorted(
                        candidate_results, key=lambda item: item["score"], reverse=True
                    )
                    best = candidate_results[0]
                    selected_cfg = _with_feature_overrides(cfg, best["params"])
                    validation_diag = {
                        "enabled": True,
                        "metric": best["metric"],
                        "selected_params": best["params"],
                        "selected_score": best["score"],
                        "candidates": candidate_results,
                        "refit_on_train_val": True,
                    }

        features = compute_features(window_prices, selected_cfg["features"])
        strategy = strategy_factory.create(selected_cfg["strategy"])
        warmup_diag: Dict[str, Any] = {}

        fit_start = window["train_start"]
        fit_end = window["train_end"]
        fit_scope = "train"
        if validation_diag.get("enabled") and window.get("val_end") is not None:
            fit_end = window["val_end"]
            fit_scope = "train_val"

        fit_prices = _slice_by_ts(window_prices, fit_start, fit_end)
        fit_features = _slice_by_ts(features, fit_start, fit_end)
        fit_returns = _slice_by_ts(window_returns, fit_start, fit_end)
        fit_diag: Dict[str, Any] = {"fit_scope": fit_scope}
        strategy.fit(fit_prices, fit_features, fit_returns, selected_cfg, diagnostics=fit_diag)

        weights_all = strategy.predict_weights(
            window_prices, features, selected_cfg, diagnostics=warmup_diag
        )
        weights = _slice_by_ts(weights_all, window["test_start"], window["test_end"])

        alignment_diag: Dict[str, Any] = {}
        returns_df, weights_df, trades_df = run_backtest(
            window_prices, weights, selected_cfg["execution"], diagnostics=alignment_diag
        )

        decision_col = "decision_ts" if "decision_ts" in returns_df.columns else "ts"
        returns_df = returns_df.loc[
            (returns_df[decision_col] >= window["test_start"])
            & (returns_df[decision_col] <= window["test_end"])
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
                    "selection_mode": universe_diag.get("selection_mode"),
                    "selection_end": universe_diag.get("selection_end"),
                },
                "alignment": alignment_diag,
                "warmup": warmup_diag,
                "fit": fit_diag,
                "validation": validation_diag,
                "leakage_check": True,
            }
        )
        window_universe_diags.append(
            {
                "window_id": window_id,
                "final_assets": universe_diag.get("final_assets", []),
                "asset_hash": universe_diag.get("asset_hash"),
                "n_symbols_out": universe_diag.get("n_symbols_out"),
                "selection_mode": universe_diag.get("selection_mode"),
                "selection_end": universe_diag.get("selection_end"),
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
        "window_universe_diagnostics": window_universe_diags,
    }

    logger.info("Walk-forward complete windows=%s", len(windows))
    return returns_all, weights_all, trades_all, diagnostics
