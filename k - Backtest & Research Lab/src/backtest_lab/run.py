from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from backtest_lab import config as cfg_mod
from backtest_lab.data.loader import load_prices
from backtest_lab.data.universe import select_universe
from backtest_lab.data.validate import validate_prices
from backtest_lab.execution.accounting import run_backtest
from backtest_lab.execution.validate_outputs import (
    validate_output_weights_df,
    validate_returns_df,
    validate_trades_df,
)
from backtest_lab.metrics.drawdown import compute_drawdown
from backtest_lab.metrics.performance import compute_metrics, write_metrics_csv
from backtest_lab.metrics.returns import compute_returns
from backtest_lab.report.build import render_html
from backtest_lab.signals.features import compute_features
from backtest_lab.strategies import factory as strategy_factory
from backtest_lab.walkforward.engine import run_walkforward
from backtest_lab.walkforward.windows import generate_walkforward_windows

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a backtest from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output-dir", help="Override output directory for artifacts.")
    parser.add_argument("--run-id", help="Override run id (folder name).")
    return parser.parse_args()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _resolve_window(prices: pd.DataFrame, cfg_dict: Dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_raw = cfg_dict.get("data", {}).get("start")
    end_raw = cfg_dict.get("data", {}).get("end")
    start = pd.Timestamp(start_raw) if start_raw else prices["ts"].min()
    end = pd.Timestamp(end_raw) if end_raw else prices["ts"].max()
    if start > end:
        raise ValueError("Resolved start_ts is after end_ts")
    return start, end


def _metrics_payload(metrics: Dict[str, Any], returns_df: pd.DataFrame) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key, val in metrics.items():
        if key in {"equity_curve", "drawdown_series"}:
            continue
        if isinstance(val, (pd.Series, dict, list, tuple)):
            continue
        payload[key] = val
    payload["n_periods"] = int(len(returns_df))
    if len(returns_df):
        payload["start_ts"] = pd.Timestamp(returns_df["ts"].min()).isoformat()
        payload["end_ts"] = pd.Timestamp(returns_df["ts"].max()).isoformat()
    else:
        payload["start_ts"] = None
        payload["end_ts"] = None
    return payload


def _compute_benchmark_metrics(returns: pd.Series) -> Dict[str, Any]:
    ann_factor = 252
    n = int(len(returns))
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if n > 0 else 0.0
    cagr = float(equity.iloc[-1] ** (ann_factor / max(n, 1)) - 1.0) if n > 0 else 0.0
    ann_ret = float(returns.mean()) * ann_factor if n > 0 else 0.0
    ann_vol = float(returns.std(ddof=0)) * (ann_factor ** 0.5) if n > 1 else 0.0
    sharpe = (
        float(returns.mean() / returns.std(ddof=0)) * (ann_factor ** 0.5)
        if n > 1 and returns.std(ddof=0) != 0
        else 0.0
    )
    return {
        "benchmark_equity_curve": equity,
        "benchmark_total_return": total_return,
        "benchmark_cagr": cagr,
        "benchmark_annual_return": ann_ret,
        "benchmark_annual_vol": ann_vol,
        "benchmark_sharpe": sharpe,
    }


def _build_benchmark_series(
    prices: pd.DataFrame, returns_df: pd.DataFrame, symbol: str = "SPY"
) -> tuple[pd.Series | None, Dict[str, Any]]:
    diag: Dict[str, Any] = {"symbol": symbol, "rows": 0, "missing_on_ts": 0}
    if "symbol" not in prices.columns:
        return None, diag
    if symbol not in set(prices["symbol"].astype(str).unique()):
        return None, diag

    returns = compute_returns(prices)
    bench = returns.loc[returns["symbol"] == symbol, ["ts", "ret"]].copy()
    if bench.empty:
        return None, diag

    aligned = returns_df[["ts"]].merge(bench, on="ts", how="left")
    missing = int(aligned["ret"].isna().sum())
    diag["rows"] = int(len(bench))
    diag["missing_on_ts"] = missing
    if missing:
        aligned["ret"] = aligned["ret"].fillna(0.0)
    series = pd.Series(aligned["ret"].values, index=pd.to_datetime(aligned["ts"]))
    return series, diag


def _check_walkforward_min_windows(prices: pd.DataFrame, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    wf_cfg = cfg_dict.get("walkforward") or {}
    windows = generate_walkforward_windows(
        prices,
        train_days=int(wf_cfg["train_days"]),
        test_days=int(wf_cfg["test_days"]),
        step_days=int(wf_cfg["step_days"]),
        val_days=int(wf_cfg.get("val_days", 0) or 0),
    )
    min_required = 6
    if len(windows) < min_required:
        raise ValueError(
            f"Walk-forward produced {len(windows)} windows; minimum required is {min_required}"
        )
    return {"window_count": len(windows), "min_required": min_required}


def _run_pipeline(cfg: cfg_mod.Config, cfg_path: Path, artifacts_dir: Path) -> None:
    cfg_dict = cfg.to_dict()
    prices = load_prices(cfg)
    prices, validate_diag = validate_prices(prices)
    integrity_report = validate_diag.get("data_integrity_report")
    if integrity_report:
        _write_json(artifacts_dir / "data_integrity.json", integrity_report)

    start_ts, end_ts = _resolve_window(prices, cfg_dict)
    prices = prices.loc[(prices["ts"] >= start_ts) & (prices["ts"] <= end_ts)].copy()

    walkforward_cfg = cfg_dict.get("walkforward") or {}
    if walkforward_cfg.get("enabled"):
        wf_integrity = _check_walkforward_min_windows(prices, cfg_dict)
        returns_df, weights_df, trades_df, wf_diag = run_walkforward(prices, cfg_dict)
        diagnostics = {
            "validate": validate_diag,
            "walkforward": {**wf_diag, "integrity": wf_integrity},
        }
    else:
        universe_cfg = cfg_dict["universe"]
        prices, universe_diag = select_universe(
            prices,
            start_ts=start_ts,
            end_ts=end_ts,
            min_history=universe_cfg["min_history_days"],
            policy=universe_cfg["missing_data_policy"],
            symbols=universe_cfg.get("symbols"),
        )

        features = compute_features(prices, cfg_dict["features"])
        strategy = strategy_factory.create(cfg_dict["strategy"])
        warmup_diag: Dict[str, Any] = {}
        weights = strategy.predict_weights(prices, features, cfg_dict, diagnostics=warmup_diag)

        alignment_diag: Dict[str, Any] = {}
        returns_df, weights_df, trades_df = run_backtest(
            prices, weights, cfg_dict["execution"], diagnostics=alignment_diag
        )

        diagnostics = {
            "validate": validate_diag,
            "universe": universe_diag,
            "alignment": alignment_diag,
            "warmup": warmup_diag,
        }

    if "window_id" in returns_df.columns:
        returns_df = returns_df.sort_values(["window_id", "ts"]).reset_index(drop=True)
        weights_df = weights_df.sort_values(["window_id", "ts", "symbol"]).reset_index(drop=True)
        trades_df = trades_df.sort_values(["window_id", "ts", "symbol"]).reset_index(drop=True)
    else:
        returns_df = returns_df.sort_values("ts").reset_index(drop=True)
        weights_df = weights_df.sort_values(["ts", "symbol"]).reset_index(drop=True)
        trades_df = trades_df.sort_values(["ts", "symbol"]).reset_index(drop=True)

    validate_returns_df(returns_df)
    validate_output_weights_df(weights_df)
    validate_trades_df(trades_df)

    metrics = compute_metrics(returns_df, cfg_dict)
    bench_series, bench_diag = _build_benchmark_series(prices, returns_df)
    if bench_series is not None:
        metrics.update(_compute_benchmark_metrics(bench_series))
    diagnostics["benchmark"] = bench_diag
    drawdown = compute_drawdown(metrics["equity_curve"])
    metrics["max_drawdown"] = drawdown["max_drawdown"]
    metrics["drawdown_series"] = drawdown["drawdown"]

    returns_df.to_csv(artifacts_dir / "returns.csv", index=False)
    weights_df.to_csv(artifacts_dir / "weights.csv", index=False)
    trades_df.to_csv(artifacts_dir / "trades.csv", index=False)
    _write_json(artifacts_dir / "diagnostics.json", diagnostics)
    _write_json(artifacts_dir / "metrics.json", _metrics_payload(metrics, returns_df))
    write_metrics_csv(metrics, artifacts_dir / "metrics.csv")

    render_html(cfg_dict, artifacts_dir, returns_df, metrics, diagnostics)

    logger.info("Pipeline complete: %s", artifacts_dir)


def run_from_config_path(
    config_path: Path,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> Path:
    cfg = cfg_mod.load_config(config_path)

    if output_dir:
        cfg.output_dir = Path(output_dir)
    if run_id:
        cfg.run_id = run_id

    cfg = cfg_mod.resolve_config(cfg, config_path)

    artifacts_dir = cfg.output_dir / cfg.run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Resolved artifacts directory: %s", artifacts_dir)

    config_sha256 = cfg_mod.write_resolved_config(artifacts_dir, cfg)
    cfg_mod.write_run_metadata(artifacts_dir, cfg, config_path, config_sha256)
    config_integrity = cfg_mod.check_config_integrity(cfg)
    _write_json(artifacts_dir / "config_integrity.json", config_integrity)

    _run_pipeline(cfg, config_path, artifacts_dir)

    return artifacts_dir


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    run_from_config_path(
        Path(args.config),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        run_id=args.run_id,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
