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
from backtest_lab.metrics.drawdown import compute_drawdown
from backtest_lab.metrics.performance import compute_metrics, write_metrics_csv
from backtest_lab.report.build import render_html
from backtest_lab.signals.features import compute_features
from backtest_lab.strategies import factory as strategy_factory
from backtest_lab.walkforward.engine import run_walkforward

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


def _run_pipeline(cfg: cfg_mod.Config, cfg_path: Path, artifacts_dir: Path) -> None:
    cfg_dict = cfg.to_dict()
    prices = load_prices(cfg)
    prices, validate_diag = validate_prices(prices)

    start_ts, end_ts = _resolve_window(prices, cfg_dict)
    prices = prices.loc[(prices["ts"] >= start_ts) & (prices["ts"] <= end_ts)].copy()

    walkforward_cfg = cfg_dict.get("walkforward") or {}
    if walkforward_cfg.get("enabled"):
        returns_df, weights_df, trades_df, wf_diag = run_walkforward(prices, cfg_dict)
        diagnostics = {
            "validate": validate_diag,
            "walkforward": wf_diag,
        }
    else:
        universe_cfg = cfg_dict["universe"]
        prices, universe_diag = select_universe(
            prices,
            start_ts=start_ts,
            end_ts=end_ts,
            min_history=universe_cfg["min_history_days"],
            policy=universe_cfg["missing_data_policy"],
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

    metrics = compute_metrics(returns_df, cfg_dict)
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
