from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from backtest_lab.metrics.tables import build_metrics_table


def _plot_series(series, path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 3))
    plt.plot(series.index, series.values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("ts")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_multi(series_map: Dict[str, Any], path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 3))
    for label, series in series_map.items():
        plt.plot(series.index, series.values, label=label)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("ts")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def render_html(
    cfg: Dict[str, Any],
    artifacts_dir: Path,
    returns_df,
    metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ts_index = pd.to_datetime(returns_df["ts"])
    equity = metrics.get("equity_curve")
    drawdown = metrics.get("drawdown_series")
    rolling_sharpe = metrics.get("rolling_sharpe")

    gross_equity = (1.0 + returns_df["gross"].fillna(0.0)).cumprod()
    gross_equity.index = ts_index
    if equity is not None and len(equity) == len(ts_index):
        equity.index = ts_index
    if drawdown is not None and len(drawdown) == len(ts_index):
        drawdown.index = ts_index
    if rolling_sharpe is not None and len(rolling_sharpe) == len(ts_index):
        rolling_sharpe.index = ts_index

    equity_path = plots_dir / "equity.png"
    drawdown_path = plots_dir / "drawdown.png"
    rolling_path = plots_dir / "rolling_sharpe.png"
    turnover_path = plots_dir / "turnover_costs.png"

    if equity is not None and len(equity) > 0:
        _plot_multi(
            {"net": equity, "gross": gross_equity},
            equity_path,
            "Equity Curve (Net vs Gross)",
            "Equity",
        )
    if drawdown is not None and len(drawdown) > 0:
        _plot_series(drawdown, drawdown_path, "Drawdown (Net)", "Drawdown")
    if rolling_sharpe is not None and len(rolling_sharpe) > 0:
        _plot_series(rolling_sharpe, rolling_path, "Rolling Sharpe (Net)", "Sharpe")

    turnover = returns_df.get("turnover")
    costs = returns_df.get("costs")
    slippage = returns_df.get("slippage_cost")
    txn_cost = returns_df.get("txn_cost")
    if turnover is not None and costs is not None:
        turnover = pd.Series(turnover.values, index=ts_index)
        costs = pd.Series(costs.values, index=ts_index)
        slippage_series = (
            pd.Series(slippage.values, index=ts_index) if slippage is not None else costs * 0.0
        )
        txn_series = pd.Series(txn_cost.values, index=ts_index) if txn_cost is not None else costs * 0.0
        _plot_multi(
            {
                "turnover": turnover,
                "costs": costs,
                "slippage": slippage_series,
                "txn_cost": txn_series,
            },
            turnover_path,
            "Turnover & Costs",
            "Value",
        )

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=True,
    )
    template = env.get_template("report.html.j2")

    metrics_table = build_metrics_table(metrics).to_dict(orient="records")

    data_diag_json = json.dumps(diagnostics.get("validate", {}), indent=2, sort_keys=True)
    universe_diag_json = json.dumps(
        diagnostics.get("universe", {}),
        indent=2,
        sort_keys=True,
    )
    walkforward_obj = diagnostics.get("walkforward")
    walkforward_json = (
        json.dumps(walkforward_obj, indent=2, sort_keys=True)
        if walkforward_obj
        else None
    )

    html = template.render(
        cfg_json=json.dumps(cfg, indent=2, sort_keys=True),
        metrics_table=metrics_table,
        diagnostics_json=json.dumps(diagnostics, indent=2, sort_keys=True),
        data_diag_json=data_diag_json,
        universe_diag_json=universe_diag_json,
        walkforward_json=walkforward_json,
        equity_plot=str(Path("plots") / equity_path.name),
        drawdown_plot=str(Path("plots") / drawdown_path.name),
        rolling_plot=str(Path("plots") / rolling_path.name),
        turnover_plot=str(Path("plots") / turnover_path.name),
        returns_preview=returns_df.head(5).to_dict(orient="records"),
    )

    out_path = artifacts_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
