"""Utility functions for the RL Trading Sandbox."""

from rl_trading.utils.metrics import (
    compute_equity_metrics,
    compute_return_series,
    compute_sharpe,
    extract_trades,
    compute_trade_stats,
)
from rl_trading.utils.random_runner import run_random_policy
from rl_trading.utils.logging_utils import (
    create_experiment_folder,
    save_config,
    append_metrics_row,
    save_json,
    load_json,
)
from rl_trading.utils.plotting import (
    plot_equity_curve,
    plot_action_distribution,
    plot_pnl_histogram,
    plot_position_over_time,
    plot_candlestick_with_trades,
)

__all__ = [
    # Metrics
    "compute_equity_metrics",
    "compute_return_series",
    "compute_sharpe",
    "extract_trades",
    "compute_trade_stats",
    # Random runner
    "run_random_policy",
    # Logging utilities
    "create_experiment_folder",
    "save_config",
    "append_metrics_row",
    "save_json",
    "load_json",
    # Plotting
    "plot_equity_curve",
    "plot_action_distribution",
    "plot_pnl_histogram",
    "plot_position_over_time",
    "plot_candlestick_with_trades",
]
