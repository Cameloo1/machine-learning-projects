"""RL Trading Sandbox - Core module for reinforcement learning trading experiments."""

from rl_trading.config import (
    DataConfig,
    FeatureConfig,
    TrainingConfig,
    get_default_configs,
)
from rl_trading.data_loader import SPYDataLoader
from rl_trading.features import (
    add_basic_features,
    get_feature_columns,
    FeatureScaler,
    prepare_features_for_split,
)
from rl_trading.envs import (
    TradingEnv,
    make_env_from_df,
    make_train_env,
    make_val_env,
    make_test_env,
)
from rl_trading.utils import (
    compute_equity_metrics,
    compute_return_series,
    compute_sharpe,
    extract_trades,
    compute_trade_stats,
    run_random_policy,
    create_experiment_folder,
    save_config,
    append_metrics_row,
    save_json,
    load_json,
    plot_equity_curve,
    plot_action_distribution,
    plot_pnl_histogram,
    plot_position_over_time,
)
from rl_trading.training import (
    evaluate_model_on_env,
    compute_buy_and_hold_equity,
    compute_buy_and_hold_metrics,
    train_dqn,
    run_test_evaluation,
)

__all__ = [
    # Config
    "DataConfig",
    "FeatureConfig",
    "TrainingConfig",
    "get_default_configs",
    # Data
    "SPYDataLoader",
    # Features
    "add_basic_features",
    "get_feature_columns",
    "FeatureScaler",
    "prepare_features_for_split",
    # Environment
    "TradingEnv",
    "make_env_from_df",
    "make_train_env",
    "make_val_env",
    "make_test_env",
    # Metrics & Utils
    "compute_equity_metrics",
    "compute_return_series",
    "compute_sharpe",
    "extract_trades",
    "compute_trade_stats",
    "run_random_policy",
    # Logging
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
    # Training
    "evaluate_model_on_env",
    "compute_buy_and_hold_equity",
    "compute_buy_and_hold_metrics",
    "train_dqn",
    "run_test_evaluation",
]

__version__ = "1.0.0"

