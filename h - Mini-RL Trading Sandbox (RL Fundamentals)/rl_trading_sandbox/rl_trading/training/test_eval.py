"""Test evaluation script for the RL Trading Sandbox.

This module provides functionality to evaluate a trained DQN model on the
held-out test set and generate comprehensive performance reports with plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import DQN

from rl_trading.config import get_default_configs, FeatureConfig
from rl_trading.data_loader import SPYDataLoader
from rl_trading.features import add_basic_features, get_feature_columns, FeatureScaler
from rl_trading.envs import make_env_from_df
from rl_trading.training.eval import (
    evaluate_model_on_env,
    compute_buy_and_hold_equity,
    compute_buy_and_hold_metrics,
    STEPS_PER_YEAR,
)
from rl_trading.utils.logging_utils import (
    create_experiment_folder,
    save_json,
)
from rl_trading.utils.plotting import (
    plot_equity_curve,
    plot_action_distribution,
    plot_pnl_histogram,
    plot_position_over_time,
    plot_candlestick_with_trades,
)


def run_test_evaluation(
    exp_name: str = "exp_001_baseline_dqn",
    trading_cost_bp: float = 1.0,
) -> dict[str, float]:
    """Run full test evaluation on a trained model.
    
    Loads the best model from an experiment folder, evaluates it on the
    test split, computes metrics, and saves artifacts (metrics, trades CSV,
    plots) to the experiment folder.
    
    Args:
        exp_name: Name of the experiment folder containing the trained model.
        trading_cost_bp: Trading cost in basis points (should match training).
    
    Returns:
        Dictionary containing test metrics.
    
    Raises:
        FileNotFoundError: If best_model.zip is not found in experiment folder.
    
    Example:
        >>> metrics = run_test_evaluation("exp_001_baseline_dqn")
        >>> print(f"Test Return: {metrics['total_return']:.2%}")
    """
    print("=" * 60)
    print("Test Evaluation Pipeline")
    print("=" * 60)
    
    # Load configs
    print("\n[1] Loading configurations...")
    data_config, feature_config, training_config = get_default_configs()
    print(f"    CSV path: {data_config.csv_path}")
    print(f"    Test period: {data_config.test_start} to {data_config.test_end or 'latest'}")
    
    # Get experiment path
    exp_path = Path("experiments") / exp_name
    model_path = exp_path / "best_model.zip"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. "
            f"Please run training first: python -m rl_trading.training.train"
        )
    
    print(f"    Experiment: {exp_path}")
    print(f"    Model: {model_path}")
    
    # Load data
    print("\n[2] Loading SPY data...")
    loader = SPYDataLoader(data_config)
    train_df_raw, val_df_raw, test_df_raw = loader.get_splits()
    print(f"    Train: {len(train_df_raw)} rows")
    print(f"    Val:   {len(val_df_raw)} rows")
    print(f"    Test:  {len(test_df_raw)} rows")
    
    # Feature engineering
    print("\n[3] Applying feature engineering...")
    train_df = add_basic_features(train_df_raw, use_log_returns=feature_config.use_log_returns)
    test_df = add_basic_features(test_df_raw, use_log_returns=feature_config.use_log_returns)
    print(f"    Train with features: {len(train_df)} rows")
    print(f"    Test with features:  {len(test_df)} rows")
    
    # Scale features (fit on train, transform test)
    print("\n[4] Scaling features...")
    feature_cols = get_feature_columns(use_log_returns=feature_config.use_log_returns)
    scaler = FeatureScaler()
    train_df = scaler.fit_transform(train_df, feature_cols)
    test_df_scaled = scaler.transform(test_df, feature_cols)
    print(f"    Feature columns: {feature_cols}")
    print(f"    Scaler fitted on train, transformed test")
    
    # Create test environment
    print("\n[5] Creating test environment...")
    max_test_steps = len(test_df_scaled) - feature_config.window_size - 1
    
    test_env = make_env_from_df(
        df=test_df_scaled,
        feature_cols=feature_cols,
        window_size=feature_config.window_size,
        trading_cost_bp=trading_cost_bp,
        max_episode_steps=max_test_steps,
        deterministic_start=True,
        seed=training_config.seed,
    )
    print(f"    Test env: {test_env}")
    print(f"    Max episode steps: {max_test_steps}")
    
    # Load model
    print("\n[6] Loading trained model...")
    model = DQN.load(model_path)
    print(f"    Model loaded from: {model_path}")
    
    # Evaluate model on test environment
    print("\n[7] Evaluating on test set...")
    test_metrics, trades_df = evaluate_model_on_env(
        model=model,
        env=test_env,
        episodes=1,
        steps_per_year=STEPS_PER_YEAR,
    )
    print(f"    Evaluation complete: {len(trades_df)} steps")
    
    # Compute buy-and-hold equity for comparison
    print("\n[8] Computing buy-and-hold baseline...")
    
    # Get the test data slice that corresponds to the traded period
    # We need to align timestamps with what the environment actually used
    test_timestamps = trades_df["timestamp"].values
    test_df_aligned = test_df_scaled[
        test_df_scaled["timestamp"].isin(test_timestamps)
    ].copy()
    
    bh_equity = compute_buy_and_hold_equity(test_df_aligned)
    bh_metrics = compute_buy_and_hold_metrics(test_df_aligned)
    print(f"    B&H Return: {bh_metrics['bh_total_return']:+.2%}")
    
    # Save artifacts
    print("\n[9] Saving artifacts...")
    
    # Combine metrics
    all_metrics = {
        **test_metrics,
        **bh_metrics,
        "excess_return": test_metrics["total_return"] - bh_metrics["bh_total_return"],
    }
    
    # Save test metrics JSON
    save_json(all_metrics, exp_path / "test_metrics.json")
    print(f"    Saved: test_metrics.json")
    
    # Save trades CSV
    trades_df.to_csv(exp_path / "trades_test.csv", index=False)
    print(f"    Saved: trades_test.csv")
    
    # Generate and save plots
    print("\n[10] Generating plots...")
    
    plot_equity_curve(
        trades_df=trades_df,
        buy_and_hold_equity=bh_equity,
        output_path=exp_path / "equity_curve_test.png",
        title=f"Test Equity Curve: {exp_name}",
    )
    print(f"    Saved: equity_curve_test.png")
    
    plot_action_distribution(
        trades_df=trades_df,
        output_path=exp_path / "action_distribution_test.png",
        title=f"Action Distribution: {exp_name}",
    )
    print(f"    Saved: action_distribution_test.png")
    
    plot_pnl_histogram(
        trades_df=trades_df,
        output_path=exp_path / "pnl_histogram_test.png",
        title=f"Step PnL Distribution: {exp_name}",
    )
    print(f"    Saved: pnl_histogram_test.png")
    
    plot_position_over_time(
        trades_df=trades_df,
        output_path=exp_path / "position_over_time_test.png",
        title=f"Position Over Time: {exp_name}",
    )
    print(f"    Saved: position_over_time_test.png")
    
    # Candlestick chart with trades
    # Use the original test_df (before feature scaling) for OHLC data
    test_df_ohlc = test_df_raw[
        test_df_raw["timestamp"].isin(test_timestamps)
    ].copy()
    
    plot_candlestick_with_trades(
        trades_df=trades_df,
        output_path=exp_path / "candlestick_trades_test.png",
        price_df=test_df_ohlc,
        title=f"Candlestick Chart with Trades: {exp_name}",
        max_bars=500,
    )
    print(f"    Saved: candlestick_trades_test.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Evaluation Summary")
    print("=" * 60)
    print(f"\nRL Strategy Performance:")
    print(f"  Final Equity:   {test_metrics['final_equity']:.4f}")
    print(f"  Total Return:   {test_metrics['total_return']:+.2%}")
    print(f"  Max Drawdown:   {test_metrics['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio:   {test_metrics['sharpe']:.2f}")
    print(f"  Trades:         {test_metrics['n_trades']}")
    print(f"  Win Rate:       {test_metrics['win_rate']:.1%}")
    print(f"  Profit Factor:  {test_metrics['profit_factor']:.2f}")
    
    print(f"\nBuy-and-Hold Baseline:")
    print(f"  Final Equity:   {bh_metrics['bh_final_equity']:.4f}")
    print(f"  Total Return:   {bh_metrics['bh_total_return']:+.2%}")
    print(f"  Max Drawdown:   {bh_metrics['bh_max_drawdown']:.2%}")
    print(f"  Sharpe Ratio:   {bh_metrics['bh_sharpe']:.2f}")
    
    print(f"\nExcess Return (RL - B&H): {all_metrics['excess_return']:+.2%}")
    
    print(f"\nArtifacts saved to: {exp_path}")
    print("  - test_metrics.json")
    print("  - trades_test.csv")
    print("  - equity_curve_test.png")
    print("  - action_distribution_test.png")
    print("  - pnl_histogram_test.png")
    print("  - position_over_time_test.png")
    
    return all_metrics


def run_test_evaluation_synthetic(
    exp_name: str = "exp_synthetic_test",
    trading_cost_bp: float = 1.0,
) -> dict[str, float]:
    """Run test evaluation with synthetic data for testing purposes.
    
    Args:
        exp_name: Experiment name (should have been trained with synthetic data).
        trading_cost_bp: Trading cost in basis points.
    
    Returns:
        Dictionary containing test metrics.
    """
    print("=" * 60)
    print("Test Evaluation with Synthetic Data")
    print("=" * 60)
    
    # Get experiment path
    exp_path = Path("experiments") / exp_name
    model_path = exp_path / "best_model.zip"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. "
            f"Run with synthetic training first: "
            f"python -m rl_trading.training.train --synthetic"
        )
    
    # Generate synthetic data matching training
    print("\n[1] Generating synthetic data...")
    np.random.seed(42)
    n_bars = 3000
    
    timestamps = pd.date_range("2019-01-01", periods=n_bars, freq="30min")
    close = 300 + np.random.randn(n_bars).cumsum() * 0.3
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": close + np.random.randn(n_bars) * 0.1,
        "high": close + np.abs(np.random.randn(n_bars) * 0.2),
        "low": close - np.abs(np.random.randn(n_bars) * 0.2),
        "close": close,
        "volume": np.random.randint(100000, 1000000, n_bars),
    })
    
    # Split same as training
    split_train = int(n_bars * 0.7)
    split_val = int(n_bars * 0.85)
    train_df_raw = df.iloc[:split_train].copy()
    test_df_raw = df.iloc[split_val:].copy()  # Use last 15% as test
    
    print(f"    Train: {len(train_df_raw)} rows")
    print(f"    Test:  {len(test_df_raw)} rows")
    
    # Feature engineering
    print("\n[2] Applying feature engineering...")
    feature_config = FeatureConfig()
    train_df = add_basic_features(train_df_raw, use_log_returns=feature_config.use_log_returns)
    test_df = add_basic_features(test_df_raw, use_log_returns=feature_config.use_log_returns)
    
    feature_cols = get_feature_columns(use_log_returns=feature_config.use_log_returns)
    scaler = FeatureScaler()
    train_df = scaler.fit_transform(train_df, feature_cols)
    test_df_scaled = scaler.transform(test_df, feature_cols)
    print(f"    Test with features: {len(test_df_scaled)} rows")
    
    # Create test environment
    print("\n[3] Creating test environment...")
    max_test_steps = len(test_df_scaled) - feature_config.window_size - 1
    
    test_env = make_env_from_df(
        df=test_df_scaled,
        feature_cols=feature_cols,
        window_size=feature_config.window_size,
        trading_cost_bp=trading_cost_bp,
        max_episode_steps=max_test_steps,
        deterministic_start=True,
        seed=42,
    )
    
    # Load model
    print("\n[4] Loading trained model...")
    model = DQN.load(model_path)
    
    # Evaluate
    print("\n[5] Evaluating on test set...")
    test_metrics, trades_df = evaluate_model_on_env(model, test_env, episodes=1)
    
    # Buy-and-hold
    test_timestamps = trades_df["timestamp"].values
    test_df_aligned = test_df_scaled[
        test_df_scaled["timestamp"].isin(test_timestamps)
    ].copy()
    
    # Get original OHLC data for candlestick chart
    test_df_ohlc = test_df_raw[
        test_df_raw["timestamp"].isin(test_timestamps)
    ].copy()
    
    bh_equity = compute_buy_and_hold_equity(test_df_aligned)
    bh_metrics = compute_buy_and_hold_metrics(test_df_aligned)
    
    # Save artifacts
    print("\n[6] Saving artifacts...")
    all_metrics = {**test_metrics, **bh_metrics}
    
    save_json(all_metrics, exp_path / "test_metrics.json")
    trades_df.to_csv(exp_path / "trades_test.csv", index=False)
    
    plot_equity_curve(trades_df, bh_equity, exp_path / "equity_curve_test.png")
    plot_action_distribution(trades_df, exp_path / "action_distribution_test.png")
    plot_pnl_histogram(trades_df, exp_path / "pnl_histogram_test.png")
    plot_position_over_time(trades_df, exp_path / "position_over_time_test.png")
    plot_candlestick_with_trades(
        trades_df=trades_df,
        output_path=exp_path / "candlestick_trades_test.png",
        price_df=test_df_ohlc,
        title=f"Candlestick Chart with Trades: {exp_name}",
        max_bars=500,
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"RL Return:  {test_metrics['total_return']:+.2%}")
    print(f"B&H Return: {bh_metrics['bh_total_return']:+.2%}")
    print(f"\nArtifacts saved to: {exp_path}")
    
    return all_metrics


if __name__ == "__main__":
    import sys
    
    # Check for synthetic flag
    use_synthetic = "--synthetic" in sys.argv or "-s" in sys.argv
    
    # Get experiment name from args if provided
    exp_name = None
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            exp_name = arg
            break
    
    if use_synthetic:
        exp_name = exp_name or "exp_synthetic_test"
        print(f"Running test evaluation with synthetic data for: {exp_name}")
        run_test_evaluation_synthetic(exp_name)
    else:
        exp_name = exp_name or "exp_001_baseline_dqn"
        try:
            run_test_evaluation(exp_name)
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nTo run with synthetic data:")
            print("  python -m rl_trading.training.test_eval --synthetic")

