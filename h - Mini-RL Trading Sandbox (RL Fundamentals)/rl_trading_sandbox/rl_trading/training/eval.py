"""Evaluation utilities for trained RL models.

This module provides functions to evaluate trained models on trading environments
and compute comprehensive performance metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import pandas as pd

from rl_trading.utils.metrics import (
    compute_equity_metrics,
    compute_sharpe,
    extract_trades,
    compute_trade_stats,
)

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm


# Default steps per year for Sharpe calculation
# Assumes ~8 30-minute bars per trading day × 252 trading days
STEPS_PER_YEAR = 252 * 8


def compute_buy_and_hold_equity(df: pd.DataFrame) -> pd.Series:
    """Compute buy-and-hold equity curve from price data.
    
    Assumes starting equity of 1.0 and being fully long from the first bar
    to the last bar. Uses closing prices.
    
    Args:
        df: DataFrame with 'timestamp' and 'close' columns.
    
    Returns:
        Series indexed by timestamp with buy-and-hold equity values,
        starting at 1.0.
    
    Example:
        >>> bh_equity = compute_buy_and_hold_equity(test_df)
        >>> print(f"B&H Return: {bh_equity.iloc[-1] - 1:.2%}")
    """
    # Ensure sorted by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Compute returns
    returns = df["close"].pct_change().fillna(0.0)
    
    # Compute cumulative equity (starting at 1.0)
    equity = (1 + returns).cumprod()
    
    # Create Series indexed by timestamp
    equity_series = pd.Series(equity.values, index=df["timestamp"])
    
    return equity_series


def compute_buy_and_hold_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute performance metrics for buy-and-hold strategy.
    
    Args:
        df: DataFrame with 'timestamp' and 'close' columns.
    
    Returns:
        Dictionary with buy-and-hold metrics.
    """
    equity = compute_buy_and_hold_equity(df)
    
    final_equity = float(equity.iloc[-1])
    total_return = final_equity - 1.0
    
    # Max drawdown
    cumulative_peak = np.maximum.accumulate(equity.values)
    drawdown = equity.values / cumulative_peak - 1.0
    max_drawdown = float(np.min(drawdown))
    
    # Sharpe (annualized)
    returns = equity.pct_change().fillna(0.0)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(STEPS_PER_YEAR) if std_ret > 1e-10 else 0.0
    
    return {
        "bh_final_equity": final_equity,
        "bh_total_return": total_return,
        "bh_max_drawdown": max_drawdown,
        "bh_sharpe": float(sharpe),
    }


def evaluate_model_on_env(
    model: "BaseAlgorithm",
    env: gym.Env,
    episodes: int = 1,
    seed: int | None = None,
    steps_per_year: int = STEPS_PER_YEAR,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a trained model on a trading environment.
    
    Runs the model for multiple episodes with deterministic actions,
    recording all step information and computing performance metrics.
    
    Args:
        model: Trained stable-baselines3 model (e.g., DQN).
        env: Trading environment to evaluate on.
        episodes: Number of episodes to run.
        seed: Optional evaluation seed pushed once before looping episodes.
        steps_per_year: Steps per year for Sharpe ratio annualization.
    
    Returns:
        Tuple of:
        - metrics_dict: Flattened dictionary of all computed metrics
        - trades_df: DataFrame with all episode steps, includes 'episode_id' column
    
    Example:
        >>> metrics, trades_df = evaluate_model_on_env(model, val_env, episodes=1)
        >>> print(f"Return: {metrics['total_return']:.2%}")
    """
    all_steps = []
    
    if seed is not None:
        env.reset(seed=seed)
    
    for episode_id in range(episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            # Get deterministic action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            
            # Record step
            all_steps.append({
                "episode_id": episode_id,
                "timestamp": info["timestamp"],
                "price": info["price"],
                "action": info["action"],
                "position": info["position"],
                "step_pnl": info["step_pnl"],
                "equity": info["equity"],
                "trade_ret": info.get("trade_ret"),
                "trade_drawdown": info.get("trade_drawdown"),
            })
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(all_steps)
    
    # Compute metrics (aggregate across all episodes)
    equity_metrics = compute_equity_metrics(trades_df)
    sharpe = compute_sharpe(trades_df, steps_per_year=steps_per_year)
    
    # Extract and analyze trades
    trades_summary = extract_trades(trades_df)
    trade_stats = compute_trade_stats(trades_summary)
    
    # Flatten all metrics into single dict
    metrics_dict = {
        # Equity metrics
        "final_equity": equity_metrics["final_equity"],
        "total_return": equity_metrics["total_return"],
        "max_drawdown": equity_metrics["max_drawdown"],
        "sharpe": sharpe,
        # Trade stats
        "n_trades": trade_stats["n_trades"],
        "win_rate": trade_stats["win_rate"],
        "avg_win": trade_stats["avg_win"],
        "avg_loss": trade_stats["avg_loss"],
        "profit_factor": trade_stats["profit_factor"],
        "avg_trade_duration": trade_stats["avg_trade_duration"],
        # Episode info
        "n_episodes": episodes,
        "total_steps": len(trades_df),
    }
    
    return metrics_dict, trades_df


def evaluate_and_print(
    model: "BaseAlgorithm",
    env: gym.Env,
    env_name: str = "Environment",
    episodes: int = 1,
    steps_per_year: int = STEPS_PER_YEAR,
) -> dict[str, float]:
    """Evaluate model and print formatted results.
    
    Args:
        model: Trained model to evaluate.
        env: Environment to evaluate on.
        env_name: Name for display (e.g., "Validation", "Test").
        episodes: Number of episodes.
        steps_per_year: For Sharpe calculation.
    
    Returns:
        Metrics dictionary.
    """
    metrics, _ = evaluate_model_on_env(
        model, env, episodes=episodes, steps_per_year=steps_per_year
    )
    
    print(f"\n{'='*50}")
    print(f"{env_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Episodes:       {metrics['n_episodes']}")
    print(f"Total Steps:    {metrics['total_steps']}")
    print(f"Final Equity:   {metrics['final_equity']:.4f}")
    print(f"Total Return:   {metrics['total_return']:+.2%}")
    print(f"Max Drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio:   {metrics['sharpe']:.2f}")
    print(f"\nTrade Statistics:")
    print(f"  Trades:       {metrics['n_trades']}")
    print(f"  Win Rate:     {metrics['win_rate']:.1%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    return metrics

