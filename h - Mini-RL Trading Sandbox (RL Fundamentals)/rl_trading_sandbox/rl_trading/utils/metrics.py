"""Metrics and analysis utilities for trading episode evaluation.

This module provides functions to compute equity metrics, extract trades,
and calculate performance statistics from episode step logs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_equity_metrics(trades_df: pd.DataFrame) -> dict[str, float]:
    """Compute key equity curve metrics from an episode log.
    
    Args:
        trades_df: DataFrame with columns including 'equity'.
            One row per environment step.
    
    Returns:
        Dictionary with keys:
        - final_equity: Ending equity value
        - total_return: Final equity - 1.0 (percentage return)
        - max_drawdown: Maximum peak-to-trough decline (negative value)
    
    Example:
        >>> metrics = compute_equity_metrics(episode_df)
        >>> print(f"Return: {metrics['total_return']:.2%}")
    """
    equity = trades_df["equity"].values
    
    # Final equity and total return
    final_equity = float(equity[-1])
    total_return = final_equity - 1.0
    
    # Max drawdown calculation
    cumulative_peak = np.maximum.accumulate(equity)
    drawdown = equity / cumulative_peak - 1.0
    max_drawdown = float(np.min(drawdown))
    
    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
    }


def compute_return_series(trades_df: pd.DataFrame) -> pd.Series:
    """Compute per-step returns from equity curve.
    
    Args:
        trades_df: DataFrame with 'equity' column.
    
    Returns:
        Series of per-step percentage returns, with first value as 0.
    
    Example:
        >>> returns = compute_return_series(episode_df)
        >>> print(f"Mean return: {returns.mean():.4%}")
    """
    equity_ret = trades_df["equity"].pct_change().fillna(0.0)
    return equity_ret


def compute_sharpe(
    trades_df: pd.DataFrame, 
    steps_per_year: int = 252 * 13  # ~13 30-min bars per day
) -> float:
    """Compute annualized Sharpe ratio from episode returns.
    
    Uses the formula: Sharpe = mean(returns) / std(returns) * sqrt(steps_per_year)
    
    Args:
        trades_df: DataFrame with 'equity' column.
        steps_per_year: Number of steps in a year for annualization.
            Default assumes ~13 30-minute bars per trading day × 252 days.
    
    Returns:
        Annualized Sharpe ratio. Returns 0.0 if std is near zero.
    
    Example:
        >>> sharpe = compute_sharpe(episode_df, steps_per_year=252*13)
        >>> print(f"Sharpe: {sharpe:.2f}")
    """
    returns = compute_return_series(trades_df)
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    if std_ret < 1e-10:
        return 0.0
    
    sharpe = mean_ret / std_ret * np.sqrt(steps_per_year)
    return float(sharpe)


def extract_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Extract individual trades from episode step log.
    
    A trade is defined as a contiguous sequence of steps with the same
    non-zero position direction. Each trade records entry/exit info and PnL.
    
    Args:
        trades_df: DataFrame with columns:
            - timestamp: Step timestamp
            - price: Close price at step
            - position: Position at step (-1, 0, or 1)
            - equity: Equity at step
    
    Returns:
        DataFrame with one row per trade, containing:
        - entry_timestamp: When position was entered
        - exit_timestamp: When position was closed/changed
        - entry_price: Price at entry
        - exit_price: Price at exit
        - direction: +1 for long, -1 for short
        - pnl: Equity change during the trade
        - duration: Number of steps in the trade
    
    Example:
        >>> trades = extract_trades(episode_df)
        >>> print(f"Number of trades: {len(trades)}")
    """
    if trades_df.empty:
        return pd.DataFrame(columns=[
            "entry_timestamp", "exit_timestamp", "entry_price", 
            "exit_price", "direction", "pnl", "duration"
        ])
    
    df = trades_df.reset_index(drop=True)
    
    trades_list = []
    in_trade = False
    entry_idx = 0
    entry_equity = 1.0
    current_direction = 0.0
    
    for i in range(len(df)):
        pos = df.loc[i, "position"]
        
        if not in_trade:
            # Start a new trade if position is non-zero
            if pos != 0:
                in_trade = True
                entry_idx = i
                entry_equity = df.loc[i, "equity"] if i > 0 else 1.0
                current_direction = pos
        else:
            # Check if trade ended (position changed or returned to flat)
            if pos != current_direction:
                # Record the completed trade
                exit_idx = i
                trades_list.append({
                    "entry_timestamp": df.loc[entry_idx, "timestamp"],
                    "exit_timestamp": df.loc[exit_idx, "timestamp"],
                    "entry_price": df.loc[entry_idx, "price"],
                    "exit_price": df.loc[exit_idx, "price"],
                    "direction": int(current_direction),
                    "pnl": df.loc[exit_idx, "equity"] - entry_equity,
                    "duration": exit_idx - entry_idx,
                })
                
                # Check if new trade starts immediately
                if pos != 0:
                    in_trade = True
                    entry_idx = i
                    entry_equity = df.loc[i, "equity"]
                    current_direction = pos
                else:
                    in_trade = False
                    current_direction = 0.0
    
    # Handle trade still open at end of episode
    if in_trade:
        exit_idx = len(df) - 1
        trades_list.append({
            "entry_timestamp": df.loc[entry_idx, "timestamp"],
            "exit_timestamp": df.loc[exit_idx, "timestamp"],
            "entry_price": df.loc[entry_idx, "price"],
            "exit_price": df.loc[exit_idx, "price"],
            "direction": int(current_direction),
            "pnl": df.loc[exit_idx, "equity"] - entry_equity,
            "duration": exit_idx - entry_idx,
        })
    
    if not trades_list:
        return pd.DataFrame(columns=[
            "entry_timestamp", "exit_timestamp", "entry_price", 
            "exit_price", "direction", "pnl", "duration"
        ])
    
    return pd.DataFrame(trades_list)


def compute_trade_stats(trades_summary: pd.DataFrame) -> dict[str, float]:
    """Compute aggregate statistics from extracted trades.
    
    Args:
        trades_summary: DataFrame from extract_trades() with columns
            including 'pnl' and 'duration'.
    
    Returns:
        Dictionary with keys:
        - n_trades: Total number of trades
        - win_rate: Fraction of profitable trades
        - avg_win: Average PnL of winning trades
        - avg_loss: Average PnL of losing trades (negative)
        - profit_factor: Sum of wins / abs(sum of losses)
        - avg_trade_duration: Mean trade duration in steps
    
    Example:
        >>> trades = extract_trades(episode_df)
        >>> stats = compute_trade_stats(trades)
        >>> print(f"Win rate: {stats['win_rate']:.1%}")
    """
    if trades_summary.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_duration": 0.0,
        }
    
    n_trades = len(trades_summary)
    pnls = trades_summary["pnl"].values
    durations = trades_summary["duration"].values
    
    # Separate wins and losses
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    # Win rate
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    
    # Average win/loss
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    
    # Profit factor
    sum_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    sum_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
    
    if sum_losses > 1e-10:
        profit_factor = sum_wins / sum_losses
    elif sum_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    
    # Average duration
    avg_trade_duration = float(np.mean(durations)) if n_trades > 0 else 0.0
    
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_trade_duration": avg_trade_duration,
    }


def print_episode_summary(
    trades_df: pd.DataFrame,
    episode_num: int | None = None,
    steps_per_year: int = 252 * 13,
) -> None:
    """Print a formatted summary of an episode's performance.
    
    Args:
        trades_df: Episode step log DataFrame.
        episode_num: Optional episode number for display.
        steps_per_year: Steps per year for Sharpe calculation.
    """
    equity_metrics = compute_equity_metrics(trades_df)
    sharpe = compute_sharpe(trades_df, steps_per_year)
    trades = extract_trades(trades_df)
    trade_stats = compute_trade_stats(trades)
    
    header = f"Episode {episode_num}" if episode_num is not None else "Episode"
    print(f"\n{'='*50}")
    print(f"{header} Summary")
    print(f"{'='*50}")
    print(f"Steps:         {len(trades_df)}")
    print(f"Final Equity:  {equity_metrics['final_equity']:.4f}")
    print(f"Total Return:  {equity_metrics['total_return']:+.2%}")
    print(f"Max Drawdown:  {equity_metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio:  {sharpe:.2f}")
    print(f"\nTrade Statistics:")
    print(f"  Trades:      {trade_stats['n_trades']}")
    print(f"  Win Rate:    {trade_stats['win_rate']:.1%}")
    print(f"  Avg Win:     {trade_stats['avg_win']:.6f}")
    print(f"  Avg Loss:    {trade_stats['avg_loss']:.6f}")
    print(f"  Profit Factor: {trade_stats['profit_factor']:.2f}")
    print(f"  Avg Duration:  {trade_stats['avg_trade_duration']:.1f} steps")

