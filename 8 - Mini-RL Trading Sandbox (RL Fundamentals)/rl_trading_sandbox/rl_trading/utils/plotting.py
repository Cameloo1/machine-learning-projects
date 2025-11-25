"""Plotting utilities for trading performance visualization.

This module provides functions to create standardized plots for analyzing
trading strategy performance, including equity curves and action distributions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


def plot_equity_curve(
    trades_df: pd.DataFrame,
    buy_and_hold_equity: pd.Series,
    output_path: Path,
    title: str = "Equity Curve: RL Strategy vs Buy-and-Hold",
) -> None:
    """Plot RL equity curve alongside buy-and-hold baseline.
    
    Args:
        trades_df: DataFrame with 'timestamp' and 'equity' columns from RL agent.
        buy_and_hold_equity: Series indexed by timestamp with buy-and-hold equity.
        output_path: Path to save the plot (e.g., 'equity_curve_test.png').
        title: Plot title.
    
    Example:
        >>> plot_equity_curve(trades_df, bh_equity, Path("plots/equity.png"))
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamps if needed
    rl_timestamps = pd.to_datetime(trades_df["timestamp"])
    rl_equity = trades_df["equity"].values
    
    bh_timestamps = pd.to_datetime(buy_and_hold_equity.index)
    bh_equity = buy_and_hold_equity.values
    
    # Plot RL equity
    ax.plot(rl_timestamps, rl_equity, label="RL Strategy", color="#2E86AB", linewidth=1.5)
    
    # Plot buy-and-hold equity
    ax.plot(bh_timestamps, bh_equity, label="Buy-and-Hold", color="#A23B72", 
            linewidth=1.5, linestyle="--", alpha=0.8)
    
    # Add horizontal line at 1.0
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Starting Equity")
    
    # Formatting
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Equity (normalized)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Compute and annotate final returns
    rl_return = (rl_equity[-1] - 1.0) * 100
    bh_return = (bh_equity[-1] - 1.0) * 100
    
    textstr = f"RL Return: {rl_return:+.2f}%\nB&H Return: {bh_return:+.2f}%"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_action_distribution(
    trades_df: pd.DataFrame,
    output_path: Path,
    title: str = "Action Distribution",
) -> None:
    """Plot bar chart of action distribution.
    
    Args:
        trades_df: DataFrame with 'action' column (values 0, 1, 2, 3).
        output_path: Path to save the plot.
        title: Plot title.
    
    Example:
        >>> plot_action_distribution(trades_df, Path("plots/actions.png"))
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Count actions
    action_counts = trades_df["action"].value_counts().sort_index()
    
    # Ensure all actions are represented
    for action in [0, 1, 2, 3]:
        if action not in action_counts.index:
            action_counts[action] = 0
    action_counts = action_counts.sort_index()
    
    # Labels and colors
    labels = ["Hold (0)", "Long (1)", "Short (2)", "Close (3)"]
    colors = ["#6C757D", "#28A745", "#DC3545", "#FFC107"]
    
    # Create bar chart
    bars = ax.bar(labels, action_counts.values, color=colors, edgecolor="black", alpha=0.8)
    
    # Add value labels on bars
    for bar, count in zip(bars, action_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({count/len(trades_df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # Formatting
    ax.set_xlabel("Action", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_pnl_histogram(
    trades_df: pd.DataFrame,
    output_path: Path,
    title: str = "Step PnL Distribution",
) -> None:
    """Plot histogram of per-step PnL values.
    
    Args:
        trades_df: DataFrame with 'step_pnl' column.
        output_path: Path to save the plot.
        title: Plot title.
    
    Example:
        >>> plot_pnl_histogram(trades_df, Path("plots/pnl_hist.png"))
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    pnl_values = trades_df["step_pnl"].values
    
    # Compute stats
    mean_pnl = np.mean(pnl_values)
    std_pnl = np.std(pnl_values)
    positive_pct = (pnl_values > 0).sum() / len(pnl_values) * 100
    
    # Create histogram
    n, bins, patches = ax.hist(pnl_values, bins=50, color="#2E86AB", 
                                edgecolor="black", alpha=0.7)
    
    # Color positive/negative bins differently
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor("#DC3545")
        else:
            patch.set_facecolor("#28A745")
    
    # Add vertical line at zero
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    
    # Add mean line
    ax.axvline(x=mean_pnl, color="#FF8C00", linestyle="-", linewidth=2, 
               label=f"Mean: {mean_pnl:.6f}")
    
    # Formatting
    ax.set_xlabel("Step PnL", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Stats annotation
    textstr = f"Mean: {mean_pnl:.6f}\nStd: {std_pnl:.6f}\nPositive: {positive_pct:.1f}%"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_position_over_time(
    trades_df: pd.DataFrame,
    output_path: Path,
    title: str = "Position Over Time",
) -> None:
    """Plot position changes over time.
    
    Args:
        trades_df: DataFrame with 'timestamp' and 'position' columns.
        output_path: Path to save the plot.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    timestamps = pd.to_datetime(trades_df["timestamp"])
    positions = trades_df["position"].values
    
    # Create step plot for positions
    ax.fill_between(timestamps, positions, 0, where=(positions > 0), 
                    color="#28A745", alpha=0.5, label="Long", step="pre")
    ax.fill_between(timestamps, positions, 0, where=(positions < 0), 
                    color="#DC3545", alpha=0.5, label="Short", step="pre")
    ax.step(timestamps, positions, where="pre", color="black", linewidth=0.5)
    
    # Formatting
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Position", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Short", "Flat", "Long"])
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _draw_candlestick(ax, timestamp, open_price, high_price, low_price, close_price, width=0.8):
    """Draw a single candlestick on the axis.
    
    Args:
        ax: Matplotlib axis
        timestamp: Timestamp for the candle
        open_price: Open price
        high_price: High price
        low_price: Low price
        close_price: Close price
        width: Width of the candle body
    """
    # Determine if bullish (green) or bearish (red)
    color = "#28A745" if close_price >= open_price else "#DC3545"
    
    # Draw the wick (high-low line)
    ax.plot([timestamp, timestamp], [low_price, high_price], 
            color="black", linewidth=0.5, zorder=1)
    
    # Draw the body (open-close rectangle)
    body_low = min(open_price, close_price)
    body_high = max(open_price, close_price)
    body_height = body_high - body_low
    
    # Use rectangle for body
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (mdates.date2num(timestamp) - width/2, body_low),
        width,
        body_height,
        facecolor=color,
        edgecolor="black",
        linewidth=0.5,
        zorder=2
    )
    ax.add_patch(rect)


def plot_candlestick_with_trades(
    trades_df: pd.DataFrame,
    output_path: Path,
    price_df: pd.DataFrame | None = None,
    title: str = "Candlestick Chart with Trades",
    max_bars: int = 500,
) -> None:
    """Plot candlestick chart with all trades overlaid.
    
    Extracts trade entry/exit points from trades_df and overlays them on
    a candlestick chart. If price_df is not provided, creates synthetic
    OHLC from trades_df price column.
    
    Args:
        trades_df: DataFrame with 'timestamp', 'price', 'action', 'position' columns.
        price_df: Optional DataFrame with OHLC data. If None, creates from trades_df.
        output_path: Path to save the plot.
        title: Plot title.
        max_bars: Maximum number of bars to display (for performance).
    
    Example:
        >>> plot_candlestick_with_trades(trades_df, price_df, Path("candles.png"))
    """
    # Prepare price data
    if price_df is None:
        # Create synthetic OHLC from trades_df price column
        price_df = trades_df[["timestamp", "price"]].copy()
        price_df["open"] = price_df["price"].shift(1).fillna(price_df["price"])
        price_df["high"] = price_df[["open", "price"]].max(axis=1) * 1.001
        price_df["low"] = price_df[["open", "price"]].min(axis=1) * 0.999
        price_df["close"] = price_df["price"]
        price_df = price_df.rename(columns={"price": "close"})
    
    # Ensure we have required columns
    required_cols = ["timestamp", "open", "high", "low", "close"]
    missing = set(required_cols) - set(price_df.columns)
    if missing:
        raise ValueError(f"price_df missing required columns: {missing}")
    
    # Convert timestamps to datetime first (work on copies)
    price_df = price_df.copy()
    trades_df = trades_df.copy()
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
    
    # Align price_df with trades_df timestamps (only show data where trades occurred)
    trade_timestamps = set(trades_df["timestamp"].values)
    price_df = price_df[price_df["timestamp"].isin(trade_timestamps)].copy()
    
    if len(price_df) == 0:
        # Fallback: use trades_df timestamps and create synthetic OHLC
        price_df = trades_df[["timestamp", "price"]].copy()
        price_df["open"] = price_df["price"].shift(1).fillna(price_df["price"])
        price_df["high"] = price_df[["open", "price"]].max(axis=1) * 1.001
        price_df["low"] = price_df[["open", "price"]].min(axis=1) * 0.999
        price_df["close"] = price_df["price"]
        price_df = price_df.drop(columns=["price"])
    
    # Sort and limit bars
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    if len(price_df) > max_bars:
        # Sample evenly
        indices = np.linspace(0, len(price_df) - 1, max_bars, dtype=int)
        price_df = price_df.iloc[indices].reset_index(drop=True)
    
    # Convert timestamps
    timestamps = price_df["timestamp"]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
    
    # Main candlestick chart
    ax1 = fig.add_subplot(gs[0])
    
    # Draw candlesticks
    candle_times = []
    for idx, row in price_df.iterrows():
        ts = mdates.date2num(row["timestamp"])
        candle_times.append(ts)
        _draw_candlestick(
            ax1, ts, row["open"], row["high"], row["low"], row["close"]
        )
    
    # Set x-axis limits to actual data range
    if candle_times:
        min_time = min(candle_times)
        max_time = max(candle_times)
        # Add small padding
        time_range = max_time - min_time
        ax1.set_xlim(min_time - time_range * 0.01, max_time + time_range * 0.01)
    
    # Extract trade entry/exit points
    trades_df_sorted = trades_df.sort_values("timestamp").reset_index(drop=True).copy()
    # Ensure timestamp is datetime (already done above, but ensure it's consistent)
    if not pd.api.types.is_datetime64_any_dtype(trades_df_sorted["timestamp"]):
        trades_df_sorted["timestamp"] = pd.to_datetime(trades_df_sorted["timestamp"])
    
    trade_entries = []
    trade_exits = []
    
    prev_position = 0.0
    for idx, row in trades_df_sorted.iterrows():
        current_position = row["position"]
        action = row["action"]
        price = row["price"]
        timestamp = row["timestamp"]  # Already datetime
        
        # Detect position changes (entry/exit)
        if prev_position == 0 and current_position != 0:
            # Entry
            trade_entries.append({
                "timestamp": timestamp,
                "price": price,
                "position": current_position,
                "action": action,
            })
        elif prev_position != 0 and current_position == 0:
            # Exit
            trade_exits.append({
                "timestamp": timestamp,
                "price": price,
                "position": prev_position,
            })
        elif prev_position != current_position and current_position != 0:
            # Position change (e.g., long to short)
            trade_exits.append({
                "timestamp": timestamp,
                "price": price,
                "position": prev_position,
            })
            trade_entries.append({
                "timestamp": timestamp,
                "price": price,
                "position": current_position,
                "action": action,
            })
        
        prev_position = current_position
    
    # Plot trade entries
    if trade_entries:
        entry_df = pd.DataFrame(trade_entries)
        entry_times = [mdates.date2num(ts) for ts in entry_df["timestamp"]]
        
        # Long entries (green up arrow)
        long_entries = entry_df[entry_df["position"] > 0]
        if len(long_entries) > 0:
            long_times = [mdates.date2num(ts) for ts in long_entries["timestamp"]]
            ax1.scatter(long_times, long_entries["price"], 
                       color="#28A745", marker="^", s=100, 
                       edgecolors="black", linewidths=1, 
                       zorder=5, label="Long Entry", alpha=0.8)
        
        # Short entries (red down arrow)
        short_entries = entry_df[entry_df["position"] < 0]
        if len(short_entries) > 0:
            short_times = [mdates.date2num(ts) for ts in short_entries["timestamp"]]
            ax1.scatter(short_times, short_entries["price"], 
                       color="#DC3545", marker="v", s=100, 
                       edgecolors="black", linewidths=1, 
                       zorder=5, label="Short Entry", alpha=0.8)
    
    # Plot trade exits
    if trade_exits:
        exit_df = pd.DataFrame(trade_exits)
        exit_times = [mdates.date2num(ts) for ts in exit_df["timestamp"]]
        
        # Long exits
        long_exits = exit_df[exit_df["position"] > 0]
        if len(long_exits) > 0:
            long_exit_times = [mdates.date2num(ts) for ts in long_exits["timestamp"]]
            ax1.scatter(long_exit_times, long_exits["price"], 
                       color="#28A745", marker="x", s=150, 
                       linewidths=2, zorder=5, label="Long Exit", alpha=0.8)
        
        # Short exits
        short_exits = exit_df[exit_df["position"] < 0]
        if len(short_exits) > 0:
            short_exit_times = [mdates.date2num(ts) for ts in short_exits["timestamp"]]
            ax1.scatter(short_exit_times, short_exits["price"], 
                       color="#DC3545", marker="x", s=150, 
                       linewidths=2, zorder=5, label="Short Exit", alpha=0.8)
    
    # Format main chart
    ax1.set_ylabel("Price", fontsize=11)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    # Position subplot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    pos_times = [mdates.date2num(ts) for ts in trades_df_sorted["timestamp"]]
    positions = trades_df_sorted["position"].values
    
    ax2.fill_between(pos_times, positions, 0, where=(positions > 0), 
                    color="#28A745", alpha=0.3, step="pre")
    ax2.fill_between(pos_times, positions, 0, where=(positions < 0), 
                    color="#DC3545", alpha=0.3, step="pre")
    ax2.step(pos_times, positions, where="pre", color="black", linewidth=0.5)
    ax2.set_ylabel("Position", fontsize=10)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_yticks([-1, 0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Volume subplot (if available)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if "volume" in price_df.columns:
        vol_times = [mdates.date2num(ts) for ts in timestamps]
        volumes = price_df["volume"].values
        ax3.bar(vol_times, volumes, width=0.8, color="#6C757D", alpha=0.6)
        ax3.set_ylabel("Volume", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "Volume data not available", 
                transform=ax3.transAxes, ha="center", va="center")
        ax3.set_ylabel("Volume", fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis on bottom subplot
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # Set reasonable date locator based on data range
    if candle_times and len(candle_times) > 0:
        time_range_days = (max_time - min_time)
        if time_range_days < 1:
            # Intraday: show hours
            ax3.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_range_days * 24 / 10))))
        elif time_range_days < 7:
            # Few days: show days
            ax3.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        elif time_range_days < 30:
            # Few weeks: show days
            ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(time_range_days / 10))))
        else:
            # Longer: show dates
            ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    else:
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    
    fig.autofmt_xdate()
    
    # Add trade count annotation
    n_entries = len(trade_entries) if trade_entries else 0
    n_exits = len(trade_exits) if trade_exits else 0
    textstr = f"Trades: {n_entries} entries, {n_exits} exits"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax1.text(0.98, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

