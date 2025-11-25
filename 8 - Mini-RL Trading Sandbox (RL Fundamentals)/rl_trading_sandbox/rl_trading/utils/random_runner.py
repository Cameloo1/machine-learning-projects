"""Random policy runner for environment testing and baseline evaluation.

This module provides utilities to run random policies through the trading
environment, useful for sanity checking and establishing baseline performance.
"""

from __future__ import annotations

import gymnasium as gym
import pandas as pd


def run_random_policy(
    env: gym.Env,
    max_episodes: int = 3,
) -> list[pd.DataFrame]:
    """Run a random policy through the environment for multiple episodes.
    
    Samples actions uniformly from the action space and records all step
    information for later analysis.
    
    Args:
        env: Gymnasium environment (should be TradingEnv or compatible).
        max_episodes: Number of episodes to run.
    
    Returns:
        List of DataFrames, one per episode, each containing columns:
        - timestamp: Step timestamp
        - price: Close price
        - action: Action taken (0, 1, or 2)
        - position: Position after action (-1, 0, or 1)
        - step_pnl: PnL for this step
        - equity: Current equity
    
    Example:
        >>> env = make_train_env(train_df, feature_cols, feature_config)
        >>> episode_logs = run_random_policy(env, max_episodes=2)
        >>> for i, log in enumerate(episode_logs):
        ...     print(f"Episode {i}: {len(log)} steps")
    """
    episode_logs = []
    
    for episode in range(max_episodes):
        obs, info = env.reset()
        
        step_records = []
        done = False
        
        while not done:
            # Sample random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Record step info
            step_records.append({
                "timestamp": info["timestamp"],
                "price": info["price"],
                "action": info["action"],
                "position": info["position"],
                "step_pnl": info["step_pnl"],
                "equity": info["equity"],
            })
        
        # Convert to DataFrame
        episode_df = pd.DataFrame(step_records)
        episode_logs.append(episode_df)
    
    return episode_logs


if __name__ == "__main__":
    """Sanity check script for environment and metrics.
    
    This script:
    1. Loads SPY data using SPYDataLoader
    2. Applies feature engineering and scaling
    3. Creates a training environment
    4. Runs random policy episodes
    5. Prints metrics and trade statistics
    
    Usage:
        python -m rl_trading.utils.random_runner
    """
    import sys
    
    print("=" * 60)
    print("Random Policy Runner - Environment Sanity Check")
    print("=" * 60)
    
    # Import modules
    from rl_trading.config import get_default_configs
    from rl_trading.data_loader import SPYDataLoader
    from rl_trading.features import (
        add_basic_features,
        get_feature_columns,
        FeatureScaler,
    )
    from rl_trading.envs import make_env_from_df
    from rl_trading.utils.metrics import (
        compute_equity_metrics,
        compute_sharpe,
        extract_trades,
        compute_trade_stats,
        print_episode_summary,
    )
    
    # Load configs
    print("\n[1] Loading configurations...")
    data_cfg, feat_cfg, train_cfg = get_default_configs()
    print(f"    CSV path: {data_cfg.csv_path}")
    print(f"    Window size: {feat_cfg.window_size}")
    print(f"    Use log returns: {feat_cfg.use_log_returns}")
    
    # Load data
    print("\n[2] Loading SPY data...")
    try:
        loader = SPYDataLoader(data_cfg)
        print(f"    Loaded {len(loader.df)} rows")
        min_ts, max_ts = loader.get_date_range()
        print(f"    Date range: {min_ts.date()} to {max_ts.date()}")
    except FileNotFoundError as e:
        print(f"\n    ERROR: {e}")
        print("\n    To test the environment, please:")
        print("    1. Place your SPY intraday CSV at: data/spy_30m_2019_2025.csv")
        print("    2. Ensure it has columns: timestamp, open, high, low, close, volume")
        print("\n    Running with synthetic data instead...")
        
        # Generate synthetic data for testing
        import numpy as np
        
        np.random.seed(42)
        n_bars = 5000
        
        timestamps = pd.date_range("2019-01-01", periods=n_bars, freq="30min")
        close = 300 + np.random.randn(n_bars).cumsum() * 0.3
        
        synthetic_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": close + np.random.randn(n_bars) * 0.1,
            "high": close + np.abs(np.random.randn(n_bars) * 0.2),
            "low": close - np.abs(np.random.randn(n_bars) * 0.2),
            "close": close,
            "volume": np.random.randint(100000, 1000000, n_bars),
        })
        
        # Use synthetic data for train split
        train_df_raw = synthetic_df.iloc[:3000].copy()
        val_df_raw = synthetic_df.iloc[3000:4000].copy()
        
        print(f"    Generated {len(synthetic_df)} synthetic bars")
        
        # Skip to feature engineering
        loader = None
    
    # Get train/val splits
    if loader is not None:
        print("\n[3] Getting train/val splits...")
        train_df_raw, val_df_raw, test_df_raw = loader.get_splits()
        print(f"    Train: {len(train_df_raw)} rows")
        print(f"    Val:   {len(val_df_raw)} rows")
        print(f"    Test:  {len(test_df_raw)} rows")
    else:
        print("\n[3] Using synthetic train/val data...")
        print(f"    Train: {len(train_df_raw)} rows")
        print(f"    Val:   {len(val_df_raw)} rows")
    
    # Apply feature engineering
    print("\n[4] Applying feature engineering...")
    train_df = add_basic_features(train_df_raw, use_log_returns=feat_cfg.use_log_returns)
    val_df = add_basic_features(val_df_raw, use_log_returns=feat_cfg.use_log_returns)
    print(f"    Train after features: {len(train_df)} rows")
    print(f"    Val after features:   {len(val_df)} rows")
    
    # Scale features
    print("\n[5] Scaling features...")
    feature_cols = get_feature_columns(use_log_returns=feat_cfg.use_log_returns)
    scaler = FeatureScaler()
    train_df = scaler.fit_transform(train_df, feature_cols)
    val_df = scaler.transform(val_df, feature_cols)
    print(f"    Feature columns: {feature_cols}")
    print(f"    Scaler fitted on train data")
    
    # Create training environment
    print("\n[6] Creating training environment...")
    env = make_env_from_df(
        df=train_df,
        feature_cols=feature_cols,
        window_size=feat_cfg.window_size,
        trading_cost_bp=1.0,
        max_episode_steps=500,  # Shorter episodes for testing
        deterministic_start=False,
        seed=train_cfg.seed,
    )
    print(f"    {env}")
    print(f"    Observation space: {env.observation_space.shape}")
    print(f"    Action space: {env.action_space}")
    
    # Run random policy
    print("\n[7] Running random policy for 2 episodes...")
    episode_logs = run_random_policy(env, max_episodes=2)
    
    # Print metrics for each episode
    for i, episode_df in enumerate(episode_logs):
        print_episode_summary(episode_df, episode_num=i + 1)
    
    # Summary statistics across episodes
    print("\n" + "=" * 50)
    print("Aggregate Statistics Across Episodes")
    print("=" * 50)
    
    all_returns = []
    all_n_trades = []
    
    for episode_df in episode_logs:
        metrics = compute_equity_metrics(episode_df)
        trades = extract_trades(episode_df)
        all_returns.append(metrics["total_return"])
        all_n_trades.append(len(trades))
    
    import numpy as np
    print(f"Mean Return:     {np.mean(all_returns):+.2%}")
    print(f"Std Return:      {np.std(all_returns):.2%}")
    print(f"Mean Trades:     {np.mean(all_n_trades):.1f}")
    
    print("\n" + "=" * 60)
    print("Sanity check complete! Environment and metrics working.")
    print("=" * 60)

