"""Custom Gymnasium trading environment for SPY intraday data.

This module implements a discrete-action trading environment where an agent
can take long, short, or flat positions based on windowed price features.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_trading.config import FeatureConfig


class TradingEnv(gym.Env):
    """Gymnasium environment for discrete position trading on SPY intraday data.
    
    The agent observes a rolling window of features plus position/trade context,
    and selects actions from four options: hold (0), long (1), short (2),
    or explicit close (3).
    
    Reward is the per-step PnL accounting for position changes and trading costs.
    
    Attributes:
        metadata: Gymnasium metadata dict.
        price_df: DataFrame with OHLCV and feature columns.
        feature_cols: List of feature column names.
        window_size: Number of historical bars in each observation.
        trading_cost_bp: Trading cost in basis points per position change.
        max_episode_steps: Maximum steps per episode.
        deterministic_start: If True, always start at earliest valid index.
        observation_space: Gymnasium Box space for observations.
        action_space: Gymnasium Discrete space with 3 actions.
    
    Example:
        >>> env = TradingEnv(price_df, feature_cols=['ret_log_1', 'vol_20'])
        >>> obs, info = env.reset()
        >>> obs, reward, done, truncated, info = env.step(1)  # go long
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        price_df: pd.DataFrame,
        feature_cols: list[str],
        window_size: int = 30,
        trading_cost_bp: float = 1.0,
        max_episode_steps: int = 2000,
        seed: int | None = None,
        deterministic_start: bool = False,
    ) -> None:
        """Initialize the trading environment.
        
        Args:
            price_df: DataFrame with timestamp, OHLCV, and feature columns.
                Must be sorted by timestamp ascending and have no NaN values
                in feature columns.
            feature_cols: Ordered list of feature column names to use in obs.
            window_size: Number of past bars to include in observation.
            trading_cost_bp: Per-trade cost in basis points.
            max_episode_steps: Max steps per episode (truncates if reached).
            seed: Random seed for reproducibility.
            deterministic_start: If True, start at earliest valid index;
                if False, random start each episode.
        
        Raises:
            ValueError: If required columns are missing or data is too short.
        """
        super().__init__()
        
        self._validate_dataframe(price_df, feature_cols, window_size)
        
        # Store parameters
        self.price_df = price_df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.trading_cost_bp = trading_cost_bp
        self.max_episode_steps = max_episode_steps
        self.deterministic_start = deterministic_start
        
        # Pre-extract numpy arrays for fast access
        self._feature_array = self.price_df[feature_cols].values.astype(np.float32)
        self._close_array = self.price_df["close"].values.astype(np.float64)
        self._timestamps = self.price_df["timestamp"].values
        
        # Compute valid start range
        self._min_start_idx = window_size - 1  # Need window_size bars including current
        self._max_start_idx = len(self.price_df) - max_episode_steps - 1
        
        if self._max_start_idx < self._min_start_idx:
            # Data is shorter than window + max_episode_steps
            self._max_start_idx = len(self.price_df) - 2
        
        # Define spaces (+1 for position, +1 for trade age, +1 trade_ret, +1 drawdown)
        n_features = len(feature_cols) + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, n_features),
            dtype=np.float32,
        )
        # 0: hold current position, 1: go/hold long, 2: go/hold short, 3: explicit close
        self.action_space = spaces.Discrete(4)
        
        # Initialize state variables (will be set in reset)
        self.current_step: int = 0
        self.episode_step_count: int = 0
        self.position: float = 0.0
        self.equity: float = 1.0
        self.last_price: float = 0.0
        self.trade_age: int = 0  # number of steps the current position has been held
        self.entry_price: float | None = None
        self.entry_equity: float | None = None
        self.trade_ret: float = 0.0
        self.trade_mfe: float = 0.0
        
        # Set random seed
        self._rng = np.random.default_rng(seed)
    
    def _validate_dataframe(
        self, 
        df: pd.DataFrame, 
        feature_cols: list[str],
        window_size: int,
    ) -> None:
        """Validate that DataFrame has required columns."""
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing_base = required - set(df.columns)
        if missing_base:
            raise ValueError(f"DataFrame missing required columns: {missing_base}")
        
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"DataFrame missing feature columns: {missing_features}")
        
        if len(df) < window_size + 10:
            raise ValueError(
                f"DataFrame has {len(df)} rows, need at least "
                f"{window_size + 10} for window_size={window_size}"
            )
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to start a new episode.
        
        Args:
            seed: Optional seed to reset the RNG.
            options: Optional dict (unused).
        
        Returns:
            Tuple of (observation, info_dict).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Choose starting index
        if self.deterministic_start:
            self.current_step = self._min_start_idx
        else:
            self.current_step = self._rng.integers(
                self._min_start_idx, 
                self._max_start_idx + 1
            )
        
        # Reset state
        self.episode_step_count = 0
        self.position = 0.0
        self.equity = 1.0
        self.last_price = self._close_array[self.current_step]
        self.trade_age = 0
        self.entry_price = None
        self.entry_equity = None
        self.trade_ret = 0.0
        self.trade_mfe = 0.0
        
        obs = self._get_observation()
        info = self._get_info(step_pnl=0.0, action=0, trade_drawdown=0.0)
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation array from current state."""
        # Get feature window: [current_step - window_size + 1, current_step] inclusive
        start_idx = self.current_step - self.window_size + 1
        end_idx = self.current_step + 1
        
        features = self._feature_array[start_idx:end_idx]  # (window_size, n_features)
        
        # Create position column
        position_col = np.full(
            (self.window_size, 1), 
            self.position, 
            dtype=np.float32
        )
        
        # Normalized trade age column (0 when flat/new, approaches 1 near window size)
        age_norm = float(self.trade_age) / float(self.window_size)
        age_col = np.full(
            (self.window_size, 1),
            age_norm,
            dtype=np.float32,
        )
        
        trade_ret_col = np.full(
            (self.window_size, 1),
            self.trade_ret,
            dtype=np.float32,
        )
        trade_dd = self.trade_ret - self.trade_mfe if self.trade_mfe != 0.0 else 0.0
        trade_dd_col = np.full(
            (self.window_size, 1),
            trade_dd,
            dtype=np.float32,
        )
        
        # Concatenate
        obs = np.concatenate(
            [features, position_col, age_col, trade_ret_col, trade_dd_col],
            axis=1,
        )
        
        return obs.astype(np.float32)
    
    def _get_info(self, step_pnl: float, action: int, trade_drawdown: float = 0.0) -> dict[str, Any]:
        """Build info dict for current state."""
        return {
            "timestamp": self._timestamps[self.current_step],
            "price": float(self._close_array[self.current_step]),
            "step_pnl": float(step_pnl),
            "equity": float(self.equity),
            "position": float(self.position),
            "action": int(action),
            "trade_ret": float(self.trade_ret),
            "trade_drawdown": float(trade_drawdown),
        }
    
    def step(
        self, 
        action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Integer action (0=flat, 1=long, 2=short).
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self.action_space.contains(action), "Invalid action"
        
        current_position = self.position
        if action == 0:
            target_position = current_position
        elif action == 1:
            target_position = 1.0
        elif action == 2:
            target_position = -1.0
        elif action == 3:
            target_position = 0.0
        else:
            raise ValueError(f"Unexpected action: {action}")
        
        trade_size = abs(target_position - current_position)
        trade_cost = trade_size * (self.trading_cost_bp / 10_000)
        
        is_closing = current_position != 0.0 and target_position == 0.0
        is_opening = current_position == 0.0 and target_position != 0.0
        is_flip = (
            current_position != 0.0
            and target_position != 0.0
            and np.sign(current_position) != np.sign(target_position)
        )
        
        if is_opening or is_closing or is_flip or target_position == 0.0:
            self.trade_age = 0
        else:
            self.trade_age += 1
        
        if is_opening or is_flip:
            self.entry_price = float(self._close_array[self.current_step])
            self.entry_equity = self.equity
            self.trade_ret = 0.0
            self.trade_mfe = 0.0
        
        # Update position
        self.position = target_position
        
        # Move to next step
        self.current_step += 1
        self.episode_step_count += 1
        
        # Compute price return
        current_price = self._close_array[self.current_step]
        price_return = (current_price - self.last_price) / self.last_price
        
        if self.position != 0.0 and self.entry_price is not None:
            signed = np.sign(self.position)
            self.trade_ret = ((current_price - self.entry_price) / self.entry_price) * signed
            self.trade_mfe = max(self.trade_mfe, self.trade_ret)
            trade_drawdown = self.trade_ret - self.trade_mfe
        else:
            self.trade_ret = 0.0
            trade_drawdown = 0.0
        
        trade_penalty = 0.0001 if trade_size > 0.0 else 0.0
        exposure_penalty = 0.0
        flip_penalty = 0.0001 if is_flip else 0.0
        
        exit_bonus = 0.0
        if is_closing and self.entry_equity not in (None, 0.0):
            realized_ret = self.equity / self.entry_equity - 1.0
            exit_bonus = 0.001 * realized_ret
        
        # Compute step PnL (position * return - costs - penalties + exit bonus)
        step_pnl = (
            self.position * price_return
            - trade_cost
            - trade_penalty
            - exposure_penalty
            - flip_penalty
        )
        step_pnl += exit_bonus
        
        # Update equity multiplicatively
        self.equity *= (1.0 + step_pnl)
        
        # Update last price
        self.last_price = current_price
        
        if target_position == 0.0:
            self.entry_price = None
            self.entry_equity = None
            self.trade_mfe = 0.0
        
        # Check termination
        terminated = (
            self.episode_step_count >= self.max_episode_steps or
            self.current_step >= len(self.price_df) - 1
        )
        truncated = False
        
        # Build observation and info
        obs = self._get_observation()
        info = self._get_info(step_pnl=step_pnl, action=action, trade_drawdown=trade_drawdown)
        
        # Reward is the step PnL
        reward = float(step_pnl)
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> None:
        """Render the environment (no-op for now)."""
        pass
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def __repr__(self) -> str:
        return (
            f"TradingEnv(rows={len(self.price_df)}, "
            f"features={len(self.feature_cols)}, "
            f"window={self.window_size}, "
            f"cost_bp={self.trading_cost_bp})"
        )


def make_env_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int = 30,
    trading_cost_bp: float = 1.0,
    max_episode_steps: int = 2000,
    deterministic_start: bool = False,
    seed: int | None = None,
) -> TradingEnv:
    """Factory function to create a TradingEnv from a DataFrame.
    
    Args:
        df: DataFrame with OHLCV and feature columns.
        feature_cols: List of feature column names.
        window_size: Number of bars in observation window.
        trading_cost_bp: Trading cost in basis points.
        max_episode_steps: Maximum steps per episode.
        deterministic_start: If True, start at fixed index.
        seed: Random seed.
    
    Returns:
        Configured TradingEnv instance.
    """
    return TradingEnv(
        price_df=df,
        feature_cols=feature_cols,
        window_size=window_size,
        trading_cost_bp=trading_cost_bp,
        max_episode_steps=max_episode_steps,
        seed=seed,
        deterministic_start=deterministic_start,
    )


def make_train_env(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    feature_config: FeatureConfig | None = None,
    trading_cost_bp: float = 1.0,
    max_episode_steps: int = 2000,
    seed: int = 42,
) -> TradingEnv:
    """Create a training environment with random episode starts.
    
    Args:
        train_df: Training DataFrame with features applied and scaled.
        feature_cols: Feature column names.
        feature_config: Optional FeatureConfig for window_size.
        trading_cost_bp: Trading cost in basis points.
        max_episode_steps: Max steps per episode.
        seed: Random seed.
    
    Returns:
        TradingEnv configured for training.
    """
    window_size = feature_config.window_size if feature_config else 30
    
    return make_env_from_df(
        df=train_df,
        feature_cols=feature_cols,
        window_size=window_size,
        trading_cost_bp=trading_cost_bp,
        max_episode_steps=max_episode_steps,
        deterministic_start=False,
        seed=seed,
    )


def make_val_env(
    val_df: pd.DataFrame,
    feature_cols: list[str],
    feature_config: FeatureConfig | None = None,
    trading_cost_bp: float = 1.0,
    max_episode_steps: int = 2000,
    seed: int | None = None,
) -> TradingEnv:
    """Create a validation environment with deterministic start.
    
    Args:
        val_df: Validation DataFrame with features applied and scaled.
        feature_cols: Feature column names.
        feature_config: Optional FeatureConfig for window_size.
        trading_cost_bp: Trading cost in basis points.
        max_episode_steps: Max steps per episode.
        seed: Random seed (less relevant since deterministic).
    
    Returns:
        TradingEnv configured for validation.
    """
    window_size = feature_config.window_size if feature_config else 30
    
    return make_env_from_df(
        df=val_df,
        feature_cols=feature_cols,
        window_size=window_size,
        trading_cost_bp=trading_cost_bp,
        max_episode_steps=max_episode_steps,
        deterministic_start=True,
        seed=seed,
    )


def make_test_env(
    test_df: pd.DataFrame,
    feature_cols: list[str],
    feature_config: FeatureConfig | None = None,
    trading_cost_bp: float = 1.0,
    max_episode_steps: int = 2000,
    seed: int | None = None,
) -> TradingEnv:
    """Create a test environment with deterministic start.
    
    Args:
        test_df: Test DataFrame with features applied and scaled.
        feature_cols: Feature column names.
        feature_config: Optional FeatureConfig for window_size.
        trading_cost_bp: Trading cost in basis points.
        max_episode_steps: Max steps per episode.
        seed: Random seed.
    
    Returns:
        TradingEnv configured for testing.
    """
    window_size = feature_config.window_size if feature_config else 30
    
    return make_env_from_df(
        df=test_df,
        feature_cols=feature_cols,
        window_size=window_size,
        trading_cost_bp=trading_cost_bp,
        max_episode_steps=max_episode_steps,
        deterministic_start=True,
        seed=seed,
    )
