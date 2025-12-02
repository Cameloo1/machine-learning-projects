"""Configuration dataclasses for the RL Trading Sandbox.

This module provides configuration structures for data loading, feature engineering,
and training hyperparameters. All configs use sensible defaults that can be overridden.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for data loading and train/val/test splits.
    
    Attributes:
        csv_path: Path to the SPY intraday CSV file.
        train_start: Start date for training data (inclusive).
        train_end: End date for training data (inclusive).
        val_start: Start date for validation data (inclusive).
        val_end: End date for validation data (inclusive).
        test_start: Start date for test data (inclusive).
        test_end: End date for test data (inclusive). None means up to latest.
    """
    csv_path: str = "data/spy_daily_2019_2024.csv"
    train_start: str = "2019-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2024-06-30"
    test_start: str = "2024-07-01"
    test_end: str | None = None


@dataclass
class FeatureConfig:
    """Configuration for feature engineering.
    
    Attributes:
        feat_cols: List of feature column names to use. Populated after feature
            engineering; defaults to empty list.
        window_size: Number of historical bars to include in observation window.
        use_log_returns: If True, use log returns; otherwise use simple returns.
    """
    feat_cols: list[str] = field(default_factory=list)
    window_size: int = 60
    use_log_returns: bool = True


@dataclass
class TrainingConfig:
    """Configuration for RL training hyperparameters.
    
    Designed primarily for DQN-style algorithms but applicable to other methods.
    
    Attributes:
        seed: Random seed for reproducibility.
        total_timesteps: Total environment steps for training.
        batch_size: Minibatch size for gradient updates.
        gamma: Discount factor for future rewards.
        learning_rate: Learning rate for the optimizer.
        buffer_size: Size of the replay buffer.
        learning_starts: Number of steps before learning begins.
        train_freq: Frequency of gradient updates (in steps).
        target_update_interval: Frequency of target network updates.
        exploration_fraction: Fraction of training for epsilon decay.
        exploration_initial_eps: Initial epsilon for exploration.
        exploration_final_eps: Final epsilon after decay.
        max_grad_norm: Maximum gradient norm for clipping.
        num_epochs: Number of training epochs for periodic validation.
        eval_episodes: Number of episodes per validation evaluation.
        policy_hidden_sizes: Hidden-layer sizes for policy network.
        patience: Early stopping patience (epochs without improvement).
    """
    seed: int = 995642
    total_timesteps: int = 150_000
    batch_size: int = 32
    gamma: float = 0.965  # Slightly longer horizon for financial rewards
    learning_rate: float = 1e-4  # Lower LR for stability
    buffer_size: int = 300_000  # Larger buffer
    learning_starts: int = 1_000
    train_freq: int = 4
    target_update_interval: int = 10_000  # Less frequent target updates
    exploration_fraction: float = 0.2  # Longer exploration
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02  # Lower final epsilon
    max_grad_norm: float = 10.0
    num_epochs: int = 10
    eval_episodes: int = 10  # Multiple validation episodes
    policy_hidden_sizes: tuple[int, int] = (256, 256)
    patience: int = 4  # Early stopping patience
    
    def to_dqn_kwargs(self) -> dict:
        """Convert config to stable-baselines3 DQN constructor kwargs.
        
        Returns:
            Dictionary of kwargs suitable for DQN initialization.
        
        Example:
            >>> config = TrainingConfig()
            >>> dqn = DQN("MlpPolicy", env, **config.to_dqn_kwargs())
        """
        return {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "target_update_interval": self.target_update_interval,
            "exploration_fraction": self.exploration_fraction,
            "exploration_final_eps": self.exploration_final_eps,
            "policy_kwargs": {
                "net_arch": list(self.policy_hidden_sizes),
            },
            "verbose": 0,
        }
    
    def to_dict(self) -> dict:
        """Convert config to a plain dictionary for serialization.
        
        Returns:
            Dictionary with all config values.
        """
        return {
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "target_update_interval": self.target_update_interval,
            "exploration_fraction": self.exploration_fraction,
            "exploration_initial_eps": self.exploration_initial_eps,
            "exploration_final_eps": self.exploration_final_eps,
            "max_grad_norm": self.max_grad_norm,
            "num_epochs": self.num_epochs,
            "eval_episodes": self.eval_episodes,
            "patience": self.patience,
            "policy_hidden_sizes": self.policy_hidden_sizes,
        }


def get_default_configs() -> tuple[DataConfig, FeatureConfig, TrainingConfig]:
    """Returns default configurations for data, features, and training.
    
    Returns:
        A tuple of (DataConfig, FeatureConfig, TrainingConfig) with sensible defaults.
    
    Example:
        >>> data_cfg, feat_cfg, train_cfg = get_default_configs()
        >>> print(data_cfg.csv_path)
        data/spy_30m_2019_2025.csv
    """
    data_config = DataConfig()
    feature_config = FeatureConfig()
    training_config = TrainingConfig()
    
    return data_config, feature_config, training_config

