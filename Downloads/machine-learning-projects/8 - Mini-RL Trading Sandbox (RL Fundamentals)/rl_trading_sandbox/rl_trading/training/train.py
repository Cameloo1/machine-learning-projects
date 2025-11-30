"""DQN training script for the RL Trading Sandbox.

This module provides the main training loop for DQN agents on SPY intraday data,
with periodic validation evaluation and best model checkpointing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import DQN

from rl_trading.config import get_default_configs, DataConfig, FeatureConfig, TrainingConfig
from rl_trading.data_loader import SPYDataLoader
from rl_trading.features import add_basic_features, get_feature_columns, FeatureScaler
from rl_trading.envs import make_env_from_df
from rl_trading.training.eval import evaluate_model_on_env, STEPS_PER_YEAR
from rl_trading.utils.logging_utils import (
    create_experiment_folder,
    save_config,
    append_metrics_row,
    save_json,
)

VAL_EPISODES = 10
LR_DECAY_EPOCHS = 5


def _prepare_data(
    data_config: DataConfig,
    feature_config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureScaler, list[str]]:
    """Load and prepare train/val data with features.
    
    Args:
        data_config: Data loading configuration.
        feature_config: Feature engineering configuration.
    
    Returns:
        Tuple of (train_df, val_df, scaler, feature_cols).
    """
    print("[1] Loading SPY data...")
    loader = SPYDataLoader(data_config)
    train_df_raw, val_df_raw, _ = loader.get_splits()
    print(f"    Train raw: {len(train_df_raw)} rows")
    print(f"    Val raw:   {len(val_df_raw)} rows")
    
    print("[2] Applying feature engineering...")
    train_df = add_basic_features(train_df_raw, use_log_returns=feature_config.use_log_returns)
    val_df = add_basic_features(val_df_raw, use_log_returns=feature_config.use_log_returns)
    print(f"    Train with features: {len(train_df)} rows")
    print(f"    Val with features:   {len(val_df)} rows")
    
    print("[3] Scaling features...")
    feature_cols = get_feature_columns(use_log_returns=feature_config.use_log_returns)
    scaler = FeatureScaler()
    train_df = scaler.fit_transform(train_df, feature_cols)
    val_df = scaler.transform(val_df, feature_cols)
    print(f"    Feature columns: {feature_cols}")
    
    return train_df, val_df, scaler, feature_cols


def _create_envs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    trading_cost_bp: float = 1.0,
    max_episode_steps: int = 2000,
) -> tuple:
    """Create training and validation environments.
    
    Args:
        train_df: Training DataFrame with scaled features.
        val_df: Validation DataFrame with scaled features.
        feature_cols: List of feature column names.
        feature_config: Feature configuration.
        training_config: Training configuration.
        trading_cost_bp: Trading cost in basis points.
        max_episode_steps: Maximum steps per episode.
    
    Returns:
        Tuple of (train_env, val_env).
    """
    print("[4] Creating environments...")
    
    train_env = make_env_from_df(
        df=train_df,
        feature_cols=feature_cols,
        window_size=feature_config.window_size,
        trading_cost_bp=trading_cost_bp,
        max_episode_steps=max_episode_steps,
        deterministic_start=False,
        seed=training_config.seed,
    )
    
    val_env = make_env_from_df(
        df=val_df,
        feature_cols=feature_cols,
        window_size=feature_config.window_size,
        trading_cost_bp=trading_cost_bp,
        max_episode_steps=max_episode_steps,
        deterministic_start=True,
        seed=training_config.seed,
    )
    
    print(f"    Train env: {train_env}")
    print(f"    Val env:   {val_env}")
    print(f"    Observation space: {train_env.observation_space.shape}")
    print(f"    Action space: {train_env.action_space}")
    
    return train_env, val_env


def train_dqn_with_configs(
    exp_name: str,
    data_config: DataConfig,
    feature_config: FeatureConfig,
    training_config: TrainingConfig,
    trading_cost_bp: float = 1.0,
    max_episode_steps: int = 2000,
) -> dict[str, float]:
    """Train a DQN agent using the provided configurations.
    
    Creates environments, trains with periodic validation, saves artifacts,
    and returns the best validation metrics (as produced by evaluate_model_on_env).
    """
    print("=" * 60)
    print("DQN Training Pipeline")
    print("=" * 60)
    
    # Prepare data
    train_df, val_df, scaler, feature_cols = _prepare_data(data_config, feature_config)
    
    # Create environments
    train_env, val_env = _create_envs(
        train_df,
        val_df,
        feature_cols,
        feature_config,
        training_config,
        trading_cost_bp,
        max_episode_steps,
    )
    
    # Create experiment folder
    print(f"\n[5] Creating experiment folder: {exp_name}")
    exp_path = create_experiment_folder("experiments", exp_name)
    print(f"    Path: {exp_path}")
    
    # Save configuration
    config_dict = {
        "exp_name": exp_name,
        "data_config": asdict(data_config),
        "feature_config": {
            "feat_cols": feature_cols,
            "window_size": feature_config.window_size,
            "use_log_returns": feature_config.use_log_returns,
        },
        "training_config": training_config.to_dict(),
        "trading_cost_bp": trading_cost_bp,
        "max_episode_steps": max_episode_steps,
    }
    save_config(config_dict, exp_path)
    print("    Saved config.json")
    
    # Initialize DQN model
    print("\n[6] Initializing DQN model...")
    dqn_kwargs = training_config.to_dqn_kwargs()
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        **dqn_kwargs,
    )
    print(f"    Policy: MlpPolicy")
    print(f"    Learning rate: {training_config.learning_rate}")
    print(f"    Buffer size: {training_config.buffer_size}")
    print(f"    Batch size: {training_config.batch_size}")
    initial_lr = training_config.learning_rate
    
    # Training loop with periodic validation
    print("\n[7] Starting training loop...")
    total_timesteps = training_config.total_timesteps
    num_epochs = training_config.num_epochs
    steps_per_epoch = total_timesteps // num_epochs
    patience = training_config.patience
    
    print(f"    Total timesteps: {total_timesteps}")
    print(f"    Epochs: {num_epochs}")
    print(f"    Steps per epoch: {steps_per_epoch}")
    print(f"    Validation episodes: {VAL_EPISODES}")
    print(f"    Early stopping patience: {patience}")
    
    best_val_metric = -np.inf
    best_epoch = -1
    best_metrics_payload: dict[str, float] | None = None
    best_metrics_raw: dict[str, float] | None = None
    last_val_metrics: dict[str, float] | None = None
    no_improve_epochs = 0
    training_log_path = exp_path / "training_log.csv"
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        decay_factor = 0.5 ** (epoch / LR_DECAY_EPOCHS)
        current_lr = initial_lr * decay_factor
        model.learning_rate = current_lr
        optimizers = []
        if hasattr(model, "optimizer"):
            optimizers.append(getattr(model, "optimizer"))
        policy_optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
        if policy_optimizer is not None:
            optimizers.append(policy_optimizer)
        seen_opt_ids: set[int] = set()
        for opt in optimizers:
            if opt is None or id(opt) in seen_opt_ids:
                continue
            seen_opt_ids.add(id(opt))
            for param_group in getattr(opt, "param_groups", []):
                param_group["lr"] = current_lr
        print(f"[train_dqn] Epoch {epoch + 1}/{num_epochs}, learning_rate={current_lr:.6g}")
        
        # Train for this epoch
        model.learn(
            total_timesteps=steps_per_epoch,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        
        timesteps_so_far = (epoch + 1) * steps_per_epoch
        
        # Evaluate on validation environment
        print(f"    Evaluating on validation set ({VAL_EPISODES} episodes)...")
        val_metrics, _ = evaluate_model_on_env(
            model=model,
            env=val_env,
            episodes=VAL_EPISODES,
            seed=training_config.seed,
            steps_per_year=STEPS_PER_YEAR,
        )
        last_val_metrics = val_metrics
        
        # Log metrics
        log_row = {
            "epoch": epoch + 1,
            "timesteps": timesteps_so_far,
            "val_final_equity": val_metrics["final_equity"],
            "val_total_return": val_metrics["total_return"],
            "val_max_drawdown": val_metrics["max_drawdown"],
            "val_sharpe": val_metrics["sharpe"],
            "val_n_trades": val_metrics["n_trades"],
            "val_win_rate": val_metrics["win_rate"],
            "val_profit_factor": val_metrics["profit_factor"],
        }
        append_metrics_row(training_log_path, log_row)
        
        print(
            f"    Val Return: {val_metrics['total_return']:+.2%}, "
            f"Sharpe: {val_metrics['sharpe']:.2f}, "
            f"MaxDD: {val_metrics['max_drawdown']:.2%}"
        )
        
        current_val_metric = float(val_metrics.get("final_equity", 0.0))
        
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_epoch = epoch + 1
            no_improve_epochs = 0
            
            # Save best model
            model.save(exp_path / "best_model")
            print(f"    * New best model! Val equity: {current_val_metric:.4f}")
            
            best_metrics_payload = {
                "epoch": best_epoch,
                "timesteps": timesteps_so_far,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            save_json(best_metrics_payload, exp_path / "best_val_metrics.json")
            best_metrics_raw = dict(val_metrics)
        else:
            no_improve_epochs += 1
            print(f"    No improvement for {no_improve_epochs} epoch(s)")
        
        if no_improve_epochs >= patience:
            print(
                f"\n[train_dqn] Early stopping triggered after {epoch + 1} epochs "
                f"with no validation improvement."
            )
            break
    
    # Save final model
    model.save(exp_path / "final_model")
    
    # Training summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Experiment folder: {exp_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val equity: {best_val_metric:.4f}")
    print("Files saved:")
    print("  - config.json")
    print("  - training_log.csv")
    print("  - best_model.zip")
    print("  - best_val_metrics.json")
    print("  - final_model.zip")
    
    return best_metrics_raw or last_val_metrics or {}


def train_dqn(exp_name: str = "exp_001_baseline_dqn") -> None:
    """Train DQN with default configurations."""
    data_config, feature_config, training_config = get_default_configs()
    train_dqn_with_configs(exp_name, data_config, feature_config, training_config)


def train_dqn_with_synthetic_data(
    exp_name: str = "exp_synthetic_test",
    n_bars: int = 5000,
    total_timesteps: int | None = None,
    num_epochs: int = 10,
) -> Path:
    """Train DQN with synthetic data for testing purposes.
    
    Useful when real SPY data is not available.
    
    Args:
        exp_name: Experiment name.
        n_bars: Number of synthetic bars to generate.
        total_timesteps: Total training timesteps. If None, uses TrainingConfig default.
        num_epochs: Number of epochs.
    
    Returns:
        Path to experiment folder.
    """
    print("=" * 60)
    print("DQN Training with Synthetic Data")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n[1] Generating synthetic data...")
    np.random.seed(42)
    
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
    
    # Split into train/val
    split_idx = int(n_bars * 0.7)
    train_df_raw = df.iloc[:split_idx].copy()
    val_df_raw = df.iloc[split_idx:].copy()
    print(f"    Train: {len(train_df_raw)} bars, Val: {len(val_df_raw)} bars")
    
    # Feature engineering
    print("\n[2] Applying feature engineering...")
    feature_config = FeatureConfig()
    train_df = add_basic_features(train_df_raw, use_log_returns=feature_config.use_log_returns)
    val_df = add_basic_features(val_df_raw, use_log_returns=feature_config.use_log_returns)
    
    feature_cols = get_feature_columns(use_log_returns=feature_config.use_log_returns)
    scaler = FeatureScaler()
    train_df = scaler.fit_transform(train_df, feature_cols)
    val_df = scaler.transform(val_df, feature_cols)
    print(f"    Train with features: {len(train_df)} rows")
    print(f"    Val with features: {len(val_df)} rows")
    
    # Create environments
    print("\n[3] Creating environments...")
    # Use default total_timesteps from TrainingConfig if not specified
    default_config = TrainingConfig()
    actual_timesteps = total_timesteps if total_timesteps is not None else default_config.total_timesteps
    
    training_config = TrainingConfig(
        seed=42,
        total_timesteps=actual_timesteps,
        num_epochs=num_epochs,
        learning_starts=1_000,
        buffer_size=50_000,  # Smaller buffer for synthetic data
    )
    
    train_env = make_env_from_df(
        df=train_df,
        feature_cols=feature_cols,
        window_size=feature_config.window_size,
        trading_cost_bp=1.0,
        max_episode_steps=500,
        deterministic_start=False,
        seed=training_config.seed,
    )
    
    val_env = make_env_from_df(
        df=val_df,
        feature_cols=feature_cols,
        window_size=feature_config.window_size,
        trading_cost_bp=1.0,
        max_episode_steps=500,
        deterministic_start=True,
        seed=training_config.seed,
    )
    
    # Create experiment folder
    print(f"\n[4] Creating experiment folder: {exp_name}")
    exp_path = create_experiment_folder("experiments", exp_name)
    
    config_dict = {
        "exp_name": exp_name,
        "synthetic_data": True,
        "n_bars": n_bars,
        "training_config": training_config.to_dict(),
    }
    save_config(config_dict, exp_path)
    
    # Initialize model
    print("\n[5] Initializing DQN model...")
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        **training_config.to_dqn_kwargs(),
    )
    initial_lr = training_config.learning_rate
    
    # Training loop with early stopping
    print("\n[6] Training...")
    total_timesteps = training_config.total_timesteps
    steps_per_epoch = total_timesteps // num_epochs
    best_val_metric = -np.inf
    no_improve_epochs = 0
    patience = training_config.patience
    training_log_path = exp_path / "training_log.csv"
    
    print(f"    Total timesteps: {total_timesteps}")
    print(f"    Steps per epoch: {steps_per_epoch}")
    print(f"    Validation episodes: {training_config.eval_episodes}")
    print(f"    Early stopping patience: {patience}")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        decay_factor = 0.5 ** (epoch / LR_DECAY_EPOCHS)
        current_lr = initial_lr * decay_factor
        model.learning_rate = current_lr
        optimizers = []
        if hasattr(model, "optimizer"):
            optimizers.append(getattr(model, "optimizer"))
        policy_optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
        if policy_optimizer is not None:
            optimizers.append(policy_optimizer)
        seen_opt_ids: set[int] = set()
        for opt in optimizers:
            if opt is None or id(opt) in seen_opt_ids:
                continue
            seen_opt_ids.add(id(opt))
            for param_group in getattr(opt, "param_groups", []):
                param_group["lr"] = current_lr
        print(f"[train_dqn_with_synthetic_data] Epoch {epoch + 1}/{num_epochs}, learning_rate={current_lr:.6g}")
        
        model.learn(
            total_timesteps=steps_per_epoch,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        
        val_metrics, _ = evaluate_model_on_env(
            model,
            val_env,
            episodes=training_config.eval_episodes,
            seed=training_config.seed,
        )
        
        log_row = {
            "epoch": epoch + 1,
            "timesteps": (epoch + 1) * steps_per_epoch,
            "val_final_equity": val_metrics["final_equity"],
            "val_total_return": val_metrics["total_return"],
            "val_sharpe": val_metrics["sharpe"],
        }
        append_metrics_row(training_log_path, log_row)
        
        print(f"    Val Return: {val_metrics['total_return']:+.2%}")
        
        current_val_metric = float(val_metrics.get("final_equity", 0.0))
        
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            no_improve_epochs = 0
            model.save(exp_path / "best_model")
            save_json(val_metrics, exp_path / "best_val_metrics.json")
            print(f"    * New best model!")
        else:
            no_improve_epochs += 1
            print(f"    No improvement for {no_improve_epochs} epoch(s)")
        
        # Early stopping check
        if no_improve_epochs >= patience:
            print(f"\n[train_dqn] Early stopping triggered after {epoch + 1} epochs "
                  f"with no validation improvement.")
            break
    
    model.save(exp_path / "final_model")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Files saved to: {exp_path}")
    print("=" * 60)
    
    return exp_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Trading Sandbox Training")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic data.")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="exp_001_baseline_dqn",
        help="Experiment name for single-run training.",
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=5000,
        help="Number of bars for synthetic data generation.",
    )
    args = parser.parse_args()
    
    if args.synthetic:
        print("Running with synthetic data...")
        train_dqn_with_synthetic_data(
            exp_name=args.exp_name,
            n_bars=args.n_bars,
            num_epochs=10,
        )
    else:
        train_dqn(exp_name=args.exp_name)

