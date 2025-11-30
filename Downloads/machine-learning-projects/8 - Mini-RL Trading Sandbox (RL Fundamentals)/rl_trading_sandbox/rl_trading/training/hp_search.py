"""Small random hyperparameter search utilities for DQN training."""

from __future__ import annotations

import dataclasses
import random
from typing import Any, List, Tuple

import numpy as np

from rl_trading.config import TrainingConfig, get_default_configs
from rl_trading.training.train import train_dqn_with_configs

_rng = np.random.default_rng()


def sample_training_config(base: TrainingConfig) -> TrainingConfig:
    """Create a sampled TrainingConfig by perturbing key hyperparameters."""
    cfg = dataclasses.replace(base)
    cfg.learning_rate = random.choice([5e-5, 1e-4, 2e-4])
    cfg.gamma = random.choice([0.99, 0.995, 0.997])
    cfg.buffer_size = random.choice([200_000, 300_000])
    cfg.exploration_fraction = random.choice([0.1, 0.2, 0.3])
    cfg.batch_size = random.choice([32, 64])
    cfg.seed = int(_rng.integers(0, 1_000_000))
    return cfg


def run_random_search(n_trials: int = 10) -> None:
    """Run a lightweight random search over several TrainingConfig samples."""
    data_cfg, feat_cfg, base_train_cfg = get_default_configs()
    results: List[Tuple[str, dict[str, Any], TrainingConfig]] = []
    
    for i in range(n_trials):
        trial_cfg = sample_training_config(base_train_cfg)
        exp_name = f"exp_hp_{i:02d}"
        print(f"[hp_search] Trial {i + 1}/{n_trials}, exp={exp_name}, cfg={trial_cfg}")
        
        val_metrics: dict[str, Any] = train_dqn_with_configs(
            exp_name=exp_name,
            data_config=data_cfg,
            feature_config=feat_cfg,
            training_config=trial_cfg,
        )
        results.append((exp_name, val_metrics, trial_cfg))
    
    results.sort(key=lambda x: x[1].get("final_equity", 0.0), reverse=True)
    
    print("\n[hp_search] Top configs by val_final_equity:")
    for rank, (exp_name, metrics, cfg) in enumerate(results[:5], start=1):
        print(
            f"{rank}. {exp_name} - final_equity={metrics.get('final_equity', 0.0):.4f}, "
            f"sharpe={metrics.get('sharpe', 0.0):.2f}, cfg={cfg}"
        )


if __name__ == "__main__":
    run_random_search()

