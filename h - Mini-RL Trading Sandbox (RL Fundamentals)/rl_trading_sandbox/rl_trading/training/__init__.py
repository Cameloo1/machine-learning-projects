"""Training utilities for the RL Trading Sandbox."""

from rl_trading.training.eval import (
    evaluate_model_on_env,
    compute_buy_and_hold_equity,
    compute_buy_and_hold_metrics,
)

__all__ = [
    "evaluate_model_on_env",
    "compute_buy_and_hold_equity",
    "compute_buy_and_hold_metrics",
    "train_dqn",
    "run_test_evaluation",
]


def train_dqn(*args, **kwargs):
    """Lazy import wrapper to avoid eager side-effects when running as a script."""
    from rl_trading.training.train import train_dqn as _train_dqn

    return _train_dqn(*args, **kwargs)


def run_test_evaluation(*args, **kwargs):
    """Lazy import wrapper to avoid eager side-effects when running as a script."""
    from rl_trading.training.test_eval import run_test_evaluation as _run_test_eval

    return _run_test_eval(*args, **kwargs)

