"""Short-Term Price Move Classifier package."""

from .config import DATA_START, DEFAULT_TICKER, RANDOM_STATE, TRAIN_FRACTION
from .data_loader import load_ohlcv
from .evaluation import (
    evaluate_classifier,
    plot_feature_importances,
    plot_logistic_coefficients,
    plot_roc_curve,
    print_evaluation_report,
)
from .features import build_feature_dataset
from .modeling import (
    compute_baseline_predictions,
    time_based_train_test_split,
    train_gradient_boosting,
    train_logistic_regression,
    train_random_forest,
)

__all__ = [
    "DATA_START",
    "DEFAULT_TICKER",
    "RANDOM_STATE",
    "TRAIN_FRACTION",
    "load_ohlcv",
    "build_feature_dataset",
    "time_based_train_test_split",
    "compute_baseline_predictions",
    "train_logistic_regression",
    "train_random_forest",
    "train_gradient_boosting",
    "evaluate_classifier",
    "print_evaluation_report",
    "plot_roc_curve",
    "plot_feature_importances",
    "plot_logistic_coefficients",
]

