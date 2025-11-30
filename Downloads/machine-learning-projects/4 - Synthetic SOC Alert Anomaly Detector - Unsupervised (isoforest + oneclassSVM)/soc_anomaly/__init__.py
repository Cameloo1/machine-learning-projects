"""Top-level package for the Synthetic SOC Alert Anomaly Detector."""

from .config import DEFAULT_RANDOM_STATE
from .data_generation import (
    generate_users,
    generate_normal_events,
    inject_anomalies,
    generate_synthetic_soc_dataset,
)
from .anomaly_detection import FEATURE_COLS, LABEL_COL

__all__ = [
    "DEFAULT_RANDOM_STATE",
    "FEATURE_COLS",
    "LABEL_COL",
    "generate_users",
    "generate_normal_events",
    "inject_anomalies",
    "generate_synthetic_soc_dataset",
]

