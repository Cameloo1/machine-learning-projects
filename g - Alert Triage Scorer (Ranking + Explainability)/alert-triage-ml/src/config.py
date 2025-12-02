"""Project-wide configuration helpers and constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import random


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "alerts_synthetic.csv"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
METRICS_DIR: Path = ARTIFACTS_DIR / "metrics"
PLOTS_DIR: Path = ARTIFACTS_DIR / "plots"
SHAP_DIR: Path = ARTIFACTS_DIR / "shap"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
EXPLANATIONS_PATH: Path = ARTIFACTS_DIR / "explanations_sample.csv"

RANDOM_STATE: int = 42
CV_FOLDS: int = 4
N_JOBS: int = -1

FEATURE_COLUMNS = [
    "alert_type",
    "src_asset_criticality",
    "dst_asset_criticality",
    "user_risk_score",
    "event_count_24h",
    "failed_login_ratio",
    "geo_distance_km",
    "rule_severity",
    "rule_historical_fpr",
    "detection_confidence",
    "is_known_fp_source",
    "hour_of_day",
    "kill_chain_stage",
]

TARGET_COLUMN = "priority"
LABEL_MAPPING = {0: "Low", 1: "Medium", 2: "High"}


@dataclass(frozen=True)
class TrainSplitConfig:
    """Configuration for dataset splits."""

    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = RANDOM_STATE


def ensure_directories() -> None:
    """Ensure that output directories exist."""

    for path in [
        DATA_DIR,
        RAW_DATA_PATH.parent,
        PROCESSED_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        SHAP_DIR,
        REPORTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed common libraries for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)

