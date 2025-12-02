"""Data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

from . import config
from .utils_io import load_dataframe, save_dataframe


def validate_alert_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, target, and null constraints."""

    expected_cols = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    non_numeric = [
        col
        for col in [
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
        ]
        if not is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise TypeError(f"Columns expected to be numeric but are not: {non_numeric}")

    if df[config.TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    if df[config.FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns contain missing values.")

    invalid_labels = set(df[config.TARGET_COLUMN].unique()).difference(config.LABEL_MAPPING.keys())
    if invalid_labels:
        raise ValueError(f"Priority column has invalid labels: {invalid_labels}")

    return df.copy()


def load_raw_alerts(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Load and validate the raw dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}. Generate it first.")
    df = load_dataframe(path)
    return validate_alert_df(df)


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COLUMN,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split."""

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    temp_size = val_size + test_size
    df_train, df_temp = train_test_split(
        df,
        train_size=train_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    relative_val_size = val_size / temp_size
    df_val, df_test = train_test_split(
        df_temp,
        train_size=relative_val_size,
        stratify=df_temp[target_col],
        random_state=random_state,
    )

    print(
        f"Split sizes: train={df_train.shape}, val={df_val.shape}, test={df_test.shape}"
    )
    return df_train, df_val, df_test


def save_processed_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Persist split dataframes to disk."""

    save_dataframe(train_df, config.PROCESSED_DIR / "train.csv")
    save_dataframe(val_df, config.PROCESSED_DIR / "val.csv")
    save_dataframe(test_df, config.PROCESSED_DIR / "test.csv")

