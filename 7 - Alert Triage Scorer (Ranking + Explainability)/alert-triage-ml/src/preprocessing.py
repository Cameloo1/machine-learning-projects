"""Preprocessing utilities for alert triage models."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = [
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

CATEGORICAL_COLS = [
    "alert_type",
    "kill_chain_stage",
]


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer with scaling and one-hot encoding."""

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )
    return preprocessor

