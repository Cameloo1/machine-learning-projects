"""Synthetic SOC dataset generation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_ANOMALY_FRACTION,
    DEFAULT_EVENTS_PER_USER,
    DEFAULT_N_USERS,
    DEFAULT_RANDOM_STATE,
)
from .utils import make_rng


USER_COLUMNS: list[str] = [
    "user_id",
    "user_risk_score",
    "typical_login_hour",
    "typical_bytes_out",
    "typical_bytes_in",
    "home_lat",
    "home_lon",
]

EVENT_COLUMNS: list[str] = [
    "user_id",
    "hour",
    "login_success",
    "bytes_out",
    "bytes_in",
    "geo_distance",
    "failed_logins_24h",
    "device_trust_score",
    "user_risk_score",
    "is_anomaly",
]


def generate_users(n_users: int, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic SOC user profiles."""
    if n_users <= 0:
        raise ValueError("n_users must be positive.")

    rng = make_rng(random_state)
    user_ids = [f"user_{i}" for i in range(n_users)]
    high_risk_flags = rng.random(n_users) < 0.1

    risk_means = np.where(high_risk_flags, 0.7, 0.2)
    user_risk_scores = np.clip(rng.normal(risk_means, 0.05), 0.0, 1.0)

    login_hour_base = np.where(high_risk_flags, 11.0, 9.0)
    typical_login_hours = np.clip(rng.normal(login_hour_base, 1.5), 6.0, 20.0)

    typical_bytes_out = rng.lognormal(mean=10.0, sigma=0.6, size=n_users)
    typical_bytes_in = rng.lognormal(mean=10.5, sigma=0.5, size=n_users)

    home_lat = rng.uniform(-60.0, 60.0, size=n_users)
    home_lon = rng.uniform(-150.0, 150.0, size=n_users)

    data = {
        "user_id": user_ids,
        "user_risk_score": user_risk_scores,
        "typical_login_hour": typical_login_hours,
        "typical_bytes_out": typical_bytes_out,
        "typical_bytes_in": typical_bytes_in,
        "home_lat": home_lat,
        "home_lon": home_lon,
    }
    return pd.DataFrame(data, columns=USER_COLUMNS)


def generate_normal_events(
    users_df: pd.DataFrame,
    events_per_user: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Generate normal SOC events for each user."""
    if events_per_user <= 0:
        raise ValueError("events_per_user must be positive.")
    if users_df.empty:
        raise ValueError("users_df must contain at least one user.")

    rng = make_rng(random_state)
    records: list[dict[str, float | int | str]] = []

    for _, user in users_df.iterrows():
        for _ in range(events_per_user):
            hour = float(np.clip(rng.normal(user["typical_login_hour"], 1.5), 0.0, 23.0))
            login_success = int(rng.random() > 0.02)
            bytes_out = float(
                np.clip(
                    rng.normal(user["typical_bytes_out"], user["typical_bytes_out"] * 0.3),
                    0.0,
                    None,
                )
            )
            bytes_in = float(
                np.clip(
                    rng.normal(user["typical_bytes_in"], user["typical_bytes_in"] * 0.3),
                    0.0,
                    None,
                )
            )
            geo_distance = float(rng.exponential(scale=25.0))
            failed_logins_24h = int(np.clip(rng.poisson(0.5), 0, 5))
            device_trust_score = float(rng.uniform(0.7, 1.0))

            records.append(
                {
                    "user_id": user["user_id"],
                    "hour": hour,
                    "login_success": login_success,
                    "bytes_out": bytes_out,
                    "bytes_in": bytes_in,
                    "geo_distance": geo_distance,
                    "failed_logins_24h": failed_logins_24h,
                    "device_trust_score": device_trust_score,
                    "user_risk_score": float(user["user_risk_score"]),
                    "is_anomaly": 0,
                }
            )

    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def inject_anomalies(
    df: pd.DataFrame,
    anomaly_fraction: float,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Inject anomalous events of multiple types into a copy of df."""
    if not 0.0 <= anomaly_fraction <= 1.0:
        raise ValueError("anomaly_fraction must be between 0 and 1.")

    df_anom = df.copy(deep=True)
    rng = make_rng(random_state)
    n_anom = int(len(df_anom) * anomaly_fraction)
    if n_anom == 0:
        return df_anom

    candidate_indices = rng.choice(df_anom.index.to_numpy(), size=n_anom, replace=False)
    anomaly_groups = np.array_split(candidate_indices, 3)

    night_hours = np.array([0, 1, 2, 3, 4, 5, 22, 23])
    for idx in anomaly_groups[0]:
        df_anom.loc[idx, "hour"] = int(rng.choice(night_hours))
        df_anom.loc[idx, "bytes_out"] = float(df_anom.loc[idx, "bytes_out"] * rng.uniform(10.0, 30.0))
        df_anom.loc[idx, "device_trust_score"] = float(rng.uniform(0.1, 0.4))
        df_anom.loc[idx, "geo_distance"] = float(df_anom.loc[idx, "geo_distance"] + rng.uniform(100.0, 1000.0))

    for idx in anomaly_groups[1]:
        df_anom.loc[idx, "geo_distance"] = float(rng.uniform(3000.0, 10000.0))
        df_anom.loc[idx, "device_trust_score"] = float(rng.uniform(0.1, 0.5))

    for idx in anomaly_groups[2]:
        df_anom.loc[idx, "failed_logins_24h"] = int(rng.integers(20, 51))
        df_anom.loc[idx, "login_success"] = 0
        df_anom.loc[idx, "device_trust_score"] = float(rng.uniform(0.1, 0.5))

    df_anom.loc[candidate_indices, "is_anomaly"] = 1
    return df_anom


def generate_synthetic_soc_dataset(
    n_users: int = DEFAULT_N_USERS,
    events_per_user: int = DEFAULT_EVENTS_PER_USER,
    anomaly_fraction: float = DEFAULT_ANOMALY_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """End-to-end dataset generation orchestrator."""
    users_df = generate_users(n_users=n_users, random_state=random_state)
    normal_events_df = generate_normal_events(
        users_df=users_df,
        events_per_user=events_per_user,
        random_state=random_state,
    )
    return inject_anomalies(
        df=normal_events_df,
        anomaly_fraction=anomaly_fraction,
        random_state=random_state,
    )

