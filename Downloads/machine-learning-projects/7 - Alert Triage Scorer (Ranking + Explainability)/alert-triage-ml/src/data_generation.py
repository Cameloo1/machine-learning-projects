"""Synthetic SOC alert data generation module."""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd

from . import config
from .utils_io import save_dataframe
from .utils_plotting import plot_class_distribution


ALERT_TYPES: List[str] = [
    "failed_login_burst",
    "privilege_escalation",
    "malware_detected",
    "data_exfil",
    "policy_violation",
    "suspicious_process",
]

KILL_CHAIN_STAGES: List[str] = [
    "recon",
    "initial_access",
    "execution",
    "lateral_movement",
    "exfiltration",
]


def assign_priority(row: pd.Series) -> int:
    """Assign a discrete priority label according to the specified rules."""

    score = 0.0
    score += 1.5 * max(row["src_asset_criticality"], row["dst_asset_criticality"])

    if row["user_risk_score"] > 80:
        score += 3.0
    elif row["user_risk_score"] > 50:
        score += 1.5

    score += row["rule_severity"]
    score += 3.0 * row["detection_confidence"]

    if row["rule_historical_fpr"] < 0.1:
        score += 3.0
    elif row["rule_historical_fpr"] > 0.4:
        score -= 2.0

    if row["is_known_fp_source"] == 1:
        score -= 3.0

    if row["geo_distance_km"] > 2000 and row["hour_of_day"] in {0, 1, 2, 3, 4}:
        score += 2.0

    if row["alert_type"] in {"data_exfil", "privilege_escalation", "malware_detected"}:
        score += 2.0

    if row["kill_chain_stage"] in {"execution", "exfiltration"}:
        score += 2.0

    if score >= 12.0:
        return 2
    if score >= 7.0:
        return 1
    return 0


def generate_synthetic_alerts(n_samples: int = 5000, random_state: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic SOC alert dataset."""

    config.ensure_directories()
    config.seed_everything(random_state)
    rng = np.random.default_rng(random_state)

    alert_type = rng.choice(
        ALERT_TYPES,
        size=n_samples,
        p=[0.25, 0.15, 0.2, 0.1, 0.15, 0.15],
    )

    criticality_base = rng.choice([1, 2, 3, 4, 5], size=(n_samples, 2), p=[0.1, 0.3, 0.3, 0.2, 0.1])
    src_asset_criticality = criticality_base[:, 0]
    dst_asset_criticality = criticality_base[:, 1]

    user_risk_score = rng.beta(1.5, 5.0, size=n_samples) * 100

    event_count_24h = rng.poisson(lam=3.0, size=n_samples)
    noise_mask = rng.random(n_samples) < 0.08
    event_count_24h = event_count_24h + noise_mask * rng.integers(15, 60, size=n_samples)

    failed_login_ratio = np.clip(rng.beta(0.5, 5.0, size=n_samples) + noise_mask * rng.uniform(0.2, 0.8, size=n_samples), 0, 1)

    geo_distance_km = np.where(
        rng.random(n_samples) < 0.7,
        rng.gamma(shape=1.2, scale=50, size=n_samples),
        rng.lognormal(mean=7.0, sigma=0.5, size=n_samples),
    )

    rule_severity = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.35, 0.35, 0.2, 0.05])

    rule_historical_fpr = np.where(
        rng.random(n_samples) < 0.7,
        rng.uniform(0.2, 0.6, size=n_samples),
        rng.uniform(0.01, 0.15, size=n_samples),
    )

    detection_confidence = rng.beta(2.0, 2.0, size=n_samples)

    is_known_fp_source = rng.binomial(n=1, p=0.25, size=n_samples)

    hour_of_day = rng.integers(0, 24, size=n_samples)

    kill_chain_stage = rng.choice(
        KILL_CHAIN_STAGES,
        size=n_samples,
        p=[0.25, 0.25, 0.2, 0.15, 0.15],
    )

    df = pd.DataFrame(
        {
            "alert_type": alert_type,
            "src_asset_criticality": src_asset_criticality,
            "dst_asset_criticality": dst_asset_criticality,
            "user_risk_score": user_risk_score,
            "event_count_24h": event_count_24h,
            "failed_login_ratio": failed_login_ratio,
            "geo_distance_km": geo_distance_km,
            "rule_severity": rule_severity,
            "rule_historical_fpr": rule_historical_fpr,
            "detection_confidence": detection_confidence,
            "is_known_fp_source": is_known_fp_source,
            "hour_of_day": hour_of_day,
            "kill_chain_stage": kill_chain_stage,
        }
    )

    df["priority"] = df.apply(assign_priority, axis=1)

    noise_rate = 0.04
    flip_mask = rng.random(n_samples) < noise_rate
    df.loc[flip_mask, "priority"] = rng.integers(0, 3, size=flip_mask.sum())

    save_dataframe(df, config.RAW_DATA_PATH)
    plot_class_distribution(df, config.TARGET_COLUMN, config.PLOTS_DIR / "class_distribution.png")
    print(f"Saved synthetic dataset with shape {df.shape} to {config.RAW_DATA_PATH}")
    return df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Generate synthetic SOC alert data.")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to generate.")
    parser.add_argument("--random_state", type=int, default=config.RANDOM_STATE, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    generate_synthetic_alerts(n_samples=args.n_samples, random_state=args.random_state)


if __name__ == "__main__":
    main()

