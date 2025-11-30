"""CLI script to generate synthetic SOC datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from soc_anomaly.config import (
    DEFAULT_ANOMALY_FRACTION,
    DEFAULT_EVENTS_PER_USER,
    DEFAULT_N_USERS,
    DEFAULT_RANDOM_STATE,
)
from soc_anomaly.data_generation import generate_synthetic_soc_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate a synthetic SOC alert dataset.")
    parser.add_argument("--output-path", required=True, help="Path to save the generated CSV dataset.")
    parser.add_argument("--n-users", type=int, default=DEFAULT_N_USERS, help="Number of synthetic users.")
    parser.add_argument(
        "--events-per-user",
        type=int,
        default=DEFAULT_EVENTS_PER_USER,
        help="Number of events to generate per user.",
    )
    parser.add_argument(
        "--anomaly-fraction",
        type=float,
        default=DEFAULT_ANOMALY_FRACTION,
        help="Fraction of events to convert into anomalies.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for dataset generation."""
    args = parse_args()
    df = generate_synthetic_soc_dataset(
        n_users=args.n_users,
        events_per_user=args.events_per_user,
        anomaly_fraction=args.anomaly_fraction,
        random_state=args.random_state,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    total_events = len(df)
    anomaly_count = int(df["is_anomaly"].sum())
    anomaly_pct = anomaly_count / total_events if total_events else 0.0
    print(f"Saved dataset to {output_path}")
    print(f"Total events: {total_events:,}")
    print(f"Anomalies: {anomaly_count:,} ({anomaly_pct:.2%})")


if __name__ == "__main__":
    main()

