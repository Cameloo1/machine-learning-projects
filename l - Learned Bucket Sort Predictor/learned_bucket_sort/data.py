"""Synthetic data generation primitives."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray


DistributionName = str

SUPPORTED_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "uniform",
    "gaussian",
    "lognormal",
    "bimodal",
    "clustered",
)


def generate_data(distribution: DistributionName, n: int, seed: int) -> NDArray[np.float64]:
    """Generate a deterministic 1D numeric dataset."""
    if n < 0:
        raise ValueError("n must be non-negative")

    rng = np.random.default_rng(seed)
    distribution_key = distribution.lower()

    if distribution_key == "uniform":
        values = rng.uniform(0.0, 1.0, size=n)
    elif distribution_key == "gaussian":
        values = rng.normal(loc=0.0, scale=1.0, size=n)
    elif distribution_key == "lognormal":
        values = rng.lognormal(mean=0.0, sigma=1.0, size=n)
    elif distribution_key == "bimodal":
        left_count = n // 2
        right_count = n - left_count
        left = rng.normal(loc=-2.0, scale=0.45, size=left_count)
        right = rng.normal(loc=2.0, scale=0.45, size=right_count)
        values = np.concatenate([left, right])
        rng.shuffle(values)
    elif distribution_key == "clustered":
        centers = np.array([-3.0, -0.5, 0.75, 4.0], dtype=np.float64)
        probabilities = np.array([0.55, 0.25, 0.15, 0.05], dtype=np.float64)
        assignments = rng.choice(len(centers), size=n, p=probabilities)
        values = centers[assignments] + rng.normal(loc=0.0, scale=0.08, size=n)
    else:
        supported = ", ".join(SUPPORTED_DISTRIBUTIONS)
        raise ValueError(f"unsupported distribution '{distribution}'; expected one of: {supported}")

    return np.asarray(values, dtype=np.float64)
