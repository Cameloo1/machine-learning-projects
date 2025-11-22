"""Model training utilities."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def time_based_train_test_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and target into time-ordered train and test sets."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    n_rows = len(df)
    if n_rows < 2:
        raise ValueError("Need at least two rows to perform a split.")

    n_train = max(1, int(n_rows * train_fraction))
    if n_train >= n_rows:
        n_train = n_rows - 1

    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def compute_baseline_predictions(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Return majority-class baseline predictions for the test set."""

    if y_train.size == 0:
        raise ValueError("y_train cannot be empty.")

    majority_class = 1 if y_train.mean() >= 0.5 else 0
    return np.full(shape=y_test.shape, fill_value=majority_class, dtype=int)


def train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a scaled logistic regression model."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int
) -> RandomForestClassifier:
    """Train a random forest classifier."""

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int
) -> GradientBoostingClassifier:
    """Train a gradient boosting classifier."""

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

