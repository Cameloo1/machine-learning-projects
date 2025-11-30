"""Utility helpers for IO operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as JSON with pretty formatting."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)


def load_json(path: Path) -> Dict[str, Any]:
    """Load and return JSON content from disk."""

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe to CSV (index-free)."""

    ensure_parent_dir(path)
    df.to_csv(path, index=False)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from CSV."""

    return pd.read_csv(path)
import os
import logging
import pandas as pd

def setup_logging(log_level=logging.INFO):
    """Configures logging for the project."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def ensure_directory(path: str):
    """Ensures that a directory exists; creates it if not."""
    os.makedirs(path, exist_ok=True)

def save_dataframe(df: pd.DataFrame, filepath: str):
    """Saves a DataFrame to CSV with directory creation and logging."""
    logger = logging.getLogger(__name__)
    ensure_directory(os.path.dirname(filepath))
    try:
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved DataFrame to {filepath} with shape {df.shape}")
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {filepath}: {e}")
        raise

