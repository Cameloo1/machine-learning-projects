"""Logging utilities for experiment tracking and metric storage.

This module provides utilities for creating experiment folders, saving configs,
and appending metrics to CSV files for structured experiment logging.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def create_experiment_folder(base_dir: str, exp_name: str) -> Path:
    """Create an experiment folder under the base directory.
    
    Creates directory structure: {base_dir}/{exp_name}/
    
    Args:
        base_dir: Base directory for experiments (e.g., "experiments").
        exp_name: Name of this experiment (e.g., "exp_001_baseline_dqn").
    
    Returns:
        Path to the created experiment folder.
    
    Example:
        >>> exp_path = create_experiment_folder("experiments", "exp_001")
        >>> print(exp_path)
        experiments/exp_001
    """
    exp_path = Path(base_dir) / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    return exp_path


def save_config(config: dict[str, Any], path: Path) -> None:
    """Save configuration dictionary as JSON.
    
    Args:
        config: Configuration dictionary to save.
        path: Directory path where config.json will be saved.
    
    Example:
        >>> save_config({"seed": 42, "lr": 0.001}, Path("experiments/exp_001"))
    """
    config_path = path / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


def append_metrics_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append a row of metrics to a CSV file.
    
    Creates the CSV file with headers if it doesn't exist.
    Appends a new row with the given metrics.
    
    Args:
        csv_path: Full path to the CSV file.
        row: Dictionary where keys are column names and values are metric values.
    
    Example:
        >>> append_metrics_row(
        ...     Path("experiments/exp_001/training_log.csv"),
        ...     {"epoch": 1, "loss": 0.5, "reward": 100.0}
        ... )
    """
    file_exists = csv_path.exists()
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)


def save_json(obj: dict[str, Any], path: Path) -> None:
    """Save a dictionary as a JSON file.
    
    Args:
        obj: Dictionary to save.
        path: Full path to the JSON file (including filename).
    
    Example:
        >>> save_json({"metric": 0.95}, Path("experiments/exp_001/best_metrics.json"))
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file as a dictionary.
    
    Args:
        path: Full path to the JSON file.
    
    Returns:
        Loaded dictionary.
    
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

