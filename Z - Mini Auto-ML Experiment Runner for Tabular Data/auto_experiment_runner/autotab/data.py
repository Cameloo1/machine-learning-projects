import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from autotab.config import ExperimentConfig

logger = logging.getLogger(__name__)

@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    numeric_features: List[str]
    categorical_features: List[str]

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series

@dataclass
class DatasetMetadata:
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int

def load_dataset(config: ExperimentConfig) -> Dataset:
    """
    Load dataset from CSV, drop ID columns, separate target, and infer feature types.
    """
    logger.info("Loading dataset from %s", config.dataset.path)
    
    try:
        df = pd.read_csv(config.dataset.path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {config.dataset.path}")
    
    # Drop ID columns
    if config.dataset.id_columns:
        df = df.drop(columns=config.dataset.id_columns, errors="ignore")
    
    # Validate target column
    if config.dataset.target_column not in df.columns:
        raise ValueError(f"Target column '{config.dataset.target_column}' not found in dataset.")
    
    y = df[config.dataset.target_column]
    X = df.drop(columns=[config.dataset.target_column])
    
    # Infer feature types
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    
    return Dataset(
        X=X,
        y=y,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

def train_valid_split(dataset: Dataset, config: ExperimentConfig) -> SplitData:
    """
    Split dataset into train and validation sets.
    """
    split_config = config.evaluation.split
    
    # Validate split type
    if split_config.type != "holdout":
        raise ValueError(f"Unsupported split type: {split_config.type}. Only 'holdout' is supported.")
    
    # Validate test_size
    if split_config.test_size <= 0 or split_config.test_size >= 1:
        raise ValueError(f"test_size must be between 0 and 1, got {split_config.test_size}")
    
    # Determine stratification
    stratify: Optional[pd.Series] = None
    if config.task.type == "classification" and split_config.stratify:
        stratify = dataset.y
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        dataset.X,
        dataset.y,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        stratify=stratify
    )
    
    return SplitData(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid
    )

def get_dataset_metadata(dataset: Dataset) -> DatasetMetadata:
    """
    Get basic metadata about the dataset.
    """
    return DatasetMetadata(
        n_rows=len(dataset.X),
        n_cols=len(dataset.X.columns),
        n_numeric=len(dataset.numeric_features),
        n_categorical=len(dataset.categorical_features)
    )
