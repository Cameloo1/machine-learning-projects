from pathlib import Path
from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field
import yaml

class DatasetConfig(BaseModel):
    path: str = Field(..., min_length=1)
    target_column: str
    id_columns: List[str] = Field(default_factory=list)

class TaskConfig(BaseModel):
    type: Literal["classification", "regression"]
    problem_name: str

class PreprocessingConfig(BaseModel):
    impute_strategy_numeric: str = "median"
    impute_strategy_categorical: str = "most_frequent"
    scale_numeric: bool = True
    one_hot_encode_categorical: bool = True
    drop_low_variance_threshold: float = 0.0

class ModelParams(BaseModel):
    name: str
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)

class SplitConfig(BaseModel):
    type: Literal["holdout"]
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

class EvaluationConfig(BaseModel):
    primary_metric: str
    additional_metrics: List[str] = Field(default_factory=list)
    split: SplitConfig

class OutputConfig(BaseModel):
    base_dir: str = "outputs"
    save_models: bool = True
    save_feature_importance: bool = True
    save_plots: bool = True
    save_classification_report: bool = True
    save_confusion_matrix: bool = True
    save_residual_plot: bool = True

class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    task: TaskConfig
    preprocessing: PreprocessingConfig
    models: List[ModelParams]
    evaluation: EvaluationConfig
    output: OutputConfig

def load_config(path: str) -> ExperimentConfig:
    """
    Read YAML, validate, and return a strongly-typed ExperimentConfig.
    """
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    config = ExperimentConfig(**raw_config)
    # Store the source config path so downstream code can copy it into outputs
    setattr(config, "_source_config_path", str(Path(path).resolve()))
    return config
