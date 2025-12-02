"""Model loading utilities with caching."""
import json
import joblib
from functools import lru_cache
from typing import Any, Dict

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> Any:
    """
    Lazy-load and cache the trained model from models/model.joblib.
    
    Returns:
        The loaded scikit-learn pipeline model.
    """
    model_path = settings.MODEL_PATH
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please train the model first using the training notebook."
        )
    
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def get_model_meta() -> Dict[str, Any]:
    """
    Load and cache model metadata from models/model_meta.json.
    
    Returns:
        Dictionary containing model metadata (features, classes, etc.).
    """
    meta_path = settings.MODEL_META_PATH
    
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata file not found at {meta_path}. "
            "Please train the model first using the training notebook."
        )
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Ensure all values are standard Python types (no numpy)
    return _convert_to_native_types(meta)


@lru_cache(maxsize=1)
def get_metrics() -> Dict[str, Any]:
    """
    Load and cache training metrics from metrics/metrics.json.
    
    Returns:
        Dictionary containing training metrics (classification_report, confusion_matrix).
    """
    metrics_path = settings.METRICS_PATH
    
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {metrics_path}. "
            "Please train the model first using the training notebook."
        )
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    # Ensure all values are standard Python types (no numpy)
    return _convert_to_native_types(metrics)


def _convert_to_native_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types.
    
    Args:
        obj: Object that may contain numpy types.
        
    Returns:
        Object with all numpy types converted to native Python types.
    """
    import numpy as np
    
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_native_types(item) for item in obj]
    else:
        return obj
