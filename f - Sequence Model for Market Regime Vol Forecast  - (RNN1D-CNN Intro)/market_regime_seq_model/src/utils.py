import os
import logging
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json

def setup_directories():
    """Ensure all project directories exist."""
    dirs = [
        "data/raw",
        "data/processed",
        "data/features",
        "data/regimes",
        "models/lstm",
        "models/cnn",
        "models/baseline_rf",
        "results/metrics",
        "results/plots",
        "results/explainability",
    ]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

def setup_logging(name: str = "market_regime") -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(name)

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_json(data: dict, path: str):
    """Save dictionary to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path: str) -> dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def save_plot(fig: plt.Figure, path: str):
    """Save matplotlib figure."""
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)

