import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler
import os

logger = logging.getLogger("market_regime.seq")

def fit_scaler_on_train(train_df: pd.DataFrame, feature_cols: list[str], output_dir: str = "models") -> StandardScaler:
    """Fit scaler only on training data to prevent leakage."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    logger.info(f"Scaler fitted and saved to {output_dir}/scaler.pkl")
    
    return scaler

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "regime",
    seq_len: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM/CNN models.
    X[t] = features from t-seq_len to t-1
    y[t] = target at t
    """
    data = df[feature_cols].values
    targets = df[target_col].values
    timestamps = df.index.values
    
    X = []
    y = []
    ts = []
    
    # Loop from seq_len to end
    # If seq_len is 30, first target is at index 30
    # Input is index 0 to 29 (30 steps)
    
    for i in range(seq_len, len(df)):
        X.append(data[i-seq_len:i])
        y.append(targets[i])
        ts.append(timestamps[i])
        
    return np.array(X), np.array(y), np.array(ts)

def time_series_split(
    timestamps: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_end: str,
    val_end: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Split data based on timestamps.
    """
    # Ensure timestamps are numpy array of datetime64 or similar
    # If they are objects/strings, convert
    ts_pd = pd.to_datetime(timestamps)
    
    train_mask = ts_pd <= pd.Timestamp(train_end)
    val_mask = (ts_pd > pd.Timestamp(train_end)) & (ts_pd <= pd.Timestamp(val_end))
    test_mask = ts_pd > pd.Timestamp(val_end)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Save split indices/timestamps for reference could be useful, but we just return data
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Val samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "train_ts": timestamps[train_mask],
        "val_ts": timestamps[val_mask],
        "test_ts": timestamps[test_mask]
    }

def save_processed_data(splits: dict, output_dir: str = "data/processed"):
    """Save numpy arrays to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name in ["train", "val", "test"]:
        X, y = splits[split_name]
        np.save(os.path.join(output_dir, f"X_{split_name}.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y)
        
        if f"{split_name}_ts" in splits:
            np.save(os.path.join(output_dir, f"ts_{split_name}.npy"), splits[f"{split_name}_ts"])
            
    logger.info(f"Saved processed data to {output_dir}")

