import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging

logger = logging.getLogger("market_regime.rf")

def build_rf_features(X_seq: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Flatten sequences into static features for Random Forest.
    X_seq shape: (n_samples, seq_len, n_features)
    """
    n_samples, seq_len, n_features = X_seq.shape
    
    if len(feature_names) != n_features:
        raise ValueError(f"Feature names count ({len(feature_names)}) mismatch with X_seq features ({n_features})")
    
    rows = []
    
    for i in range(n_samples):
        seq = X_seq[i] # shape (seq_len, n_features)
        
        # Convert to DataFrame for easier aggregation
        df_seq = pd.DataFrame(seq, columns=feature_names)
        
        features = {}
        
        for col in feature_names:
            features[f"{col}_mean"] = df_seq[col].mean()
            features[f"{col}_std"] = df_seq[col].std()
            features[f"{col}_last"] = df_seq[col].iloc[-1]
            
            # Volatility specific features check
            # Prompt: "For volatility features: Min, Max"
            # We'll apply min/max to any feature with 'vol' in name
            if 'vol' in col:
                features[f"{col}_min"] = df_seq[col].min()
                features[f"{col}_max"] = df_seq[col].max()
                
        rows.append(features)
        
    return pd.DataFrame(rows)

def train_random_forest(
    X_train_rf: pd.DataFrame,
    y_train: np.ndarray,
    X_val_rf: pd.DataFrame,
    y_val: np.ndarray,
    output_dir: str = "models/baseline_rf"
) -> RandomForestClassifier:
    """Train Random Forest Classifier."""
    os.makedirs(output_dir, exist_ok=True)
    
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Training Random Forest...")
    rf.fit(X_train_rf, y_train)
    
    # Validation Score
    score = rf.score(X_val_rf, y_val)
    logger.info(f"RF Validation Accuracy: {score:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(rf, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return rf

