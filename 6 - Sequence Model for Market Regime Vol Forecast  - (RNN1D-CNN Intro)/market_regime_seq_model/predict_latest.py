import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict")

def predict_next_day():
    # Base dir helper
    base_dir = os.path.dirname(os.path.abspath(__file__))
    def get_path(rel): return os.path.join(base_dir, rel)

    # 1. Load Dependencies
    logger.info("Loading models and data...")
    try:
        lstm_path = get_path("models/lstm/model.h5")
        scaler_path = get_path("models/scaler.pkl")
        features_path = get_path("data/features/features_with_regime.csv")
        
        logger.info(f"Model path: {lstm_path}")
        
        lstm = tf.keras.models.load_model(lstm_path)
        scaler = joblib.load(scaler_path)
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
    except Exception as e:
        logger.error(f"Could not load dependencies: {e}")
        return

    # 2. Get the most recent sequence
    SEQ_LEN = 30
    
    target_col = "regime"
    feature_cols = [c for c in df.columns if c != target_col]
    
    last_sequence_df = df.iloc[-SEQ_LEN:]
    
    if len(last_sequence_df) < SEQ_LEN:
        logger.error("Not enough data for a prediction.")
        return

    logger.info(f"Predicting for date after: {last_sequence_df.index[-1].date()}")

    # 3. Preprocess
    # Ensure feature columns match those used in scaler fit
    # (Scaler was fit on specific columns, we must match order and count)
    try:
        X_raw = last_sequence_df[feature_cols].values
        X_scaled = scaler.transform(X_raw)
    except ValueError as e:
        # Sometimes feature engineering adds cols or order changes
        logger.error(f"Feature mismatch: {e}")
        return
    
    X_input = X_scaled.reshape(1, SEQ_LEN, len(feature_cols))

    # 4. Predict
    y_probs = lstm.predict(X_input, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)[0]
    
    print("\n" + "="*40)
    print(f"  PREDICTION FOR NEXT TRADING DAY")
    print("="*40)
    print(f"Reference Date: {last_sequence_df.index[-1].date()}")
    print(f"Predicted Regime: {y_pred}")
    print(f"Confidence:       {y_probs[0][y_pred]*100:.2f}%")
    print("-" * 40)
    print(f"Full Probabilities: {np.round(y_probs[0], 3)}")
    print("="*40 + "\n")

if __name__ == "__main__":
    predict_next_day()
