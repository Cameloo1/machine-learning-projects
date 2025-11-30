import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug")

def debug_models():
    # Base dir is the directory containing this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Helper to get full path
    def get_path(rel_path):
        # rel_path comes in as "market_regime_seq_model/data/..."
        # If we are in "market_regime_seq_model" folder, we might need to adjust
        # But simpler: construct path relative to this script file
        # This script is in market_regime_seq_model/
        # So data is in ./data/
        return os.path.join(base_dir, rel_path.replace("market_regime_seq_model/", ""))

    logger.info(f"Base dir: {base_dir}")

    # Load data
    try:
        X_test = np.load(get_path("data/processed/X_test.npy"))
        y_test = np.load(get_path("data/processed/y_test.npy"))
    except FileNotFoundError:
        logger.error(f"Processed data not found at {get_path('data/processed/X_test.npy')}")
        return

    # Load Models
    try:
        lstm_path = get_path("models/lstm/model.h5")
        cnn_path = get_path("models/cnn/model.h5")
        lstm = tf.keras.models.load_model(lstm_path)
        cnn = tf.keras.models.load_model(cnn_path)
    except Exception as e:
        logger.error(f"Could not load models: {e}")
        return

    # Evaluate LSTM
    logger.info("Evaluating LSTM...")
    y_probs_lstm = lstm.predict(X_test, verbose=0)
    y_pred_lstm = np.argmax(y_probs_lstm, axis=1)
    # Use zero_division=0 to suppress warning but we want to see it
    f1_lstm = f1_score(y_test, y_pred_lstm, average='macro')
    
    logger.info(f"LSTM Baseline F1: {f1_lstm}")
    logger.info(f"LSTM Unique Preds: {np.unique(y_pred_lstm)}")
    logger.info(f"LSTM True Labels:   {np.unique(y_test)}")
    logger.info(f"LSTM Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lstm)}")
    
    # Evaluate CNN
    logger.info("Evaluating CNN...")
    y_probs_cnn = cnn.predict(X_test, verbose=0)
    y_pred_cnn = np.argmax(y_probs_cnn, axis=1)
    f1_cnn = f1_score(y_test, y_pred_cnn, average='macro')
    
    logger.info(f"CNN Baseline F1: {f1_cnn}")
    logger.info(f"CNN Unique Preds: {np.unique(y_pred_cnn)}")
    logger.info(f"CNN Confusion Matrix:\n{confusion_matrix(y_test, y_pred_cnn)}")

if __name__ == "__main__":
    debug_models()
