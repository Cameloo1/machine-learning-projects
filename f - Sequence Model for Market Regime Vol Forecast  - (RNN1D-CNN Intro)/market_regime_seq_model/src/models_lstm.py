import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os
import logging

logger = logging.getLogger("market_regime.lstm")

def build_lstm_model(seq_len: int, num_features: int, num_classes: int) -> tf.keras.Model:
    """Build LSTM architecture."""
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    return model

def train_lstm(
    X_train, y_train,
    X_val, y_val,
    output_dir: str = "models/lstm",
    epochs: int = 30,
    batch_size: int = 32,
    num_classes: int = None
) -> tf.keras.Model:
    """Train LSTM model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine input shape
    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]
    
    if num_classes is None:
        # Fallback, but risky if y_train doesn't cover all classes
        num_classes = int(max(y_train) + 1)
    
    model = build_lstm_model(seq_len, num_features, num_classes)
    model.summary(print_fn=logger.info)
    
    callbacks = [
        EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2
    )
    
    # Save model
    model_path = os.path.join(output_dir, "model.h5")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save history
    history_path = os.path.join(output_dir, "history.json")
    # Convert values to float for JSON serialization
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(hist_dict, f, indent=4)
        
    return model

