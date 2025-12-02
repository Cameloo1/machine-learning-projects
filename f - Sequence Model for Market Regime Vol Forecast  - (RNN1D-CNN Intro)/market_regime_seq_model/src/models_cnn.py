import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os
import logging

logger = logging.getLogger("market_regime.cnn")

def build_cnn_model(seq_len: int, num_features: int, num_classes: int) -> tf.keras.Model:
    """Build 1D-CNN architecture."""
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        Conv1D(64, kernel_size=3, activation="relu"),
        Conv1D(32, kernel_size=3, activation="relu"),
        MaxPool1D(),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    return model

def train_cnn(
    X_train, y_train,
    X_val, y_val,
    output_dir: str = "models/cnn",
    epochs: int = 30,
    batch_size: int = 32,
    num_classes: int = None
) -> tf.keras.Model:
    """Train CNN model."""
    os.makedirs(output_dir, exist_ok=True)
    
    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]
    
    if num_classes is None:
        num_classes = int(max(y_train) + 1)
    
    model = build_cnn_model(seq_len, num_features, num_classes)
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
    
    model_path = os.path.join(output_dir, "model.h5")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    history_path = os.path.join(output_dir, "history.json")
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(hist_dict, f, indent=4)
        
    return model

