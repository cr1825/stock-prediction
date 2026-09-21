"""Stacked LSTM model definition and training."""

from typing import Sequence, Tuple

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(lookback: int, n_features: int,
                     units: Tuple[int, int] = (64, 32), dropout: float = 0.2,
                     learning_rate: float = 1e-3) -> keras.Model:
    """Two stacked LSTM layers -> small dense head -> next-day (scaled) return."""
    model = keras.Sequential(
        [
            layers.Input(shape=(lookback, n_features)),
            layers.LSTM(units[0], return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(units[1]),
            layers.Dropout(dropout),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ],
        name="stacked_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(),  # robust to the fat tails of financial returns
        metrics=["mae"],
    )
    return model


def train_model(model: keras.Model, X_train, y_train, X_val, y_val,
                epochs: int = 50, batch_size: int = 32, verbose: int = 1,
                extra_callbacks: Sequence[keras.callbacks.Callback] = ()):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
        *extra_callbacks,
    ]
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
