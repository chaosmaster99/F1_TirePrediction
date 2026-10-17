"""
Deep Learning Model: LSTM (Long Short-Term Memory)
Sequential model that captures temporal dependencies in tire degradation.

Input: Sliding-window sequences of per-lap features
       [TyreLife, CompoundEncoded, FuelEffect, DegradationRate,
        TyreDegradation, TyreLifeNorm, RaceProgress, AvgSpeed]
Output: Predicted degradation (seconds above baseline) at next time step
"""

import numpy as np
import pandas as pd
import os
import warnings
import logging
import joblib

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Must match preprocessing.py
COMPOUND_MAP = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
COMPOUND_EXPECTED_LIFE = {"SOFT": 18, "MEDIUM": 28, "HARD": 40}

# Maximum realistic degradation cap (seconds above baseline)
MAX_DEGRADATION = 5.0


def build_lstm_model(n_features: int, seq_length: int):
    """Build simplified LSTM model architecture — single layer, less
    regularisation for better generalisation on small datasets."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, BatchNormalization,
    )
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(32, return_sequences=False,
             recurrent_dropout=0.1,
             input_shape=(seq_length, n_features)),
        Dropout(0.15),
        BatchNormalization(),
        Dense(16, activation="relu"),
        Dropout(0.1),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss="huber",
        metrics=["mae"],
    )

    return model


class LSTMDegradationModel:
    """
    LSTM-based tire degradation model.

    Processes sequences of lap data to predict future degradation.
    Captures non-linear temporal patterns that simpler time series
    models may miss.
    """

    def __init__(self, seq_length: int = 8, epochs: int = 120, batch_size: int = 32):
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.metrics = {}
        self.scaler = None
        self.feature_cols = None
        self.compound_profiles = {}  # per-compound median feature values
        self.name = "LSTM"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, lstm_data: dict):
        """Train LSTM model on prepared sequence data."""
        if lstm_data is None:
            logger.error("No LSTM data provided!")
            return

        logger.info("Fitting LSTM model...")

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        # Fixed seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)

        X_train = lstm_data["X_train"]
        y_train = lstm_data["y_train"]
        X_test = lstm_data["X_test"]
        y_test = lstm_data["y_test"]
        self.scaler = lstm_data["scaler"]
        self.feature_cols = lstm_data["feature_cols"]
        self.seq_length = lstm_data["seq_length"]

        n_features = X_train.shape[2]

        # Store per-compound feature profiles for predict_degradation()
        self._build_compound_profiles(X_train)

        # Build model
        self.model = build_lstm_model(n_features, self.seq_length)

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6),
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        logger.info(
            f"LSTM training complete. Final val_loss: "
            f"{self.history.history['val_loss'][-1]:.4f}"
        )

    def _build_compound_profiles(self, X_train_scaled: np.ndarray):
        """
        Extract per-compound median feature values from training data
        (inverse-transformed).  Used to build synthetic initial sequences
        for predict_degradation().
        """
        n_samples, n_steps, n_feats = X_train_scaled.shape
        flat = X_train_scaled.reshape(-1, n_feats)
        raw = self.scaler.inverse_transform(flat).reshape(n_samples, n_steps, n_feats)

        comp_idx = self._fidx("CompoundEncoded")
        for cname, cval in COMPOUND_MAP.items():
            mask = np.abs(raw[:, -1, comp_idx] - cval) < 0.5
            if mask.sum() == 0:
                continue
            self.compound_profiles[cname] = np.median(raw[mask, -1, :], axis=0)

        logger.info(f"Compound profiles built: {sorted(self.compound_profiles.keys())}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict degradation from pre-scaled sequences."""
        return self.model.predict(X, verbose=0).flatten()

    def predict_degradation(self, compound: str, current_tyre_life: int,
                            forecast_laps: int = 15) -> np.ndarray:
        """
        Predict degradation curve for *compound* starting at
        *current_tyre_life* for *forecast_laps* into the future.

        Returns np.ndarray of shape (forecast_laps,) — degradation in
        seconds above baseline, capped at MAX_DEGRADATION.
        """
        compound = compound.upper()
        if compound not in COMPOUND_MAP:
            return np.zeros(forecast_laps)

        profile = self.compound_profiles.get(compound)
        if profile is None:
            return np.zeros(forecast_laps)

        n_feats = len(profile)
        expected_life = COMPOUND_EXPECTED_LIFE.get(compound, 30)

        # Build an initial window of seq_length steps ending at
        # current_tyre_life, using the compound's median feature profile
        # for context-independent features (FuelEffect, AvgSpeed, etc.)
        seq_raw = np.zeros((self.seq_length, n_feats))
        for s in range(self.seq_length):
            tl = max(1, current_tyre_life - (self.seq_length - 1 - s))
            step = profile.copy()
            step[self._fidx("TyreLife")] = tl
            step[self._fidx("CompoundEncoded")] = COMPOUND_MAP[compound]
            step[self._fidx("StintLap")] = tl
            step[self._fidx("TyreLifeNorm")] = tl / expected_life
            seq_raw[s] = step

        # Scale the whole window
        current = self.scaler.transform(seq_raw)

        # Autoregressive forecasting
        predictions = []
        for _ in range(forecast_laps):
            pred = float(
                self.model.predict(
                    current.reshape(1, self.seq_length, n_feats), verbose=0
                )[0, 0]
            )
            pred = np.clip(pred, 0.0, MAX_DEGRADATION)
            predictions.append(pred)

            # Build next timestep in raw space, then re-scale
            new_raw = self.scaler.inverse_transform(
                current[-1:].reshape(1, -1)
            )[0].copy()
            new_raw[self._fidx("TyreLife")] += 1
            new_raw[self._fidx("StintLap")] = new_raw[self._fidx("TyreLife")]
            new_raw[self._fidx("TyreLifeNorm")] = (
                new_raw[self._fidx("TyreLife")] / expected_life
            )
            new_scaled = self.scaler.transform(new_raw.reshape(1, -1))[0]
            current = np.vstack([current[1:], new_scaled])

        return np.array(predictions)

    def _fidx(self, name: str) -> int:
        """Feature column index by name."""
        if self.feature_cols and name in self.feature_cols:
            return self.feature_cols.index(name)
        # Fallback matching preprocessing.py default order
        _fallback = {
            "TyreLife": 0, "CompoundEncoded": 1, "StintLap": 2,
            "FuelEffect": 3, "TyreLifeNorm": 4, "RaceProgress": 5,
            "AvgSpeed": 6, "CircuitEncoded": 7, "DriverEncoded": 8,
        }
        return _fallback.get(name, 0)

    def predict_degradation_sequence(self, initial_sequence: np.ndarray,
                                     forecast_laps: int = 15) -> np.ndarray:
        """Autoregressive prediction from a pre-scaled sequence."""
        predictions = []
        current_seq = initial_sequence.copy()

        tl_idx = self._fidx("TyreLife")
        sl_idx = self._fidx("StintLap")

        for _ in range(forecast_laps):
            pred = float(
                self.model.predict(
                    current_seq.reshape(1, *current_seq.shape), verbose=0
                )[0, 0]
            )
            pred = np.clip(pred, 0.0, MAX_DEGRADATION)
            predictions.append(pred)

            new_step = current_seq[-1].copy()
            new_step[tl_idx] += 1
            new_step[sl_idx] = new_step[tl_idx]

            current_seq = np.vstack([current_seq[1:], new_step])

        return np.array(predictions)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, lstm_data: dict = None, **_) -> dict:
        """Evaluate LSTM model on test sequences."""
        if lstm_data is None:
            return {"rmse": float("inf"), "mae": float("inf"), "r2": -1.0}

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        preds = self.predict(lstm_data["X_test"])
        y_test = lstm_data["y_test"]

        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "mae": float(mean_absolute_error(y_test, preds)),
            "r2": float(r2_score(y_test, preds)) if len(y_test) > 1 else 0.0,
        }

        logger.info(
            f"LSTM Evaluation: RMSE={self.metrics['rmse']:.4f}, "
            f"MAE={self.metrics['mae']:.4f}, R2={self.metrics['r2']:.4f}"
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Strategy helper
    # ------------------------------------------------------------------

    def get_pit_recommendation(self, compound: str, current_tyre_life: int,
                               threshold: float = 1.5,
                               max_forecast: int = 30) -> int:
        """
        Recommend laps until pit stop.
        Returns -1 if degradation stays below *threshold* within window.
        """
        forecast = self.predict_degradation(compound, current_tyre_life, max_forecast)
        for i, deg in enumerate(forecast):
            if deg > threshold:
                return i + 1
        return -1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save all model artefacts."""
        os.makedirs(path, exist_ok=True)
        if self.model:
            self.model.save(os.path.join(path, "lstm_model.keras"))
        joblib.dump({
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "seq_length": self.seq_length,
            "metrics": self.metrics,
            "compound_profiles": self.compound_profiles,
        }, os.path.join(path, "lstm_config.pkl"))
        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str):
        """Load all model artefacts."""
        import tensorflow as tf

        model_path = os.path.join(path, "lstm_model.keras")
        config_path = os.path.join(path, "lstm_config.pkl")

        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)

        if os.path.exists(config_path):
            cfg = joblib.load(config_path)
            self.scaler = cfg["scaler"]
            self.feature_cols = cfg.get("feature_cols")
            self.seq_length = cfg.get("seq_length", 5)
            self.metrics = cfg.get("metrics", {})
            self.compound_profiles = cfg.get("compound_profiles", {})
        else:
            # Legacy fallback — old format saved scaler separately
            scaler_path = os.path.join(path, "lstm_scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

        logger.info(f"LSTM model loaded from {path}")
