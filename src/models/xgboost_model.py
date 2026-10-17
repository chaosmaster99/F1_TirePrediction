"""
ML Model: XGBoost Gradient Boosting
For both regression (lap time degradation) and classification (pit decision).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
)
import joblib
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Maximum degradation cap (seconds) — same as ARIMA/ETS/LSTM
MAX_DEGRADATION = 5.0

# Reverse map for CompoundEncoded produced by preprocessing (COMPOUND_MAP)
_COMPOUND_REVERSE = {1: "SOFT", 2: "MEDIUM", 3: "HARD"}


class XGBoostPitModel:
    """
    XGBoost model for pit stop prediction.

    Two sub-models:
    1. Regression: Predicts tire degradation (seconds lost)
    2. Classification: Predicts whether to pit (binary)

    Also stores per-compound degradation profiles learned from
    training data so it can produce degradation curves for plotting
    and pit-stop recommendations without external preprocessing
    artifacts.
    """

    def __init__(self):
        self.reg_model = None
        self.cls_model = None
        self.metrics = {}
        self.feature_importance = {}
        self.compound_profiles = {}
        self._pit_threshold = 0.38
        self.name = "XGBoost"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, ml_data: dict):
        """Fit both regression and classification XGBoost models."""
        logger.info("Fitting XGBoost models...")

        X_train = ml_data["X_train"]
        y_reg_train = ml_data["y_reg_train"]
        y_cls_train = ml_data["y_cls_train"]
        feature_cols = ml_data["feature_cols"]

        # Regression model for degradation prediction
        self.reg_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self.reg_model.fit(
            X_train, y_reg_train,
            eval_set=[(ml_data["X_test"], ml_data["y_reg_test"])],
            verbose=False,
        )

        # Classification model for pit decision (handle class imbalance)
        n_pos = sum(y_cls_train == 1)
        n_neg = sum(y_cls_train == 0)
        scale_pos_weight = n_neg / max(n_pos, 1)

        self.cls_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            scale_pos_weight=scale_pos_weight,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self.cls_model.fit(
            X_train, y_cls_train,
            eval_set=[(ml_data["X_test"], ml_data["y_cls_test"])],
            verbose=False,
        )
        self._pit_threshold = 0.52  # balanced threshold for precision/recall

        # Feature importance
        if feature_cols:
            self.feature_importance = dict(zip(
                feature_cols,
                self.reg_model.feature_importances_,
            ))

        # Build compound degradation profiles from training data
        self._build_compound_profiles(ml_data)

        logger.info("XGBoost models fitted successfully")

    def _build_compound_profiles(self, ml_data: dict):
        """
        Extract per-compound degradation curves from the unscaled
        feature DataFrame so that ``predict_degradation(compound, …)``
        can produce a curve without external preprocessing artifacts.

        TyreLife == 1 is excluded (out-lap / pit-exit contamination).
        Median is used instead of mean for robustness, and the resulting
        curve is enforced to be monotonically non-decreasing.
        """
        feature_df = ml_data.get("feature_df")
        if feature_df is None:
            logger.warning("feature_df not in ml_data — compound profiles unavailable")
            return

        self.compound_profiles = {}
        for enc, cname in _COMPOUND_REVERSE.items():
            mask = feature_df["CompoundEncoded"] == enc
            if mask.sum() < 5:
                continue
            subset = feature_df.loc[mask]
            # Drop TyreLife <= 1 — first lap of a stint is contaminated
            # by pit-exit / cold-tyre effects and is not real degradation.
            subset = subset[subset["TyreLife"] >= 2]
            if len(subset) < 5:
                continue
            grouped = subset.groupby(subset["TyreLife"].astype(int))["DegradationTarget"].median()
            tyre_lives = grouped.index.values.astype(float)
            degradation = np.clip(grouped.values, 0.0, MAX_DEGRADATION)
            order = np.argsort(tyre_lives)
            tyre_lives = tyre_lives[order]
            degradation = degradation[order]
            # Enforce monotonically non-decreasing
            for i in range(1, len(degradation)):
                if degradation[i] < degradation[i - 1]:
                    degradation[i] = degradation[i - 1]
            # Prepend TyreLife=1 with a smooth start value
            if len(tyre_lives) > 0 and tyre_lives[0] > 1:
                start_val = max(degradation[0] * 0.9, 0.0)
                tyre_lives = np.concatenate([[1.0], tyre_lives])
                degradation = np.concatenate([[start_val], degradation])
            self.compound_profiles[cname] = {
                "tyre_lives": tyre_lives,
                "degradation": degradation,
            }
        logger.info(
            "Compound profiles built: %s",
            {k: len(v["tyre_lives"]) for k, v in self.compound_profiles.items()},
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_degradation(self, compound_or_X, tyre_life=None, forecast_laps=None):
        """Predict degradation — two calling conventions:

        1. ``predict_degradation(X: np.ndarray)``
           Raw regression prediction on a scaled feature array.
           Used by HybridModel and strategy.py.

        2. ``predict_degradation(compound: str, tyre_life: int, forecast_laps: int)``
           Compound-based degradation curve used for comparison plots.
           Uses stored compound_profiles with interpolation/extrapolation.
           Capped at 5.0 s.
        """
        # ---- Raw array path (backward compatible) ----
        if isinstance(compound_or_X, np.ndarray):
            return self.reg_model.predict(compound_or_X)

        # ---- Compound-based curve path ----
        compound = str(compound_or_X)
        tyre_life = int(tyre_life) if tyre_life is not None else 0
        forecast_laps = int(forecast_laps) if forecast_laps is not None else 15

        if compound in self.compound_profiles:
            profile = self.compound_profiles[compound]
            tl = profile["tyre_lives"]
            dg = profile["degradation"]
        else:
            # Minimal fallback when profiles are missing
            tl = np.array([1.0, 40.0])
            dg = np.array([0.0, 2.0])

        result = np.empty(forecast_laps, dtype=np.float64)
        for i in range(forecast_laps):
            life = tyre_life + i + 1
            # np.interp handles in-range interpolation
            pred = float(np.interp(life, tl, dg))
            # Extrapolation beyond data range
            if life > tl[-1] and len(tl) >= 2:
                rate = (dg[-1] - dg[-2]) / max(tl[-1] - tl[-2], 1.0)
                pred = float(dg[-1]) + rate * (life - tl[-1])
            result[i] = min(max(pred, 0.0), MAX_DEGRADATION)
        return result

    def predict_pit(self, X: np.ndarray) -> np.ndarray:
        """Predict pit stop decision (0/1) using tuned threshold."""
        proba = self.cls_model.predict_proba(X)[:, 1]
        threshold = getattr(self, '_pit_threshold', 0.52)
        return (proba >= threshold).astype(int)

    def predict_pit_probability(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of needing a pit stop."""
        return self.cls_model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Pit recommendation
    # ------------------------------------------------------------------
    def get_pit_recommendation(self, compound: str, current_tyre_life: int,
                               threshold: float = 1.5,
                               max_forecast: int = 30) -> int:
        """Return laps until degradation exceeds *threshold*, or -1."""
        forecast = self.predict_degradation(compound, current_tyre_life, max_forecast)
        for i, deg in enumerate(forecast):
            if deg > threshold:
                return i + 1
        return -1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, ml_data: dict) -> dict:
        """Evaluate both models on test data."""
        X_test = ml_data["X_test"]
        y_reg_test = ml_data["y_reg_test"]
        y_cls_test = ml_data["y_cls_test"]

        # Regression metrics
        reg_pred = self.reg_model.predict(X_test)
        reg_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_reg_test, reg_pred))),
            "mae": float(mean_absolute_error(y_reg_test, reg_pred)),
            "r2": float(r2_score(y_reg_test, reg_pred)),
        }

        # Classification metrics
        cls_pred = self.predict_pit(X_test)
        cls_metrics = {
            "accuracy": float(accuracy_score(y_cls_test, cls_pred)),
            "precision": float(precision_score(y_cls_test, cls_pred, zero_division=0)),
            "recall": float(recall_score(y_cls_test, cls_pred, zero_division=0)),
            "f1": float(f1_score(y_cls_test, cls_pred, zero_division=0)),
        }

        self.metrics = {
            "regression": reg_metrics,
            "classification": cls_metrics,
            "rmse": reg_metrics["rmse"],
            "mae": reg_metrics["mae"],
            "r2": reg_metrics["r2"],
        }

        logger.info(
            "XGBoost Regression: RMSE=%.4f, MAE=%.4f, R2=%.4f",
            reg_metrics["rmse"], reg_metrics["mae"], reg_metrics["r2"],
        )
        logger.info(
            "XGBoost Classification: Acc=%.4f, F1=%.4f",
            cls_metrics["accuracy"], cls_metrics["f1"],
        )

        return self.metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save models, metrics, and compound profiles to disk."""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.reg_model, os.path.join(path, "xgboost_reg.pkl"))
        joblib.dump(self.cls_model, os.path.join(path, "xgboost_cls.pkl"))
        joblib.dump(self.feature_importance, os.path.join(path, "xgboost_features.pkl"))
        joblib.dump(self.metrics, os.path.join(path, "xgboost_metrics.pkl"))
        joblib.dump(self.compound_profiles, os.path.join(path, "xgboost_profiles.pkl"))

    def load(self, path: str):
        """Load models, metrics, and compound profiles from disk."""
        self.reg_model = joblib.load(os.path.join(path, "xgboost_reg.pkl"))
        self.cls_model = joblib.load(os.path.join(path, "xgboost_cls.pkl"))
        self.feature_importance = joblib.load(os.path.join(path, "xgboost_features.pkl"))

        # Metrics (added in rewrite — graceful fallback for old saves)
        metrics_path = os.path.join(path, "xgboost_metrics.pkl")
        if os.path.exists(metrics_path):
            self.metrics = joblib.load(metrics_path)
        else:
            self.metrics = {}

        # Compound profiles (added in rewrite — graceful fallback)
        profiles_path = os.path.join(path, "xgboost_profiles.pkl")
        if os.path.exists(profiles_path):
            self.compound_profiles = joblib.load(profiles_path)
        else:
            self.compound_profiles = {}
