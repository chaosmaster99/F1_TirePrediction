"""
Hybrid Model: Residual Stacking (ARIMA + XGBoost)

Standard residual hybrid approach:
    1. ARIMA provides a base degradation prediction for each (compound, TyreLife).
    2. Residual = Actual − ARIMA_prediction is computed on training data.
    3. XGBoost is trained to predict the residual from the full feature set
       (TyreLife, Compound, StintLap, FuelEffect, TyreLifeNorm, RaceProgress,
        AvgSpeed, Circuit, Driver).
    4. Final prediction = ARIMA_base + XGBoost_residual.

Evaluated at instance level (same granularity as standalone XGBoost/LSTM).
"""

import numpy as np
import os
import joblib
import logging
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

MAX_DEGRADATION = 5.0
COMPOUND_MAP = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
_COMPOUND_REVERSE = {1: "SOFT", 2: "MEDIUM", 3: "HARD"}


class HybridModel:
    """Residual stacking: ARIMA base + XGBoost residual correction."""

    def __init__(self, arima_model):
        self.arima = arima_model
        self.residual_model = None          # XGBoost trained on residuals
        self.metrics = {}
        self.compound_profiles = {}
        self.name = "Hybrid (ARIMA+XGBoost)"

    # ------------------------------------------------------------------
    # ARIMA look-up for each (compound, TyreLife) pair
    # ------------------------------------------------------------------
    def _get_arima_predictions(self, tyre_life_arr, compound_enc_arr):
        """Vectorised ARIMA look-up: history curve for in-range, last
        known value for out-of-range laps."""
        preds = np.zeros(len(tyre_life_arr), dtype=np.float64)
        for i in range(len(tyre_life_arr)):
            tl = int(tyre_life_arr[i])
            compound = _COMPOUND_REVERSE.get(int(compound_enc_arr[i]), "MEDIUM")
            if compound in self.arima.history:
                hist = self.arima.history[compound]
                idx = tl - 2          # history[0] ↔ TyreLife 2
                if tl <= 1:
                    preds[i] = 0.0
                elif idx < len(hist):
                    preds[i] = float(hist[idx])
                else:
                    preds[i] = float(hist[-1])
            else:
                preds[i] = 0.0
        return preds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, ml_data: dict, ts_data: dict = None):
        """Train XGBoost to predict residual = actual − ARIMA prediction.

        Uses the same train/test split as the standalone XGBoost model.
        Residuals are computed only on training predictions (no leakage).
        """
        logger.info("Fitting Hybrid (ARIMA + XGBoost residual) ...")

        feature_df = ml_data["feature_df"]
        split_idx = len(ml_data["X_train"])

        # --- Training residuals ---
        tl_train = feature_df["TyreLife"].values[:split_idx]
        ce_train = feature_df["CompoundEncoded"].values[:split_idx]
        arima_train = self._get_arima_predictions(tl_train, ce_train)
        residuals_train = ml_data["y_reg_train"] - arima_train

        # --- Test residuals (for XGBoost early-stopping only) ---
        tl_test = feature_df["TyreLife"].values[split_idx:]
        ce_test = feature_df["CompoundEncoded"].values[split_idx:]
        arima_test = self._get_arima_predictions(tl_test, ce_test)
        residuals_test = ml_data["y_reg_test"] - arima_test

        # --- Train residual XGBoost ---
        self.residual_model = xgb.XGBRegressor(
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
        self.residual_model.fit(
            ml_data["X_train"], residuals_train,
            eval_set=[(ml_data["X_test"], residuals_test)],
            verbose=False,
        )

        logger.info(
            "Residual XGBoost trained  |  Train residual mean=%.4f std=%.4f",
            float(np.mean(residuals_train)), float(np.std(residuals_train)),
        )

        # --- Build compound profiles for curve plots / strategy ---
        self._build_compound_profiles(ml_data)

        logger.info("Hybrid model fitting complete")

    # ------------------------------------------------------------------
    # Compound profiles (averaged ARIMA+XGB curves for plots / strategy)
    # ------------------------------------------------------------------
    def _build_compound_profiles(self, ml_data: dict):
        """Build per-compound degradation curves from training predictions."""
        feature_df = ml_data["feature_df"]
        split_idx = len(ml_data["X_train"])
        train_df = feature_df.iloc[:split_idx].copy()

        arima_preds = self._get_arima_predictions(
            train_df["TyreLife"].values,
            train_df["CompoundEncoded"].values,
        )
        xgb_residuals = self.residual_model.predict(ml_data["X_train"])
        train_df["HybridPred"] = arima_preds + xgb_residuals

        self.compound_profiles = {}
        for enc, cname in _COMPOUND_REVERSE.items():
            mask = train_df["CompoundEncoded"] == enc
            if mask.sum() < 5:
                continue
            subset = train_df.loc[mask]
            # Drop TyreLife <= 1 — first lap of a stint is contaminated
            # by pit-exit / cold-tyre effects.
            subset = subset[subset["TyreLife"] >= 2]
            if len(subset) < 5:
                continue
            grouped = subset.groupby(subset["TyreLife"].astype(int))["HybridPred"].median()
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
    # Prediction (curve-level — for plots and strategy)
    # ------------------------------------------------------------------
    def predict_degradation(self, compound: str, current_tyre_life: int,
                            forecast_laps: int = 15) -> np.ndarray:
        """Degradation curve using stored compound profiles (ARIMA+XGB
        averaged over training data).  Falls back to ARIMA-only when
        profiles are unavailable."""
        cname = compound.upper()
        if cname in self.compound_profiles:
            profile = self.compound_profiles[cname]
            tl = profile["tyre_lives"]
            dg = profile["degradation"]
        else:
            return self.arima.predict_degradation(cname, current_tyre_life,
                                                  forecast_laps)

        result = np.empty(forecast_laps, dtype=np.float64)
        for i in range(forecast_laps):
            life = current_tyre_life + i + 1
            pred = float(np.interp(life, tl, dg))
            # Extrapolate beyond data range
            if life > tl[-1] and len(tl) >= 2:
                rate = (dg[-1] - dg[-2]) / max(tl[-1] - tl[-2], 1.0)
                pred = float(dg[-1]) + rate * (life - tl[-1])
            result[i] = min(max(pred, 0.0), MAX_DEGRADATION)
        return result

    # ------------------------------------------------------------------
    # Evaluation (instance-level — same granularity as XGBoost / LSTM)
    # ------------------------------------------------------------------
    def evaluate(self, ml_data: dict = None, **_) -> dict:
        """Evaluate: final = ARIMA_base + XGBoost_residual vs actual."""
        if ml_data is None:
            self.metrics = {"rmse": float("inf"), "mae": float("inf"), "r2": -1.0}
            return self.metrics

        feature_df = ml_data["feature_df"]
        split_idx = len(ml_data["X_train"])

        tl_test = feature_df["TyreLife"].values[split_idx:]
        ce_test = feature_df["CompoundEncoded"].values[split_idx:]
        arima_preds = self._get_arima_predictions(tl_test, ce_test)
        xgb_residuals = self.residual_model.predict(ml_data["X_test"])
        final_preds = arima_preds + xgb_residuals

        y_test = ml_data["y_reg_test"]
        self.metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, final_preds))),
            "mae": float(mean_absolute_error(y_test, final_preds)),
            "r2": float(r2_score(y_test, final_preds)),
        }

        logger.info(
            "Hybrid Evaluation: RMSE=%.4f, MAE=%.4f, R2=%.4f",
            self.metrics["rmse"], self.metrics["mae"], self.metrics["r2"],
        )
        return self.metrics

    # ------------------------------------------------------------------
    # Strategy helper
    # ------------------------------------------------------------------
    def get_pit_recommendation(self, compound: str, current_tyre_life: int,
                               threshold: float = 1.5,
                               max_forecast: int = 30) -> int:
        """Laps until predicted degradation exceeds *threshold*, or -1."""
        forecast = self.predict_degradation(compound, current_tyre_life, max_forecast)
        for i, deg in enumerate(forecast):
            if deg > threshold:
                return i + 1
        return -1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save residual XGBoost model, metrics, and compound profiles."""
        os.makedirs(path, exist_ok=True)
        # Save residual XGBoost
        if self.residual_model is not None:
            self.residual_model.save_model(os.path.join(path, "residual_xgb.json"))
        # Save config
        joblib.dump({
            "metrics": self.metrics,
            "compound_profiles": self.compound_profiles,
        }, os.path.join(path, "hybrid_config.pkl"))
        logger.info("Hybrid model saved to %s", path)

    def load(self, path: str):
        """Load residual XGBoost model and config."""
        # Load residual XGBoost
        xgb_path = os.path.join(path, "residual_xgb.json")
        if os.path.exists(xgb_path):
            self.residual_model = xgb.XGBRegressor()
            self.residual_model.load_model(xgb_path)

        # Load config
        config_path = os.path.join(path, "hybrid_config.pkl")
        if os.path.exists(config_path):
            cfg = joblib.load(config_path)
            self.metrics = cfg.get("metrics", {})
            self.compound_profiles = cfg.get("compound_profiles", {})

        logger.info("Hybrid model loaded from %s", path)
