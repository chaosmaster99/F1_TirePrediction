"""
Pit Stop Strategy Engine
Uses the best model to recommend optimal pit stop timing.
This is the core decision-making module used by the frontend.
"""

import numpy as np
import pandas as pd
import os
import json
import joblib
import logging

logger = logging.getLogger(__name__)

# Default degradation thresholds (seconds) by compound
# When degradation exceeds this, it's time to pit
DEFAULT_THRESHOLDS = {
    "SOFT": 1.2,
    "MEDIUM": 1.5,
    "HARD": 2.0,
}

# Typical pit stop time loss (seconds)
PIT_STOP_LOSS = 22.0  # Average time lost during a pit stop

# Optimal strategies for common scenarios
COMPOUND_STRATEGIES = {
    "SOFT": {"next": ["MEDIUM", "HARD"], "typical_stint": 18},
    "MEDIUM": {"next": ["HARD", "SOFT"], "typical_stint": 28},
    "HARD": {"next": ["MEDIUM", "SOFT"], "typical_stint": 40},
}

# Driver name mapping: full name -> abbreviation used in data
DRIVER_ABBREV = {
    "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM",
    "Charles Leclerc": "LEC",
    "Lando Norris": "NOR",
    "Carlos Sainz": "SAI",
}

# Driver-specific degradation profiles calibrated to realistic F1 stint lengths.
# deg_rate is back-solved from: threshold = rate * (0.015 * life² + 0.6 * life)
# Target stint lengths reflect typical race-distance pitting (not early strategic pits).
DRIVER_PROFILES = {
    "VER": {
        "name": "Max Verstappen",
        "team": "Red Bull Racing",
        "style": "aggressive",
        "SOFT":   {"deg_rate": 0.0826, "typical_stint": 17},
        "MEDIUM": {"deg_rate": 0.0553, "typical_stint": 27},
        "HARD":   {"deg_rate": 0.0450, "typical_stint": 38},
    },
    "HAM": {
        "name": "Lewis Hamilton",
        "team": "Ferrari",
        "style": "balanced",
        "SOFT":   {"deg_rate": 0.0893, "typical_stint": 16},
        "MEDIUM": {"deg_rate": 0.0583, "typical_stint": 26},
        "HARD":   {"deg_rate": 0.0468, "typical_stint": 37},
    },
    "LEC": {
        "name": "Charles Leclerc",
        "team": "Ferrari",
        "style": "smooth",
        "SOFT":   {"deg_rate": 0.0970, "typical_stint": 15},
        "MEDIUM": {"deg_rate": 0.0651, "typical_stint": 24},
        "HARD":   {"deg_rate": 0.0530, "typical_stint": 34},
    },
    "NOR": {
        "name": "Lando Norris",
        "team": "McLaren",
        "style": "aggressive",
        "SOFT":   {"deg_rate": 0.1058, "typical_stint": 14},
        "MEDIUM": {"deg_rate": 0.0690, "typical_stint": 23},
        "HARD":   {"deg_rate": 0.0554, "typical_stint": 33},
    },
    "SAI": {
        "name": "Carlos Sainz",
        "team": "Williams",
        "style": "smooth",
        "SOFT":   {"deg_rate": 0.0970, "typical_stint": 15},
        "MEDIUM": {"deg_rate": 0.0690, "typical_stint": 23},
        "HARD":   {"deg_rate": 0.0554, "typical_stint": 33},
    },
}

# Default (generic) profile used when driver is unknown
DEFAULT_PROFILE = {
    "SOFT":   {"deg_rate": 0.0970, "typical_stint": 15},
    "MEDIUM": {"deg_rate": 0.0651, "typical_stint": 24},
    "HARD":   {"deg_rate": 0.0508, "typical_stint": 35},
}

# Total race laps for known circuits (2024/2025 season)
RACE_LAPS = {
    "Melbourne": 58,
    "Albert Park": 58,
    "Bahrain": 57,
    "Jeddah": 50,
    "Suzuka": 53,
    "Shanghai": 56,
    "Miami": 57,
    "Imola": 63,
    "Monaco": 78,
    "Montreal": 70,
    "Barcelona": 66,
    "Catalunya": 66,
    "Red Bull Ring": 71,
    "Spielberg": 71,
    "Silverstone": 52,
    "Budapest": 70,
    "Hungaroring": 70,
    "Spa": 44,
    "Spa-Francorchamps": 44,
    "Zandvoort": 72,
    "Monza": 53,
    "Baku": 51,
    "Marina Bay": 62,
    "Singapore": 62,
    "COTA": 56,
    "Austin": 56,
    "Mexico City": 71,
    "Interlagos": 71,
    "São Paulo": 71,
    "Las Vegas": 50,
    "Lusail": 57,
    "Qatar": 57,
    "Yas Marina": 58,
    "Abu Dhabi": 58,
}


class PitStopStrategy:
    """
    Pit Stop Strategy Engine.
    
    Given a driver, circuit, and starting compound, predicts:
    1. When each pit stop should occur
    2. What compound to switch to
    3. Expected time saved/lost from the strategy
    4. Full race strategy breakdown
    
    Uses the best trained model (XGBoost) for predictions, with an
    analytical fallback when model files are unavailable.
    """
    
    # Compound name → integer encoding (matches LabelEncoder trained order)
    COMPOUND_ENCODING = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.best_model = None
        self.best_model_name = None
        self.arima_model = None
        self.var_model = None
        self.lstm_model = None
        self.all_models = {}
        self.processed_df = None
        # XGBoost inference artefacts
        self.xgb_model = None
        self.scaler = None
        self.le_circuit = None
        self.le_driver = None
        self.feature_cols = None
        self.baseline_stats = None
        self._xgb_ready = False
    
    def load_models(self, comparison=None):
        """Load models from comparison object or disk."""
        if comparison:
            self.all_models = comparison.models
            self.best_model_name = comparison.best_model_name
            self.best_model = comparison.best_model
            if "ARIMA" in comparison.models:
                self.arima_model = comparison.models["ARIMA"]
            if "VAR" in comparison.models:
                self.var_model = comparison.models["VAR"]
            if "LSTM" in comparison.models:
                self.lstm_model = comparison.models["LSTM"]
        
        # Load best-model metadata
        info_path = os.path.join(self.models_dir, "best_model_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
                self.best_model_name = info.get("best_model", self.best_model_name)
        
        # ---- Load XGBoost model ----
        xgb_dir = os.path.join(self.models_dir, "xgboost")
        prep_dir = os.path.join(self.models_dir, "preprocessing")
        try:
            from src.models.xgboost_model import XGBoostPitModel
            model = XGBoostPitModel()
            model.load(xgb_dir)
            self.xgb_model = model
            
            self.scaler = joblib.load(os.path.join(prep_dir, "scaler.pkl"))
            self.le_circuit = joblib.load(os.path.join(prep_dir, "le_circuit.pkl"))
            self.le_driver = joblib.load(os.path.join(prep_dir, "le_driver.pkl"))
            self.feature_cols = joblib.load(os.path.join(prep_dir, "feature_cols.pkl"))
            self.baseline_stats = joblib.load(os.path.join(prep_dir, "baseline_stats.pkl"))
            self._xgb_ready = True
            logger.info("XGBoost model + preprocessing artifacts loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load XGBoost model: {e}. Using analytical fallback.")
            self._xgb_ready = False
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_driver_code(self, driver: str = None) -> str:
        """Resolve a driver full name or abbreviation to a profile key."""
        if driver is None:
            return None
        if driver in DRIVER_PROFILES:
            return driver
        code = DRIVER_ABBREV.get(driver)
        if code and code in DRIVER_PROFILES:
            return code
        return None

    def _encode_safe(self, encoder, value, fallback: int = 0) -> int:
        """Encode a label, returning *fallback* for unseen values."""
        try:
            return int(encoder.transform([value])[0])
        except (ValueError, KeyError):
            return fallback

    # ------------------------------------------------------------------
    # XGBoost feature-vector construction
    # ------------------------------------------------------------------
    def _build_feature_row(self, tyre_life: int, compound: str,
                            stint_lap: int, race_progress: float,
                            total_race_laps: int,
                            driver_code: str, circuit: str) -> np.ndarray:
        """
        Build a single feature row in the same column order the XGBoost
        model was trained on.
        
        Feature order (9 cols — no leaked features):
        TyreLife, CompoundEncoded, StintLap, RaceProgress,
        FuelEffect, TyreLifeNorm, AvgSpeed,
        CircuitEncoded, DriverEncoded
        """
        compound_enc = self.COMPOUND_ENCODING.get(compound, 1)
        circuit_enc = self._encode_safe(self.le_circuit, circuit) if self.le_circuit else 0
        driver_enc = self._encode_safe(self.le_driver, driver_code) if self.le_driver else 0
        
        # Look up baseline stats for this driver + compound
        key = f"{driver_code}_{compound}"
        drv_stats = (self.baseline_stats or {}).get(key, {})
        global_stats = (self.baseline_stats or {}).get("global", {})
        circuit_stats = (self.baseline_stats or {}).get(f"circuit_{circuit}", {})
        
        # Fuel effect: decreases linearly as race progresses (lighter car)
        fuel_effect = max(0.0, 3.0 * (1.0 - race_progress))
        
        # TyreLifeNorm: normalised tyre life (fraction of expected compound life)
        # Must match COMPOUND_EXPECTED_LIFE used during training in preprocessing.py
        COMPOUND_EXPECTED_LIFE = {"SOFT": 18, "MEDIUM": 28, "HARD": 40}
        expected_life = COMPOUND_EXPECTED_LIFE.get(compound, 30)
        tyre_life_norm = tyre_life / expected_life
        
        # AvgSpeed: use circuit-specific or driver-specific median
        avg_speed = circuit_stats.get("AvgSpeed",
                    drv_stats.get("AvgSpeed",
                    global_stats.get("AvgSpeed", 250.0)))
        
        row = np.array([[
            tyre_life,
            compound_enc,
            stint_lap,
            race_progress,
            fuel_effect,
            tyre_life_norm,
            avg_speed,
            circuit_enc,
            driver_enc,
        ]])
        return row

    # ------------------------------------------------------------------
    # Degradation prediction
    # ------------------------------------------------------------------
    def predict_degradation_curve(self, compound: str, total_laps: int,
                                   start_tyre_life: int = 0,
                                   driver: str = None,
                                   circuit: str = None,
                                   total_race_laps: int = None) -> np.ndarray:
        """
        Predict the degradation curve for a compound over multiple laps.
        
        Uses the trained XGBoost model when available; otherwise falls
        back to the driver-aware analytical model.
        
        Returns array of degradation values (seconds lost vs fresh-tyre pace).
        """
        forecast_laps = total_laps - start_tyre_life
        driver_code = self._resolve_driver_code(driver) or "VER"
        total_race = total_race_laps or total_laps
        
        # ---- XGBoost path ----
        if self._xgb_ready:
            try:
                return self._xgb_degradation(
                    compound, start_tyre_life, forecast_laps,
                    driver_code, circuit or "Unknown", total_race
                )
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}. Falling back to analytical.")
        
        # ---- Analytical fallback ----
        return self._analytical_degradation(compound, start_tyre_life, forecast_laps, driver)

    def _xgb_degradation(self, compound: str, start_life: int,
                          forecast_laps: int, driver_code: str,
                          circuit: str, total_race_laps: int) -> np.ndarray:
        """
        XGBoost-calibrated degradation model.

        Uses the MEDIUM compound as a reference probe to derive a
        driver+circuit specific degradation rate from the XGBoost model,
        then applies physically-grounded compound scaling factors.

        The model (R²≈0.45) reliably captures circuit and driver effects
        (e.g. Sakhir ≈ 3× Melbourne) but cannot reliably order compounds
        due to TyreLifeNorm coupling.  We therefore:
        1. Always probe with MEDIUM → circuit+driver specific slope
        2. Apply compound ratios: SOFT 1.5×, MEDIUM 1.0×, HARD 0.75×
        3. Generate a smooth quadratic curve from the resulting deg_rate
        """
        # -- Probe model with MEDIUM compound (stable reference) ---------
        probe_lives = [3, 6, 10, 14, 18, 20]
        med_preds = []
        for tl in probe_lives:
            row = self._build_feature_row(
                tyre_life=tl, compound="MEDIUM", stint_lap=tl,
                race_progress=tl / max(total_race_laps, 1),
                total_race_laps=total_race_laps,
                driver_code=driver_code, circuit=circuit,
            )
            row_scaled = self.scaler.transform(row)
            med_preds.append(float(
                self.xgb_model.predict_degradation(row_scaled)[0]
            ))

        med_slope = float(np.polyfit(probe_lives, med_preds, 1)[0])
        med_slope = max(med_slope, 0.01)

        # MEDIUM deg_rate from model (no calibration factor needed —
        # the model's absolute scale is correct for MEDIUM/HARD)
        med_deg_rate = med_slope / 0.945

        # Compound scaling factors relative to MEDIUM
        COMPOUND_SCALE = {"SOFT": 1.5, "MEDIUM": 1.0, "HARD": 0.75}
        deg_rate = med_deg_rate * COMPOUND_SCALE.get(compound, 1.0)

        # -- Generate smooth quadratic degradation curve -----------------
        b = deg_rate * 0.6
        a = deg_rate * 0.015

        degradation = []
        base_deg = a * (start_life + 1) ** 2 + b * (start_life + 1)
        for i in range(forecast_laps):
            life = start_life + i + 1
            deg = a * life ** 2 + b * life - base_deg
            degradation.append(max(deg, 0.0))

        return np.array(degradation)

    def _analytical_degradation(self, compound: str, start_life: int,
                                 forecast_laps: int, driver: str = None) -> np.ndarray:
        """
        Driver-aware analytical degradation model.
        Uses a quadratic model: deg = a * tyre_life^2 + b * tyre_life
        
        Coefficients are derived from historical per-driver stint data.
        The linear term (b) is the driver's median degradation rate and
        quadratic term (a) adds realistic acceleration of wear.
        """
        driver_code = self._resolve_driver_code(driver)
        
        if driver_code and driver_code in DRIVER_PROFILES:
            profile = DRIVER_PROFILES[driver_code]
            compound_profile = profile.get(compound, DEFAULT_PROFILE.get(compound, {}))
            deg_rate = compound_profile.get("deg_rate", 0.04)
        else:
            deg_rate = DEFAULT_PROFILE.get(compound, {}).get("deg_rate", 0.04)
        
        # Derive quadratic (a) and linear (b) coefficients from the degradation rate.
        # Split: linear dominant + mild quadratic acceleration gives realistic curves
        # that start slow and accelerate gently as tires age.
        b = deg_rate * 0.6
        a = deg_rate * 0.015  # Mild quadratic to model accelerating wear
        
        degradation = []
        for i in range(forecast_laps):
            life = start_life + i + 1
            deg = a * life ** 2 + b * life
            degradation.append(deg)
        
        return np.array(degradation)
    
    def find_optimal_pit_lap(self, compound: str, remaining_laps: int,
                              current_tyre_life: int = 0,
                              threshold: float = None,
                              driver: str = None,
                              circuit: str = None,
                              total_race_laps: int = None) -> dict:
        """
        Find the optimal lap to pit for the current stint.
        
        Uses a cost-benefit analysis:
        - Cost of pitting: ~22 seconds lost
        - Benefit: Resetting tire degradation to fresh tires
        - Optimal pit: when cumulative time loss from degradation
          would exceed pit stop cost + degradation on new tires
        """
        if threshold is None:
            threshold = DEFAULT_THRESHOLDS.get(compound, 1.5)
        
        total_race = total_race_laps or (remaining_laps + current_tyre_life)
        
        # Predict degradation curve (driver-aware, model-backed)
        deg_curve = self.predict_degradation_curve(
            compound, remaining_laps + current_tyre_life, current_tyre_life,
            driver, circuit, total_race
        )
        
        # Find first lap where degradation exceeds threshold
        pit_lap = -1
        for i, deg in enumerate(deg_curve):
            if deg > threshold:
                pit_lap = current_tyre_life + i + 1
                break
        
        if pit_lap == -1:
            # No pit needed (tires can last the remaining laps)
            pit_lap = current_tyre_life + remaining_laps
        
        return {
            "pit_lap": pit_lap,
            "laps_until_pit": pit_lap - current_tyre_life,
            "degradation_at_pit": deg_curve[min(pit_lap - current_tyre_life - 1, len(deg_curve) - 1)] if len(deg_curve) > 0 else 0,
            "degradation_curve": deg_curve,
        }
    
    def generate_strategy(self, starting_compound: str, total_race_laps: int,
                           circuit: str = None, driver: str = None,
                           threshold: float = None) -> dict:
        """
        Generate a complete pit stop strategy for the race.
        
        Args:
            starting_compound: Starting tire compound
            total_race_laps: Total laps in the race
            circuit: Circuit name (optional, for circuit-specific adjustments)
            driver: Driver identifier (optional)
            threshold: Degradation threshold for pit decision
        
        Returns:
            Complete strategy with stint information
        """
        # Get total laps from circuit if not specified
        if total_race_laps is None and circuit:
            total_race_laps = RACE_LAPS.get(circuit, 56)
        
        strategy = {
            "circuit": circuit or "Unknown",
            "driver": driver or "Generic",
            "total_laps": total_race_laps,
            "stints": [],
            "total_pit_stops": 0,
            "total_pit_time_loss": 0,
            "compounds_used": [],
        }
        
        current_lap = 1
        current_compound = starting_compound
        current_tyre_life = 0
        stint_number = 1
        compounds_used = [current_compound]
        
        # Compute per-compound thresholds.
        # If the user supplied a threshold via the slider, treat it as a
        # scaling factor relative to the starting compound's default.
        # E.g. slider=1.8 with starting MEDIUM (default 1.5) → scale=1.2
        #      → HARD threshold = 2.0 * 1.2 = 2.4, SOFT = 1.2 * 1.2 = 1.44
        if threshold is not None:
            base_default = DEFAULT_THRESHOLDS.get(starting_compound, 1.5)
            scale = threshold / base_default if base_default > 0 else 1.0
        else:
            scale = 1.0
        compound_thresholds = {
            cpd: DEFAULT_THRESHOLDS.get(cpd, 1.5) * scale
            for cpd in DEFAULT_THRESHOLDS
        }
        
        while current_lap <= total_race_laps:
            remaining_laps = total_race_laps - current_lap + 1
            
            # Find optimal pit lap for current stint (model-backed, driver-aware)
            # Use per-compound threshold so HARD/SOFT get appropriate values
            stint_threshold = compound_thresholds.get(current_compound, 1.5)
            pit_info = self.find_optimal_pit_lap(
                current_compound, remaining_laps,
                current_tyre_life, stint_threshold, driver,
                circuit, total_race_laps
            )
            
            actual_pit_lap = min(
                current_lap + pit_info["laps_until_pit"] - 1,
                total_race_laps
            )
            stint_length = actual_pit_lap - current_lap + 1
            
            # Degradation curve for this stint
            stint_deg = pit_info["degradation_curve"][:stint_length]
            
            stint_info = {
                "stint_number": stint_number,
                "compound": current_compound,
                "start_lap": current_lap,
                "end_lap": actual_pit_lap,
                "stint_length": stint_length,
                "tyre_life_at_end": current_tyre_life + stint_length,
                "degradation_curve": stint_deg.tolist() if isinstance(stint_deg, np.ndarray) else stint_deg,
                "avg_degradation": float(np.mean(stint_deg)) if len(stint_deg) > 0 else 0,
                "max_degradation": float(np.max(stint_deg)) if len(stint_deg) > 0 else 0,
            }
            
            strategy["stints"].append(stint_info)
            
            if actual_pit_lap < total_race_laps:
                # Pit stop needed - choose next compound
                next_options = COMPOUND_STRATEGIES.get(current_compound, {}).get("next", ["MEDIUM"])
                
                # Choose compound based on remaining laps
                laps_left = total_race_laps - actual_pit_lap
                if laps_left > 35:
                    # Long stint needed: HARD
                    next_compound = "HARD"
                elif laps_left > 20:
                    # Medium stint: MEDIUM
                    next_compound = "MEDIUM"
                else:
                    # Short stint: SOFT (for speed)
                    next_compound = "SOFT"
                
                # Ensure we use at least 2 different compounds (regulation requirement)
                if len(compounds_used) == 1 and next_compound == compounds_used[0]:
                    next_compound = next_options[0]
                
                strategy["total_pit_stops"] += 1
                strategy["total_pit_time_loss"] += PIT_STOP_LOSS
                
                current_compound = next_compound
                compounds_used.append(current_compound)
                current_tyre_life = 0
                current_lap = actual_pit_lap + 1
                stint_number += 1
            else:
                break
        
        strategy["compounds_used"] = list(set(compounds_used))
        
        # F1 Regulation: must use at least 2 different compounds in a dry race.
        # If strategy has only 1 compound, force a late pit stop.
        if len(strategy["compounds_used"]) < 2 and strategy["total_pit_stops"] == 0:
            last_stint = strategy["stints"][-1]
            # Pit 8 laps from the end for a quick final stint on a different compound
            forced_pit_lap = max(last_stint["start_lap"] + 5, total_race_laps - 8)
            second_compound = COMPOUND_STRATEGIES.get(starting_compound, {}).get("next", ["MEDIUM"])[0]
            
            # Trim the last stint
            orig_end = last_stint["end_lap"]
            last_stint["end_lap"] = forced_pit_lap
            last_stint["stint_length"] = forced_pit_lap - last_stint["start_lap"] + 1
            last_stint["tyre_life_at_end"] = last_stint["stint_length"]
            trim_len = last_stint["stint_length"]
            if isinstance(last_stint["degradation_curve"], list) and len(last_stint["degradation_curve"]) > trim_len:
                last_stint["degradation_curve"] = last_stint["degradation_curve"][:trim_len]
            if last_stint["degradation_curve"]:
                last_stint["avg_degradation"] = float(np.mean(last_stint["degradation_curve"]))
                last_stint["max_degradation"] = float(np.max(last_stint["degradation_curve"]))
            
            # Add final stint on new compound
            final_laps = orig_end - forced_pit_lap
            final_deg = self.predict_degradation_curve(
                second_compound, final_laps, 0, driver, circuit, total_race_laps
            )
            strategy["stints"].append({
                "stint_number": stint_number + 1,
                "compound": second_compound,
                "start_lap": forced_pit_lap + 1,
                "end_lap": orig_end,
                "stint_length": final_laps,
                "tyre_life_at_end": final_laps,
                "degradation_curve": final_deg.tolist() if isinstance(final_deg, np.ndarray) else final_deg,
                "avg_degradation": float(np.mean(final_deg)) if len(final_deg) > 0 else 0,
                "max_degradation": float(np.max(final_deg)) if len(final_deg) > 0 else 0,
            })
            strategy["total_pit_stops"] = 1
            strategy["total_pit_time_loss"] = PIT_STOP_LOSS
            strategy["compounds_used"] = [starting_compound, second_compound]
        
        # Calculate estimated total time impact
        total_deg_time = sum(
            s["avg_degradation"] * s["stint_length"] for s in strategy["stints"]
        )
        strategy["estimated_total_degradation_loss"] = round(total_deg_time, 2)
        strategy["estimated_total_pit_time_loss"] = round(strategy["total_pit_time_loss"], 2)
        strategy["estimated_total_time_impact"] = round(
            total_deg_time + strategy["total_pit_time_loss"], 2
        )
        
        return strategy
    
    def compare_strategies(self, circuit: str, total_laps: int = None,
                            driver: str = None, threshold: float = None) -> list:
        """
        Compare strategies starting with each compound.
        Returns list of strategies sorted by total time impact.
        """
        if total_laps is None:
            total_laps = RACE_LAPS.get(circuit, 56)
        
        strategies = []
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            strat = self.generate_strategy(
                compound, total_laps, circuit, driver, threshold
            )
            strategies.append(strat)
        
        # Sort by total time impact (lower is better)
        strategies.sort(key=lambda s: s["estimated_total_time_impact"])
        
        # Mark best
        for i, s in enumerate(strategies):
            s["rank"] = i + 1
            s["is_optimal"] = (i == 0)
        
        return strategies
    
    def get_recommendation_text(self, strategy: dict) -> str:
        """Generate human-readable strategy recommendation."""
        lines = []
        lines.append(f"Race Strategy for {strategy['driver']} at {strategy['circuit']}")
        lines.append(f"Total Race Laps: {strategy['total_laps']}")
        lines.append(f"Number of Pit Stops: {strategy['total_pit_stops']}")
        lines.append(f"Compounds Used: {', '.join(strategy['compounds_used'])}")
        lines.append("")
        
        for stint in strategy["stints"]:
            lines.append(f"Stint {stint['stint_number']}: {stint['compound']}")
            lines.append(f"  Laps {stint['start_lap']} - {stint['end_lap']} "
                         f"({stint['stint_length']} laps)")
            lines.append(f"  Tire Life at End: {stint['tyre_life_at_end']} laps")
            lines.append(f"  Avg Degradation: {stint['avg_degradation']:.3f}s")
            lines.append(f"  Max Degradation: {stint['max_degradation']:.3f}s")
            lines.append("")
        
        lines.append(f"Estimated Time Lost to Degradation: "
                      f"{strategy['estimated_total_degradation_loss']:.1f}s")
        lines.append(f"Estimated Time Lost to Pit Stops: "
                      f"{strategy['estimated_total_pit_time_loss']:.1f}s")
        lines.append(f"Total Estimated Time Impact: "
                      f"{strategy['estimated_total_time_impact']:.1f}s")
        
        return "\n".join(lines)
