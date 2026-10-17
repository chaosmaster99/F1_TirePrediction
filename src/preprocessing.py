"""
Data Preprocessing Module for F1 Pit Stop Prediction
Handles cleaning, feature engineering, and dataset preparation for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings
import logging
import os

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Compound mapping (softest to hardest)
COMPOUND_MAP = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
    "INTERMEDIATE": 4,
    "WET": 5,
}

# Compound expected life (laps) - approximate base values
COMPOUND_EXPECTED_LIFE = {
    "SOFT": 18,
    "MEDIUM": 28,
    "HARD": 40,
    "INTERMEDIATE": 35,
    "WET": 30,
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw lap data:
    - Remove invalid lap times
    - Remove pit in/out laps (they skew lap times)
    - Remove safety car laps
    - Handle missing values
    """
    logger.info(f"Cleaning data. Initial shape: {df.shape}")
    clean = df.copy()
    
    # Remove rows with missing lap times
    clean = clean.dropna(subset=["LapTimeSec"])
    
    # Remove unrealistic lap times (< 50s or > 200s typically)
    clean = clean[(clean["LapTimeSec"] > 50) & (clean["LapTimeSec"] < 200)]
    
    # Remove pit in/out laps (they have inflated times due to pit lane travel)
    clean = clean[~clean["PitInTime"]]
    clean = clean[~clean["PitOutTime"]]
    
    # Remove safety car laps (TrackStatus != "1" means non-green flag) 
    # TrackStatus "1" = Green, "2" = Yellow, "4" = SC, "6" = VSC, "7" = Red
    clean = clean[clean["TrackStatus"].astype(str).isin(["1", "nan", ""])]
    
    # Remove unknown compounds
    clean = clean[clean["Compound"].isin(COMPOUND_MAP.keys())]
    
    # Remove wet weather compounds for our analysis (pit strategy is very different)
    clean = clean[clean["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
    
    # Fill missing speed values with median per circuit
    speed_cols = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
    for col in speed_cols:
        if col in clean.columns:
            clean[col] = clean.groupby("Circuit")[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Fill remaining missing sector times
    for col in ["Sector1TimeSec", "Sector2TimeSec", "Sector3TimeSec"]:
        if col in clean.columns:
            clean[col] = clean.groupby(["Circuit", "Driver"])[col].transform(
                lambda x: x.fillna(x.median())
            )
    
    clean = clean.dropna(subset=["LapTimeSec", "Compound", "TyreLife"])
    
    logger.info(f"After cleaning: {clean.shape}")
    return clean.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer advanced features for tire degradation modeling.
    
    New features:
    - CompoundEncoded: Numeric encoding of tire compound
    - TyreLifeNorm: Normalized tire life (0-1 based on expected compound life)
    - DegradationRate: Rolling degradation rate (lap time increase per lap)
    - FuelCorrectedLapTime: Approximate fuel-corrected lap time
    - LapTimeDelta: Difference from driver's best lap
    - AvgLapTime: Running average lap time in stint
    - StintLength: Total laps in current stint
    - LapFraction: Fraction of race completed
    - TyreDegradation: Cumulative degradation estimate
    """
    logger.info("Engineering features...")
    feat = df.copy()
    
    # Compound encoding
    feat["CompoundEncoded"] = feat["Compound"].map(COMPOUND_MAP).fillna(0).astype(int)
    
    # Normalized tire life based on expected compound life
    feat["TyreLifeNorm"] = feat.apply(
        lambda row: row["TyreLife"] / COMPOUND_EXPECTED_LIFE.get(row["Compound"], 30), axis=1
    )
    
    # Fuel correction: F1 cars lose ~0.06s per lap due to fuel burn (~1.4kg/lap)
    # Earlier laps are heavier; we add a correction
    max_laps_per_race = feat.groupby(["Year", "RaceName"])["LapNumber"].transform("max")
    fuel_effect = (max_laps_per_race - feat["LapNumber"]) * 0.06
    feat["FuelEffect"] = fuel_effect
    feat["FuelCorrectedLapTime"] = feat["LapTimeSec"] - fuel_effect
    
    # Per-driver, per-race best lap time
    feat["DriverBestLap"] = feat.groupby(["Year", "RaceName", "Driver"])["FuelCorrectedLapTime"].transform("min")
    feat["LapTimeDelta"] = feat["FuelCorrectedLapTime"] - feat["DriverBestLap"]
    
    # Per-circuit baseline (median clean lap time)
    feat["CircuitBaseline"] = feat.groupby("Circuit")["FuelCorrectedLapTime"].transform("median")
    feat["LapTimeVsBaseline"] = feat["FuelCorrectedLapTime"] - feat["CircuitBaseline"]
    
    # Stint-based features
    feat["StintLap"] = feat.groupby(["Year", "RaceName", "Driver", "Stint"]).cumcount() + 1
    feat["StintLength"] = feat.groupby(["Year", "RaceName", "Driver", "Stint"])["LapNumber"].transform("count")
    
    # Rolling degradation: change in lap time over last 3 laps within a stint
    feat = feat.sort_values(["Year", "RaceName", "Driver", "LapNumber"])
    feat["DegradationRate"] = feat.groupby(["Year", "RaceName", "Driver", "Stint"])[
        "FuelCorrectedLapTime"
    ].transform(lambda x: x.diff().rolling(3, min_periods=1).mean())
    
    # Cumulative degradation from stint start
    feat["StintStartLapTime"] = feat.groupby(["Year", "RaceName", "Driver", "Stint"])[
        "FuelCorrectedLapTime"
    ].transform("first")
    feat["TyreDegradation"] = feat["FuelCorrectedLapTime"] - feat["StintStartLapTime"]
    
    # Race progress (fraction of total laps completed)
    feat["RaceProgress"] = feat["LapNumber"] / max_laps_per_race
    
    # Running average lap time within stint
    feat["StintAvgLapTime"] = feat.groupby(["Year", "RaceName", "Driver", "Stint"])[
        "FuelCorrectedLapTime"
    ].transform(lambda x: x.expanding().mean())
    
    # Average speed (if available)
    speed_cols = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
    available_speeds = [c for c in speed_cols if c in feat.columns and feat[c].notna().any()]
    if available_speeds:
        feat["AvgSpeed"] = feat[available_speeds].mean(axis=1)
    else:
        feat["AvgSpeed"] = np.nan
    
    # Fill NaN degradation rates (first laps)
    feat["DegradationRate"] = feat["DegradationRate"].fillna(0)
    feat["TyreDegradation"] = feat["TyreDegradation"].fillna(0)
    
    logger.info(f"Engineered features. Shape: {feat.shape}")
    return feat


def create_degradation_target(df: pd.DataFrame, threshold_pct: float = 0.03) -> pd.DataFrame:
    """
    Create the target variable: whether the car should pit in the next N laps.
    
    The idea: when tire degradation causes lap times to increase beyond a threshold
    relative to optimal pace, it's time to pit.
    
    Args:
        df: DataFrame with engineered features
        threshold_pct: Percentage degradation threshold (default 3%)
    
    Returns:
        DataFrame with 'ShouldPit' target column
    """
    feat = df.copy()
    
    # Target 1: degradation exceeds threshold
    feat["DegradationPct"] = feat["LapTimeDelta"] / feat["DriverBestLap"]
    feat["ShouldPit"] = (feat["DegradationPct"] > threshold_pct).astype(int)
    
    # Target 2: predicted degradation (continuous) for regression
    feat["DegradationTarget"] = feat["LapTimeDelta"]
    
    return feat


def prepare_time_series_data(df: pd.DataFrame, driver: str = None, circuit: str = None) -> dict:
    """
    Prepare data specifically for time series models.
    Returns data organized by stint for time series analysis.
    """
    ts_data = df.copy()
    
    if driver:
        ts_data = ts_data[ts_data["Driver"] == driver]
    if circuit:
        ts_data = ts_data[ts_data["Circuit"] == circuit]
    
    # Group by stint and create time series per stint
    stints = {}
    for (year, race, drv, stint), group in ts_data.groupby(["Year", "RaceName", "Driver", "Stint"]):
        if len(group) >= 5:  # Minimum stint length for meaningful analysis
            key = f"{year}_{race}_{drv}_stint{stint}"
            stints[key] = group.sort_values("LapNumber")[
                ["LapNumber", "TyreLife", "FuelCorrectedLapTime", "DegradationRate",
                 "TyreDegradation", "Compound", "CompoundEncoded", "LapTimeDelta",
                 "StintLap", "AvgSpeed"]
            ].reset_index(drop=True)
    
    return stints


def prepare_ml_data(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Prepare train/test split for ML/DL models.
    
    Features used for prediction:
    - TyreLife, CompoundEncoded, StintLap, RaceProgress
    - FuelEffect, DegradationRate, AvgSpeed
    - Circuit-encoded features
    
    Target: FuelCorrectedLapTime (regression) or ShouldPit (classification)
    """
    # NOTE: DegradationRate, TyreDegradation and LapTimeDelta are EXCLUDED
    # because they are derived from actual lap times (the target variable).
    # Including them causes data leakage — the model learns to copy its inputs
    # instead of predicting degradation from knowable race-state features.
    feature_cols = [
        "TyreLife", "CompoundEncoded", "StintLap", "RaceProgress",
        "FuelEffect", "TyreLifeNorm",
    ]
    
    # Add speed if available
    if "AvgSpeed" in df.columns and df["AvgSpeed"].notna().any():
        df["AvgSpeed"] = df["AvgSpeed"].fillna(df["AvgSpeed"].median())
        feature_cols.append("AvgSpeed")
    
    # Encode Circuit
    le_circuit = LabelEncoder()
    df["CircuitEncoded"] = le_circuit.fit_transform(df["Circuit"].astype(str))
    feature_cols.append("CircuitEncoded")
    
    # Encode Driver
    le_driver = LabelEncoder()
    df["DriverEncoded"] = le_driver.fit_transform(df["Driver"].astype(str))
    feature_cols.append("DriverEncoded")
    
    # Clean up any remaining NaN
    ml_df = df[feature_cols + ["FuelCorrectedLapTime", "ShouldPit", "DegradationTarget"]].dropna()
    
    X = ml_df[feature_cols].values
    y_regression = ml_df["DegradationTarget"].values
    y_classification = ml_df["ShouldPit"].values
    
    # Temporal train/test split: data is sorted chronologically from
    # engineer_features(), so a positional split gives time-based separation.
    # Scaler is fit on training portion only to prevent data leakage.
    split_idx = int(len(X) * (1 - test_size))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split_idx])
    X_test = scaler.transform(X[split_idx:])
    y_reg_train, y_reg_test = y_regression[:split_idx], y_regression[split_idx:]
    y_cls_train, y_cls_test = y_classification[:split_idx], y_classification[split_idx:]
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_reg_train": y_reg_train,
        "y_reg_test": y_reg_test,
        "y_cls_train": y_cls_train,
        "y_cls_test": y_cls_test,
        "scaler": scaler,
        "le_circuit": le_circuit,
        "le_driver": le_driver,
        "feature_cols": feature_cols,
        "feature_df": ml_df,
        "processed_df": df,
    }


def prepare_lstm_sequences(df: pd.DataFrame, seq_length: int = 8) -> dict:
    """
    Prepare sequential data for LSTM model.
    Creates sliding window sequences from stint data.
    """
    feature_cols = [
        "TyreLife", "CompoundEncoded", "StintLap", "FuelEffect",
        "TyreLifeNorm", "RaceProgress",
    ]
    
    # Add speed if available
    if "AvgSpeed" in df.columns and df["AvgSpeed"].notna().any():
        df["AvgSpeed"] = df["AvgSpeed"].fillna(df["AvgSpeed"].median())
        feature_cols.append("AvgSpeed")
    
    # Encode Circuit and Driver (same approach as prepare_ml_data)
    le_circuit = LabelEncoder()
    df["CircuitEncoded"] = le_circuit.fit_transform(df["Circuit"].astype(str))
    feature_cols.append("CircuitEncoded")
    
    le_driver = LabelEncoder()
    df["DriverEncoded"] = le_driver.fit_transform(df["Driver"].astype(str))
    feature_cols.append("DriverEncoded")
    
    X_sequences = []
    y_values = []
    
    # Group by stint
    for (year, race, driver, stint), group in df.groupby(["Year", "RaceName", "Driver", "Stint"]):
        group = group.sort_values("LapNumber")
        
        if len(group) < seq_length + 1:
            continue
        
        stint_features = group[feature_cols].values
        stint_targets = group["DegradationTarget"].values
        
        # Clip extreme targets: >5s above baseline are likely anomalies
        # (safety cars, incidents) that shouldn't be predicted
        stint_targets = np.clip(stint_targets, -1.0, 5.0)
        
        # Handle NaN
        if np.isnan(stint_features).any() or np.isnan(stint_targets).any():
            stint_features = np.nan_to_num(stint_features, nan=0.0)
            stint_targets = np.nan_to_num(stint_targets, nan=0.0)
        
        # Create sliding windows
        for i in range(len(group) - seq_length):
            X_sequences.append(stint_features[i:i + seq_length])
            y_values.append(stint_targets[i + seq_length])
    
    if not X_sequences:
        return None
    
    X = np.array(X_sequences)
    y = np.array(y_values)
    
    # Temporal split: sequences are built from chronologically sorted stints.
    # Fit scaler on training portion only to prevent data leakage.
    n_samples, n_steps, n_features = X.shape
    split_idx = int(n_samples * 0.8)
    train_flat = X[:split_idx].reshape(-1, n_features)
    scaler = StandardScaler()
    scaler.fit(train_flat)
    X_scaled = scaler.transform(X.reshape(-1, n_features)).reshape(n_samples, n_steps, n_features)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "seq_length": seq_length,
    }


def full_preprocessing_pipeline(raw_data_path: str, save_dir: str = None) -> dict:
    """
    Complete preprocessing pipeline.
    
    Returns dict with cleaned data and all prepared datasets.
    """
    logger.info("Starting full preprocessing pipeline...")
    
    # Load data
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded raw data: {df.shape}")
    
    # Clean
    df_clean = clean_data(df)
    
    # Engineer features
    df_features = engineer_features(df_clean)
    
    # Create targets
    df_final = create_degradation_target(df_features)
    
    # Save processed data
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        df_final.to_csv(os.path.join(save_dir, "processed_data.csv"), index=False)
    
    # Prepare for different model types
    ml_data = prepare_ml_data(df_final.copy())
    lstm_data = prepare_lstm_sequences(df_final.copy())
    ts_data = prepare_time_series_data(df_final)
    
    # Temporal split for curve-level models (ARIMA, ETS, Hybrid)
    ts_keys = sorted(ts_data.keys())
    ts_split = int(len(ts_keys) * 0.8)
    ts_train = {k: ts_data[k] for k in ts_keys[:ts_split]}
    ts_test = {k: ts_data[k] for k in ts_keys[ts_split:]}
    
    # Save preprocessing artifacts (scaler, encoders) for inference
    if save_dir:
        import joblib as _jl
        artifacts_dir = os.path.join(save_dir, "..", "models", "preprocessing")
        os.makedirs(artifacts_dir, exist_ok=True)
        _jl.dump(ml_data["scaler"], os.path.join(artifacts_dir, "scaler.pkl"))
        _jl.dump(ml_data["le_circuit"], os.path.join(artifacts_dir, "le_circuit.pkl"))
        _jl.dump(ml_data["le_driver"], os.path.join(artifacts_dir, "le_driver.pkl"))
        _jl.dump(ml_data["feature_cols"], os.path.join(artifacts_dir, "feature_cols.pkl"))
        # Save per-driver, per-compound baseline stats for feature construction
        baseline_stats = _compute_baseline_stats(ml_data["processed_df"])
        _jl.dump(baseline_stats, os.path.join(artifacts_dir, "baseline_stats.pkl"))
        logger.info(f"Saved preprocessing artifacts to {artifacts_dir}")
    
    result = {
        "raw_shape": df.shape,
        "clean_shape": df_clean.shape,
        "final_shape": df_final.shape,
        "processed_df": df_final,
        "ml_data": ml_data,
        "lstm_data": lstm_data,
        "ts_data": ts_data,
        "ts_train": ts_train,
        "ts_test": ts_test,
    }
    
    logger.info("Preprocessing pipeline complete!")
    return result


def _compute_baseline_stats(df: pd.DataFrame) -> dict:
    """
    Compute baseline statistics needed for constructing XGBoost feature vectors
    at inference time. This captures median values per driver/compound/circuit.
    """
    stats = {}
    
    # Global medians for fallback
    stats["global"] = {
        "AvgSpeed": float(df["AvgSpeed"].median()) if "AvgSpeed" in df.columns else 250.0,
        "FuelEffect": float(df["FuelEffect"].median()) if "FuelEffect" in df.columns else 1.5,
    }
    
    # Per-driver + compound medians for DegradationRate and LapTimeDelta
    for (drv, comp), grp in df.groupby(["Driver", "Compound"]):
        key = f"{drv}_{comp}"
        stats[key] = {
            "DegradationRate_median": float(grp["DegradationRate"].median()),
            "LapTimeDelta_median": float(grp["LapTimeDelta"].median()),
            "AvgSpeed": float(grp["AvgSpeed"].median()) if "AvgSpeed" in grp.columns and grp["AvgSpeed"].notna().any() else stats["global"]["AvgSpeed"],
        }
    
    # Per-circuit median speed
    if "Circuit" in df.columns and "AvgSpeed" in df.columns:
        for circuit, grp in df.groupby("Circuit"):
            stats[f"circuit_{circuit}"] = {
                "AvgSpeed": float(grp["AvgSpeed"].median()) if grp["AvgSpeed"].notna().any() else stats["global"]["AvgSpeed"],
            }
    
    return stats


if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    raw_path = os.path.join(DATA_DIR, "f1_lap_data.csv")
    
    if os.path.exists(raw_path):
        result = full_preprocessing_pipeline(raw_path, save_dir=DATA_DIR)
        print(f"Raw: {result['raw_shape']}")
        print(f"Clean: {result['clean_shape']}")
        print(f"Final: {result['final_shape']}")
        print(f"ML train shape: {result['ml_data']['X_train'].shape}")
        if result['lstm_data']:
            print(f"LSTM train shape: {result['lstm_data']['X_train'].shape}")
        print(f"Time series stints: {len(result['ts_data'])}")
    else:
        print(f"Raw data not found at {raw_path}. Run data_fetcher.py first.")
