"""
F1 Data Fetcher using FastF1 API
Fetches historical race data including lap times, tire information, pit stops, and weather.
"""

import fastf1
import pandas as pd
import numpy as np
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable FastF1 cache
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Races we want to fetch data for (recent seasons for good data quality)
# We focus on 2022-2025 for modern tire compounds and regulations
SEASONS_AND_RACES = {
    2023: None,  # None = all races in season
    2024: None,
}

# Known circuits and their characteristics
CIRCUIT_INFO = {
    "Albert Park": {"type": "street-like", "abrasiveness": "medium", "corners": 14},
    "Bahrain": {"type": "permanent", "abrasiveness": "high", "corners": 15},
    "Jeddah": {"type": "street", "abrasiveness": "low", "corners": 27},
    "Melbourne": {"type": "street-like", "abrasiveness": "medium", "corners": 14},
    "Suzuka": {"type": "permanent", "abrasiveness": "high", "corners": 18},
    "Monza": {"type": "permanent", "abrasiveness": "low", "corners": 11},
    "Silverstone": {"type": "permanent", "abrasiveness": "high", "corners": 18},
    "Spa": {"type": "permanent", "abrasiveness": "medium", "corners": 19},
    "Monaco": {"type": "street", "abrasiveness": "low", "corners": 19},
    "Singapore": {"type": "street", "abrasiveness": "high", "corners": 23},
    "Hungaroring": {"type": "permanent", "abrasiveness": "medium", "corners": 14},
    "Zandvoort": {"type": "permanent", "abrasiveness": "medium", "corners": 14},
    "Interlagos": {"type": "permanent", "abrasiveness": "high", "corners": 15},
    "COTA": {"type": "permanent", "abrasiveness": "high", "corners": 20},
    "Yas Marina": {"type": "permanent", "abrasiveness": "medium", "corners": 16},
    "Red Bull Ring": {"type": "permanent", "abrasiveness": "medium", "corners": 10},
    "Catalunya": {"type": "permanent", "abrasiveness": "high", "corners": 16},
    "Imola": {"type": "permanent", "abrasiveness": "medium", "corners": 19},
    "Miami": {"type": "street-like", "abrasiveness": "medium", "corners": 19},
    "Las Vegas": {"type": "street", "abrasiveness": "low", "corners": 17},
    "Qatar": {"type": "permanent", "abrasiveness": "high", "corners": 16},
    "Mexico City": {"type": "permanent", "abrasiveness": "medium", "corners": 17},
    "Shanghai": {"type": "permanent", "abrasiveness": "medium", "corners": 16},
}


def fetch_race_data(year: int, race_number: int = None, race_name: str = None) -> dict:
    """
    Fetch complete race data for a specific race.
    
    Args:
        year: Season year
        race_number: Race number in season (1-indexed)
        race_name: Race name (e.g., 'Australian Grand Prix')
    
    Returns:
        Dictionary with lap data, pit stops, and session info
    """
    try:
        if race_name:
            session = fastf1.get_session(year, race_name, "R")
        else:
            session = fastf1.get_session(year, race_number, "R")
        
        session.load()
        logger.info(f"Loaded: {year} {session.event['EventName']}")
        
        laps = session.laps
        if laps.empty:
            logger.warning(f"No lap data for {year} race {race_number or race_name}")
            return None
        
        result = {
            "year": year,
            "race_name": session.event["EventName"],
            "circuit": session.event["Location"],
            "laps": laps,
            "results": session.results,
            "weather": getattr(session, "weather_data", None),
            "event": session.event,
        }
        return result
        
    except Exception as e:
        logger.error(f"Error fetching {year} race {race_number or race_name}: {e}")
        return None


def fetch_season_data(year: int, max_races: int = None) -> list:
    """Fetch data for all races in a season."""
    schedule = fastf1.get_event_schedule(year)
    race_events = schedule[schedule["EventFormat"] != "testing"]
    
    all_races = []
    count = 0
    
    for _, event in race_events.iterrows():
        if max_races and count >= max_races:
            break
        
        race_data = fetch_race_data(year, race_name=event["EventName"])
        if race_data:
            all_races.append(race_data)
            count += 1
    
    return all_races


def extract_lap_features(race_data: dict) -> pd.DataFrame:
    """
    Extract relevant features from raw lap data for modeling.
    
    Features extracted:
    - Driver, Team, Circuit
    - Lap number, Lap time (seconds)
    - Tire compound, Tire life (stint length)
    - Sector times
    - Pit in/out lap flags
    - Track status (safety car, VSC, etc.)
    """
    if race_data is None:
        return pd.DataFrame()
    
    laps = race_data["laps"].copy()
    
    if laps.empty:
        return pd.DataFrame()
    
    # Convert timedeltas to seconds
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        if col in laps.columns:
            laps[f"{col}Sec"] = laps[col].dt.total_seconds()
    
    # Build features DataFrame
    n = len(laps)
    features = pd.DataFrame()
    features["Driver"] = laps["Driver"].values
    features["DriverNumber"] = laps["DriverNumber"].values
    features["Team"] = laps["Team"].values
    features["Year"] = [race_data["year"]] * n
    features["RaceName"] = [race_data["race_name"]] * n
    features["Circuit"] = [race_data["circuit"]] * n
    features["LapNumber"] = laps["LapNumber"].values
    features["LapTimeSec"] = laps["LapTimeSec"].values if "LapTimeSec" in laps.columns else np.nan
    features["Sector1TimeSec"] = laps["Sector1TimeSec"].values if "Sector1TimeSec" in laps.columns else np.nan
    features["Sector2TimeSec"] = laps["Sector2TimeSec"].values if "Sector2TimeSec" in laps.columns else np.nan
    features["Sector3TimeSec"] = laps["Sector3TimeSec"].values if "Sector3TimeSec" in laps.columns else np.nan
    features["Compound"] = laps["Compound"].values if "Compound" in laps.columns else "UNKNOWN"
    features["TyreLife"] = laps["TyreLife"].values if "TyreLife" in laps.columns else 0
    features["Stint"] = laps["Stint"].values if "Stint" in laps.columns else 1
    features["FreshTyre"] = laps["FreshTyre"].values if "FreshTyre" in laps.columns else True
    features["TrackStatus"] = laps["TrackStatus"].values if "TrackStatus" in laps.columns else "1"
    features["IsPersonalBest"] = laps["IsPersonalBest"].values if "IsPersonalBest" in laps.columns else False
    
    # Pit stop flags
    features["PitInTime"] = (~laps["PitInTime"].isna()).values if "PitInTime" in laps.columns else False
    features["PitOutTime"] = (~laps["PitOutTime"].isna()).values if "PitOutTime" in laps.columns else False
    
    # Position
    features["Position"] = laps["Position"].values if "Position" in laps.columns else np.nan
    
    # SpeedI1, SpeedI2, SpeedFL, SpeedST (speed trap data)
    for speed_col in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]:
        if speed_col in laps.columns:
            features[speed_col] = laps[speed_col].values
        else:
            features[speed_col] = np.nan
    
    return features


def build_dataset(seasons: dict = None, save_path: str = None) -> pd.DataFrame:
    """
    Build the complete dataset from multiple seasons.
    
    Args:
        seasons: Dict of {year: max_races} (None for all races)
        save_path: Path to save the combined CSV
    
    Returns:
        Combined DataFrame with all lap features
    """
    if seasons is None:
        seasons = SEASONS_AND_RACES
    
    all_features = []
    
    for year, max_races in seasons.items():
        logger.info(f"Fetching {year} season data...")
        races = fetch_season_data(year, max_races=max_races)
        
        for race_data in races:
            features = extract_lap_features(race_data)
            if not features.empty:
                all_features.append(features)
                logger.info(f"  Extracted {len(features)} laps from {race_data['race_name']}")
    
    if not all_features:
        logger.error("No data extracted!")
        return pd.DataFrame()
    
    combined = pd.concat(all_features, ignore_index=True)
    logger.info(f"Total dataset: {len(combined)} laps from {combined['RaceName'].nunique()} races")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        combined.to_csv(save_path, index=False)
        logger.info(f"Saved to {save_path}")
    
    return combined


def fetch_specific_race(year: int, race_name: str) -> pd.DataFrame:
    """Fetch and process data for a specific race (used for predictions)."""
    race_data = fetch_race_data(year, race_name=race_name)
    if race_data:
        return extract_lap_features(race_data)
    return pd.DataFrame()


if __name__ == "__main__":
    # Fetch and build the dataset
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    save_path = os.path.join(DATA_DIR, "f1_lap_data.csv")
    
    df = build_dataset(save_path=save_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:")
    print(df.head(10))
    print(f"\nCompound distribution:")
    print(df["Compound"].value_counts())
    print(f"\nCircuits: {df['Circuit'].unique()}")
