"""
Main Pipeline Script
Runs the complete F1 Pit Stop Prediction pipeline:
1. Fetch data from FastF1
2. Preprocess data
3. Train all 5 models
4. Evaluate and compare
5. Save results and best model
"""

import os
import sys
import logging
import json
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ]
)
logger = logging.getLogger("pipeline")

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR = os.path.join(PROJECT_DIR, "plots")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def step1_fetch_data():
    """Step 1: Fetch F1 data using FastF1."""
    logger.info("=" * 70)
    logger.info("STEP 1: FETCHING F1 DATA")
    logger.info("=" * 70)
    
    from src.data_fetcher import build_dataset
    
    save_path = os.path.join(DATA_DIR, "f1_lap_data.csv")
    
    if os.path.exists(save_path):
        import pandas as pd
        df = pd.read_csv(save_path)
        logger.info(f"Using existing data: {df.shape[0]} laps from {save_path}")
        return df
    
    # Fetch 2023-2024 seasons
    seasons = {
        2023: None,  # All races
        2024: None,  # All races
    }
    
    df = build_dataset(seasons=seasons, save_path=save_path)
    logger.info(f"Fetched {df.shape[0]} laps from {df['RaceName'].nunique()} races")
    return df


def step2_preprocess(raw_path: str = None):
    """Step 2: Preprocess data."""
    logger.info("=" * 70)
    logger.info("STEP 2: PREPROCESSING DATA")
    logger.info("=" * 70)
    
    from src.preprocessing import full_preprocessing_pipeline
    
    if raw_path is None:
        raw_path = os.path.join(DATA_DIR, "f1_lap_data.csv")
    
    result = full_preprocessing_pipeline(raw_path, save_dir=DATA_DIR)
    
    logger.info(f"Preprocessing complete:")
    logger.info(f"  Raw: {result['raw_shape']}")
    logger.info(f"  Clean: {result['clean_shape']}")
    logger.info(f"  Final: {result['final_shape']}")
    logger.info(f"  ML train: {result['ml_data']['X_train'].shape}")
    if result['lstm_data']:
        logger.info(f"  LSTM train: {result['lstm_data']['X_train'].shape}")
    logger.info(f"  TS stints: {len(result['ts_data'])} (train: {len(result['ts_train'])}, test: {len(result['ts_test'])})")
    
    return result


def step3_train_and_compare(preprocess_result: dict):
    """Step 3: Train all models and compare."""
    logger.info("=" * 70)
    logger.info("STEP 3: TRAINING AND COMPARING MODELS")
    logger.info("=" * 70)
    
    from src.model_comparison import ModelComparison
    
    comparison = ModelComparison(plots_dir=PLOTS_DIR, models_dir=MODELS_DIR)
    
    ml_data = preprocess_result["ml_data"]
    lstm_data = preprocess_result["lstm_data"]
    ts_train = preprocess_result["ts_train"]
    ts_test = preprocess_result["ts_test"]
    
    # Train all models (curve models use ts_train only)
    comparison.train_all(ml_data, lstm_data, ts_train)
    
    # Evaluate all models (curve models use held-out ts_test)
    comparison.evaluate_all(ml_data, lstm_data, ts_test)
    
    # Generate comparison outputs
    comparison.plot_comparison()
    comparison.plot_feature_importance()
    
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        comparison.plot_degradation_curves(ts_test, compound)
    
    # Save degradation prediction data for interactive frontend
    comparison.save_degradation_data()
    
    # Save results
    comparison.save_results()
    
    # Generate report
    report = comparison.generate_full_report()
    logger.info("\n" + report)
    
    return comparison


def step4_generate_strategy(comparison: "ModelComparison"):
    """Step 4: Generate example strategies."""
    logger.info("=" * 70)
    logger.info("STEP 4: GENERATING PIT STOP STRATEGIES")
    logger.info("=" * 70)
    
    from src.strategy import PitStopStrategy
    
    engine = PitStopStrategy(models_dir=MODELS_DIR)
    engine.load_models(comparison)
    
    # Generate strategy for Australian GP (March 8 target!)
    logger.info("\n--- Australian GP Strategy Preview ---")
    strategies = engine.compare_strategies(
        circuit="Melbourne",
        total_laps=58,
        driver="Example Driver"
    )
    
    for strat in strategies:
        text = engine.get_recommendation_text(strat)
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting on {strat['stints'][0]['compound']} (Rank #{strat['rank']})")
        logger.info(text)
    
    return engine


def main():
    """Run the complete pipeline."""
    start_time = time.time()
    
    logger.info("🏎️ F1 PIT STOP PREDICTION PIPELINE")
    logger.info("=" * 70)
    
    # Step 1: Fetch data
    step1_fetch_data()
    
    # Step 2: Preprocess
    preprocess_result = step2_preprocess()
    
    # Step 3: Train and compare models
    comparison = step3_train_and_compare(preprocess_result)
    
    # Step 4: Generate strategies
    engine = step4_generate_strategy(comparison)
    
    elapsed = time.time() - start_time
    logger.info(f"\nPipeline complete in {elapsed:.1f} seconds")
    logger.info(f"Best model: {comparison.best_model_name}")
    logger.info(f"Results saved to {PLOTS_DIR}/ and {MODELS_DIR}/")
    logger.info("Run 'streamlit run app.py' to launch the frontend!")
    
    return comparison, engine


if __name__ == "__main__":
    main()
