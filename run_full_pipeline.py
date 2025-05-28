# File: run_full_pipeline.py
"""
Complete pipeline to collect NHL data, engineer features, and train advanced xG model
Target: Beat MoneyPuck's 0.78 AUC
"""

import sys

# import os  # Unused import
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

# Add project paths
sys.path.append(str(Path(__file__).parent))

# Import our modules
from data.collect_live_nhl_data import NHLDataCollector
from data.goalie_features import GoalieFeatureEngineer
from train.train_xg_final import AdvancedXGModel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        "nhlpy": "nhl-api-py",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "torch": "torch",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        "numpy": "numpy",
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.error(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def run_data_collection(start_date: str, end_date: str, output_dir: str = "data/nhl/enhanced"):
    """Step 1: Collect NHL data with enhanced features"""

    logger.info("=" * 60)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 60)

    collector = NHLDataCollector(data_dir=output_dir)

    # Collect game data
    logger.info(f"Collecting games from {start_date} to {end_date}")
    shots = collector.collect_date_range(start_date, end_date, game_types=[2])

    if not shots:
        logger.error("No shots collected! Check date range and API connection.")
        return False

    logger.info(f"Collected {len(shots)} shots")

    # Collect goalie season stats
    season = "20242025"  # Adjust based on dates
    collector.collect_goalie_season_stats(season)

    return True


def run_feature_engineering(input_dir: str = "data/nhl/enhanced"):
    """Step 2: Engineer advanced features, especially goalie-focused"""

    logger.info("=" * 60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 60)

    engineer = GoalieFeatureEngineer(data_dir=input_dir)

    # Find the most recent data files
    data_path = Path(input_dir)
    shot_files = list(data_path.glob("enhanced_shots_*.csv"))
    goalie_files = list(data_path.glob("goalie_stats_*.csv"))
    game_files = list(data_path.glob("games_*.csv"))

    if not shot_files:
        logger.error("No shot data files found!")
        return False

    # Use most recent files
    shots_df = pd.read_csv(sorted(shot_files)[-1])
    goalie_stats_df = pd.read_csv(sorted(goalie_files)[-1]) if goalie_files else pd.DataFrame()
    games_df = pd.read_csv(sorted(game_files)[-1]) if game_files else pd.DataFrame()

    logger.info(f"Loaded {len(shots_df)} shots for feature engineering")

    # Engineer features
    enhanced_shots = engineer.engineer_all_features(shots_df, goalie_stats_df, games_df)

    # Save enhanced dataset
    output_name = f"enhanced_full_{datetime.now().strftime('%Y%m%d')}"
    engineer.save_engineered_features(enhanced_shots, output_name)

    # Print feature summary
    goalie_features = [col for col in enhanced_shots.columns if "goalie_" in col]
    logger.info(f"Total features: {len(enhanced_shots.columns)}")
    logger.info(f"Goalie-specific features: {len(goalie_features)}")
    logger.info(f"Sample goalie features: {goalie_features[:5]}")

    return output_name


def run_model_training(data_file: str, model_dir: str = "models/production"):
    """Step 3: Train advanced ensemble model"""

    logger.info("=" * 60)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("=" * 60)

    model = AdvancedXGModel(model_dir=model_dir)

    # Find the data file
    data_path = Path("data/nhl/enhanced") / f"{data_file}_with_goalie_features.csv"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return None

    logger.info(f"Training on: {data_path}")

    # Train the model
    final_auc = model.train_full_pipeline(str(data_path))

    return final_auc


def compare_with_moneypuck(model_auc: float):
    """Compare our model with MoneyPuck"""

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)

    moneypuck_auc = 0.7781

    logger.info(f"Our Model AUC: {model_auc:.4f}")
    logger.info(f"MoneyPuck AUC: {moneypuck_auc:.4f}")
    logger.info(f"Difference: {model_auc - moneypuck_auc:+.4f}")

    if model_auc > moneypuck_auc:
        logger.info("🎉 WE BEAT MONEYPUCK! 🎉")
        logger.info("Key advantages:")
        logger.info("- Real-time NHL API data with pre-shot context")
        logger.info("- Advanced goalie workload and fatigue features")
        logger.info("- Shooter-goalie matchup history")
        logger.info("- Neural network meta-learner ensemble")
        logger.info("- Proper probability calibration")
    else:
        logger.info("Close but not quite there yet!")
        logger.info(f"Need {moneypuck_auc - model_auc:.4f} more AUC points")

    return model_auc > moneypuck_auc


def run_quick_test():
    """Quick test with existing data"""

    logger.info("Running quick test with synthetic data...")

    # Create minimal test data
    n_samples = 10000
    np.random.seed(42)

    test_df = pd.DataFrame(
        {
            # Basic features
            "shot_distance": np.random.uniform(5, 60, n_samples),
            "shot_angle": np.random.uniform(0, 90, n_samples),
            "x_coord": np.random.uniform(-42, 42, n_samples),
            "y_coord": np.random.uniform(-100, 100, n_samples),
            "danger_zone": np.random.choice(["high", "medium", "low"], n_samples),
            "shot_type": np.random.choice(["Wrist", "Slap", "Snap", "Backhand"], n_samples),
            "is_rush": np.random.choice([0, 1], n_samples),
            "is_rebound": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "is_one_timer": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "traffic_score": np.random.randint(0, 6, n_samples),
            "passes_before_shot": np.random.randint(0, 5, n_samples),
            "time_since_zone_entry": np.random.uniform(0, 30, n_samples),
            "time_since_faceoff": np.random.uniform(0, 60, n_samples),
            # Goalie features
            "goalie_savePctg": np.random.uniform(0.88, 0.94, n_samples),
            "goalie_quality_rating": np.random.uniform(0.5, 1.0, n_samples),
            "goalie_shots_faced_period": np.random.randint(0, 20, n_samples),
            "goalie_shots_faced_last_5min": np.random.randint(0, 10, n_samples),
            "goalie_save_pct_last_10_shots": np.random.uniform(0.7, 1.0, n_samples),
            "goalie_save_pct_period": np.random.uniform(0.8, 1.0, n_samples),
            "time_since_last_shot": np.random.uniform(0, 300, n_samples),
            "consecutive_saves": np.random.randint(0, 20, n_samples),
            "goalie_high_danger_save_pct": np.random.uniform(0.7, 0.9, n_samples),
            "goalie_fatigue_score": np.random.uniform(0, 1, n_samples),
            "goalie_cold_start": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "goalie_recent_form_trend": np.random.choice([-1, 0, 1], n_samples),
            "shooter_vs_goalie_career_shooting_pct": np.random.uniform(0, 0.2, n_samples),
            # Game state
            "period": np.random.choice([1, 2, 3], n_samples),
            "time_remaining": np.random.uniform(0, 3600, n_samples),
            "score_differential": np.random.randint(-5, 5, n_samples),
            "is_home_team": np.random.choice([0, 1], n_samples),
            "is_power_play": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            "momentum_score": np.random.uniform(-5, 5, n_samples),
            "zone_time": np.random.uniform(0, 120, n_samples),
            "shot_attempts_sequence": np.random.randint(0, 5, n_samples),
            # Target (with some logic)
            "is_goal": 0,
        }
    )

    # Create somewhat realistic goal probability
    goal_prob = 0.05  # Base
    goal_prob += (60 - test_df["shot_distance"]) / 1000  # Closer = better
    goal_prob += (1 - test_df["goalie_quality_rating"]) * 0.1  # Worse goalie = more goals
    goal_prob += test_df["is_rebound"] * 0.15  # Rebounds score more
    goal_prob += test_df["is_power_play"] * 0.05  # Power play advantage
    goal_prob += test_df["goalie_fatigue_score"] * 0.05  # Tired goalies

    test_df["is_goal"] = (np.random.random(n_samples) < goal_prob).astype(int)

    # Save test data
    test_path = "data/test_data.csv"
    test_df.to_csv(test_path, index=False)

    # Train model
    model = AdvancedXGModel(model_dir="models/test")
    auc = model.train_full_pipeline(test_path)

    return auc


def main():
    """Run the complete pipeline"""

    logger.info("=" * 60)
    logger.info("NHL xG MODEL PIPELINE - BEAT MONEYPUCK EDITION")
    logger.info("=" * 60)

    # Check dependencies
    if not check_dependencies():
        return

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "train-only", "test"], default="full", help="Run mode")
    parser.add_argument("--days", type=int, default=30, help="Days of data to collect")
    parser.add_argument("--data-file", type=str, help="Existing data file for train-only mode")
    args = parser.parse_args()

    if args.mode == "test":
        # Quick test mode
        auc = run_quick_test()
        compare_with_moneypuck(float(auc))
        return

    if args.mode == "full":
        # Full pipeline
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=args.days)

        # Step 1: Collect data
        success = run_data_collection(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        if not success:
            logger.error("Data collection failed!")
            return

        # Step 2: Engineer features
        output_name = run_feature_engineering()

        if not output_name:
            logger.error("Feature engineering failed!")
            return

        # Step 3: Train model
        auc = run_model_training(output_name)

    elif args.mode == "train-only":
        # Train on existing data
        if not args.data_file:
            logger.error("Please provide --data-file for train-only mode")
            return

        auc = run_model_training(args.data_file)

    # Compare results
    if auc:
        compare_with_moneypuck(float(auc))

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
