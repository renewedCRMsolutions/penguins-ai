# File: train/process_moneypuck_data.py
"""
Process MoneyPuck NHL data for xG model training
MoneyPuck provides comprehensive shot data from 2007-2024
"""

import pandas as pd
import numpy as np
import requests
import zipfile
import os
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoneyPuckDataProcessor:
    """Process MoneyPuck data for xG model training"""

    def __init__(self):
        self.base_url = "https://moneypuck.com/moneypuck/playerData/"
        self.data_dir = "data/moneypuck"
        self.shots_data = None
        self.player_data = None

    def download_moneypuck_data(self, seasons: Optional[List[str]] = None):
        """Download MoneyPuck data for specified seasons"""
        if seasons is None:
            # Default to recent seasons
            seasons = ["2021", "2022", "2023", "2024"]

        os.makedirs(self.data_dir, exist_ok=True)

        for season in seasons:
            self._download_season(season)

    def _download_season(self, season: str):
        """Download data for a single season"""
        logger.info(f"Downloading MoneyPuck data for {season} season...")

        # MoneyPuck file structure
        files = {"shots": f"shots_{season}.zip", "players": f"skaters_{season}.zip", "goalies": f"goalies_{season}.zip"}

        for data_type, filename in files.items():
            url = f"{self.base_url}{filename}"
            local_path = os.path.join(self.data_dir, filename)

            # Download if not exists
            if not os.path.exists(local_path):
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()

                    with open(local_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    logger.info(f"  ✓ Downloaded {filename}")

                    # Extract ZIP
                    with zipfile.ZipFile(local_path, "r") as zip_ref:
                        zip_ref.extractall(self.data_dir)
                    logger.info(f"  ✓ Extracted {filename}")

                except Exception as e:
                    logger.error(f"  ✗ Failed to download {filename}: {e}")
            else:
                logger.info(f"  → {filename} already exists")

    def load_shots_data(self, seasons: Optional[List[str]] = None) -> pd.DataFrame:
        """Load and combine shots data from multiple seasons"""
        if seasons is None:
            seasons = ["2021", "2022", "2023", "2024"]

        all_shots = []

        for season in seasons:
            csv_path = os.path.join(self.data_dir, f"shots_{season}.csv")
            if os.path.exists(csv_path):
                logger.info(f"Loading shots data for {season}...")
                df = pd.read_csv(csv_path)
                df["season"] = season
                all_shots.append(df)
                logger.info(f"  Loaded {len(df):,} shots")
            else:
                logger.warning(f"  No data found for {season}")

        if all_shots:
            self.shots_data = pd.concat(all_shots, ignore_index=True)
            logger.info(f"\nTotal shots loaded: {len(self.shots_data):,}")
            return self.shots_data
        else:
            raise ValueError("No shots data found")

    def explore_shot_features(self):
        """Explore available features in MoneyPuck shots data"""
        if self.shots_data is None:
            raise ValueError("No shots data loaded. Run load_shots_data() first.")

        logger.info("\n" + "=" * 60)
        logger.info("MONEYPUCK SHOT DATA ANALYSIS")
        logger.info("=" * 60)

        # Basic info
        logger.info(f"\nDataset shape: {self.shots_data.shape}")
        logger.info(f"Seasons: {sorted(self.shots_data['season'].unique())}")
        logger.info(f"Total shots: {len(self.shots_data):,}")
        logger.info(f"Goals: {self.shots_data['goal'].sum():,} ({self.shots_data['goal'].mean() * 100:.1f}%)")

        # Key features
        logger.info("\nKey shot features available:")

        # Location features
        location_cols = ["xCordAdjusted", "yCordAdjusted", "shotDistance", "shotAngleAdjusted"]
        logger.info("\n1. Location features:")
        for col in location_cols:
            if col in self.shots_data.columns:
                logger.info(f"  ✓ {col}")

        # Shot type features
        logger.info("\n2. Shot type features:")
        if "shotType" in self.shots_data.columns:
            shot_types = self.shots_data["shotType"].value_counts()
            for stype, count in shot_types.head().items():
                logger.info(f"  • {stype}: {count:,} shots")

        # Game situation
        situation_cols = ["home", "isPlayoffGame", "homeSkatersOnIce", "awaySkatersOnIce"]
        logger.info("\n3. Situation features:")
        for col in situation_cols:
            if col in self.shots_data.columns:
                logger.info(f"  ✓ {col}")

        # xG features
        xg_cols = ["xGoal", "goalProbability", "xReboundxGoal", "xPlayContinuedInZone"]
        logger.info("\n4. Expected goals features:")
        for col in xg_cols:
            if col in self.shots_data.columns:
                logger.info(f"  ✓ {col}: mean={self.shots_data[col].mean():.3f}")

        # Shot quality features
        quality_cols = ["shotRush", "shotRebound", "lastEventCategory", "speedFromLastEvent"]
        logger.info("\n5. Shot quality features:")
        for col in quality_cols:
            if col in self.shots_data.columns:
                logger.info(f"  ✓ {col}")

        # Check for shot speed/velocity
        logger.info("\n6. Checking for shot speed data...")
        speed_cols = [
            col
            for col in self.shots_data.columns
            if any(term in col.lower() for term in ["speed", "velocity", "mph", "kph"])
        ]

        if speed_cols:
            logger.info("  🎯 SHOT SPEED COLUMNS FOUND:")
            for col in speed_cols:
                non_null = self.shots_data[col].notna().sum()
                logger.info(f"    {col}: {non_null:,} non-null values")
        else:
            logger.info("  ❌ No direct shot speed columns found")

    def prepare_xg_features(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare features for xG model training"""
        logger.info("\nPreparing features for xG model...")

        if self.shots_data is None:
            raise ValueError("No shots data loaded. Run load_shots_data() first.")

        df = self.shots_data.copy()

        # Basic cleaning
        df = df[df["shotType"].notna()]  # Remove shots without type

        # Feature engineering
        # 1. Distance and angle (already calculated by MoneyPuck)
        df["distance"] = df["shotDistance"]
        df["angle"] = df["shotAngleAdjusted"].abs()

        # 2. Shot type quality scores
        shot_type_map = {
            "TIP-IN": 0.33,
            "DEFLECTED": 0.25,
            "WRAP-AROUND": 0.20,
            "BACK": 0.18,
            "SLAP": 0.15,
            "SNAP": 0.14,
            "WRIST": 0.12,
            "BAT": 0.10,
        }
        df["shot_quality"] = df["shotType"].map(shot_type_map).fillna(0.10)

        # 3. Game situation
        df["strength"] = df.apply(
            lambda x: self._determine_strength(x["homeSkatersOnIce"], x["awaySkatersOnIce"], x["home"]), axis=1
        )
        df["is_powerplay"] = (df["strength"] == "PP").astype(int)
        df["is_shorthanded"] = (df["strength"] == "SH").astype(int)

        # 4. Pre-shot events (if available)
        if "shotRush" in df.columns:
            df["is_rush"] = df["shotRush"].astype(int)
        else:
            df["is_rush"] = 0

        if "shotRebound" in df.columns:
            df["is_rebound"] = df["shotRebound"].astype(int)
        else:
            df["is_rebound"] = 0

        # 5. Time features
        df["period_seconds"] = df["time"] % 1200  # Seconds into period
        df["game_seconds"] = df["time"]  # Total game seconds

        # 6. Score state
        df["score_diff"] = df["homeTeamGoals"] - df["awayTeamGoals"]
        if "home" in df.columns:
            df["score_diff"] = df.apply(lambda x: x["score_diff"] if x["home"] == 1 else -x["score_diff"], axis=1)

        # 7. MoneyPuck's own features
        moneypuck_features = ["xGoal", "goalProbability", "xReboundxGoal", "xPlayContinuedInZone", "xPlayStopped"]

        # Select features
        feature_cols = [
            "distance",
            "angle",
            "shot_quality",
            "is_powerplay",
            "is_shorthanded",
            "is_rush",
            "is_rebound",
            "period_seconds",
            "game_seconds",
            "score_diff",
            "homeSkatersOnIce",
            "awaySkatersOnIce",
        ]

        # Add MoneyPuck features if available
        for feat in moneypuck_features:
            if feat in df.columns:
                feature_cols.append(feat)

        # Add location coordinates
        if "xCordAdjusted" in df.columns:
            feature_cols.extend(["xCordAdjusted", "yCordAdjusted"])

        # Create feature matrix
        X = df[feature_cols].fillna(0)
        y = df["goal"]

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Features used: {len(feature_cols)}")
        logger.info(f"Goals: {y.sum():,} ({y.mean() * 100:.1f}%)")

        return X, y, df

    def _determine_strength(self, home_skaters: int, away_skaters: int, is_home: bool) -> str:
        """Determine game strength"""
        if home_skaters == away_skaters:
            return "EV"
        elif (is_home and home_skaters > away_skaters) or (not is_home and away_skaters > home_skaters):
            return "PP"
        else:
            return "SH"

    def compare_with_moneypuck_xg(self, predictions: np.ndarray):
        """Compare our predictions with MoneyPuck's xGoal"""
        if self.shots_data is None or "xGoal" not in self.shots_data.columns:
            logger.warning("No xGoal column in MoneyPuck data for comparison")
            return

        mp_xg = self.shots_data["xGoal"].values[: len(predictions)]

        # Calculate correlation
        correlation = np.corrcoef(predictions, mp_xg)[0, 1]

        # Mean absolute error
        mae = np.mean(np.abs(predictions - mp_xg))

        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON WITH MONEYPUCK xG")
        logger.info("=" * 60)
        logger.info(f"Correlation: {correlation:.3f}")
        logger.info(f"Mean Absolute Error: {mae:.3f}")
        logger.info(f"Our mean xG: {predictions.mean():.3f}")
        logger.info(f"MoneyPuck mean xG: {mp_xg.mean():.3f}")


def main():
    """Process MoneyPuck data"""
    processor = MoneyPuckDataProcessor()

    # Download data
    seasons = ["2021", "2022", "2023", "2024"]
    processor.download_moneypuck_data(seasons)

    # Load shots data
    processor.load_shots_data(seasons)

    # Explore features
    processor.explore_shot_features()

    # Prepare for training
    X, y, full_df = processor.prepare_xg_features()

    # Save processed data
    output_path = "data/moneypuck/processed_shots.csv"
    full_df.to_csv(output_path, index=False)
    logger.info(f"\nProcessed data saved to {output_path}")


if __name__ == "__main__":
    main()
