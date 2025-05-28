# File: train/process_moneypuck_fixed.py
"""
Fixed MoneyPuck data processor with correct URLs
MoneyPuck stores all shots in a single file, not by season
"""

import pandas as pd
# import numpy as np  # Unused import
import requests
import os
from typing import List, Tuple, Optional
import logging
# from datetime import datetime  # Unused import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoneyPuckDataProcessor:
    """Process MoneyPuck data for xG model training"""

    def __init__(self):
        # MoneyPuck hosts data on their site
        # Based on the page, all shots are in a single file
        self.base_url = "https://moneypuck.com/moneypuck/playerData/"
        self.data_dir = "data/moneypuck"
        self.shots_data = None

    def download_all_shots(self):
        """Download all shots data from MoneyPuck"""
        logger.info("Downloading MoneyPuck all shots data...")

        os.makedirs(self.data_dir, exist_ok=True)

        # The actual URL structure based on MoneyPuck's site
        # They provide all shots in one file, not separated by season
        shot_file_url = "https://moneypuck.com/moneypuck/playerData/allShotsData.zip"
        local_path = os.path.join(self.data_dir, "allShotsData.zip")

        if not os.path.exists(local_path):
            try:
                logger.info(f"Downloading from {shot_file_url}")
                response = requests.get(shot_file_url, stream=True)
                response.raise_for_status()

                # Download with progress
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"\rDownloading: {percent:.1f}%", end="", flush=True)

                print()  # New line after download
                logger.info("✓ Downloaded all shots data")

                # Extract ZIP
                import zipfile

                with zipfile.ZipFile(local_path, "r") as zip_ref:
                    zip_ref.extractall(self.data_dir)
                logger.info("✓ Extracted shot data")

            except Exception as e:
                logger.error(f"Failed to download shot data: {e}")
                logger.info("\nTrying alternative approach...")
                self.download_from_kaggle()
        else:
            logger.info(f"→ Shot data already exists at {local_path}")

    def download_from_kaggle(self):
        """Alternative: Guide user to download from Kaggle"""
        logger.info("\n" + "=" * 60)
        logger.info("ALTERNATIVE DATA SOURCE")
        logger.info("=" * 60)
        logger.info("\nMoneyPuck data is also available on Kaggle:")
        logger.info("https://www.kaggle.com/datasets/mexwell/nhl-database")
        logger.info("\nTo download:")
        logger.info("1. Create a free Kaggle account")
        logger.info("2. Download the dataset")
        logger.info("3. Extract shots_YYYY.csv files to data/moneypuck/")
        logger.info("\nOr try downloading directly from MoneyPuck:")
        logger.info("https://moneypuck.com/data.htm")
        logger.info("Look for 'All Shots' download link")

    def load_shots_data(self, seasons: Optional[List[str]] = None) -> pd.DataFrame:
        """Load shots data for specified seasons"""
        all_shots = []

        # First, try to find the all shots file
        all_shots_path = os.path.join(self.data_dir, "shots.csv")
        if not os.path.exists(all_shots_path):
            # Try alternative names
            for filename in ["allShots.csv", "shots_all.csv", "all_shots.csv"]:
                path = os.path.join(self.data_dir, filename)
                if os.path.exists(path):
                    all_shots_path = path
                    break

        if os.path.exists(all_shots_path):
            logger.info(f"Loading all shots from {all_shots_path}")
            df = pd.read_csv(all_shots_path)

            # Filter by seasons if specified
            if seasons and "season" in df.columns:
                df = df[df["season"].astype(str).isin(seasons)]
                logger.info(f"Filtered to seasons: {seasons}")

            self.shots_data = df
            logger.info(f"Loaded {len(df):,} shots")
            return df

        # If no all shots file, try individual season files
        if seasons is None:
            seasons = ["2021", "2022", "2023", "2024"]

        for season in seasons:
            # Try different naming conventions
            for pattern in [f"shots_{season}.csv", f"shots{season}.csv", f"{season}_shots.csv"]:
                csv_path = os.path.join(self.data_dir, pattern)
                if os.path.exists(csv_path):
                    logger.info(f"Loading {pattern}...")
                    df = pd.read_csv(csv_path)
                    df["season"] = season
                    all_shots.append(df)
                    logger.info(f"  Loaded {len(df):,} shots")
                    break

        if all_shots:
            self.shots_data = pd.concat(all_shots, ignore_index=True)
            logger.info(f"\nTotal shots loaded: {len(self.shots_data):,}")
            return self.shots_data
        else:
            logger.error("\nNo shot data found!")
            logger.info("\nPlease download MoneyPuck data manually:")
            logger.info("1. Go to https://moneypuck.com/data.htm")
            logger.info("2. Download the 'All Shots' data")
            logger.info("3. Extract to data/moneypuck/")
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
        logger.info(f"Total shots: {len(self.shots_data):,}")

        # Check for goal column
        if "goal" in self.shots_data.columns:
            logger.info(f"Goals: {self.shots_data['goal'].sum():,} ({self.shots_data['goal'].mean() * 100:.1f}%)")

        # Show all columns
        logger.info(f"\nTotal columns: {len(self.shots_data.columns)}")
        logger.info("\nFirst 20 columns:")
        for i, col in enumerate(self.shots_data.columns[:20]):
            logger.info(f"  {i + 1:2d}. {col}")

        # Key features check
        key_features = {
            "Location": ["xCord", "yCord", "xCordAdjusted", "yCordAdjusted", "shotDistance", "shotAngle"],
            "Shot Type": ["shotType", "shotRush", "shotRebound"],
            "Game State": ["period", "time", "homeTeamGoals", "awayTeamGoals"],
            "Situation": ["homeSkatersOnIce", "awaySkatersOnIce"],
            "Expected Goals": ["xGoal", "goalProbability", "xReboundxGoal"],
            "Shot Speed": ["shotSpeed", "velocity", "speedMPH", "shotVelocity"],
        }

        for category, cols in key_features.items():
            logger.info(f"\n{category} features:")
            found = False
            for col in cols:
                if col in self.shots_data.columns:
                    logger.info(f"  ✓ {col}")
                    found = True
            if not found:
                logger.info("  ❌ None found")

    def prepare_xg_features(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare features for xG model training"""
        logger.info("\nPreparing features for xG model...")

        if self.shots_data is None:
            raise ValueError("No shots data loaded. Run load_shots_data() first.")

        df = self.shots_data.copy()

        # Basic cleaning - remove rows with missing shot type
        if "shotType" in df.columns:
            df = df[df["shotType"].notna()]

        # Feature engineering based on available columns
        features = []

        # 1. Location features
        if "shotDistance" in df.columns:
            features.append("shotDistance")
        elif "distance" in df.columns:
            features.append("distance")

        if "shotAngle" in df.columns:
            features.append("shotAngle")
        elif "angle" in df.columns:
            features.append("angle")

        # 2. Coordinates
        for coord in ["xCord", "yCord", "xCordAdjusted", "yCordAdjusted"]:
            if coord in df.columns:
                features.append(coord)

        # 3. Shot type
        if "shotType" in df.columns:
            # Convert to numeric
            shot_type_map = {
                "WRIST": 0,
                "SLAP": 1,
                "SNAP": 2,
                "BACKHAND": 3,
                "TIP-IN": 4,
                "DEFLECTED": 5,
                "WRAP-AROUND": 6,
            }
            df["shotTypeNumeric"] = df["shotType"].map(shot_type_map).fillna(7)
            features.append("shotTypeNumeric")

        # 4. Game situation
        return df[features], df['goal'], df
