# File: train/load_moneypuck_data.py
"""
Load and process your existing MoneyPuck data
Works with the skaters.csv file you already downloaded
"""

import pandas as pd
import numpy as np
import os
import logging
# from typing import Tuple  # Unused import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoneyPuckDataLoader:
    """Load existing MoneyPuck data from your data folder"""

    def __init__(self):
        self.data_dir = "data"
        self.skaters_data = None
        self.shots_data = None

    def explore_available_files(self):
        """Check what MoneyPuck files are available"""
        logger.info("🔍 Checking for MoneyPuck data files...")
        logger.info(f"Looking in: {os.path.abspath(self.data_dir)}")

        # Look for CSV files
        csv_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith(".csv"):
                csv_files.append(file)
                file_path = os.path.join(self.data_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"  ✓ Found: {file} ({size_mb:.1f} MB)")

        return csv_files

    def load_skaters_data(self) -> pd.DataFrame:
        """Load the skaters.csv file"""
        skaters_path = os.path.join(self.data_dir, "skaters.csv")

        if not os.path.exists(skaters_path):
            raise FileNotFoundError(f"Skaters file not found at {skaters_path}")

        logger.info(f"\nLoading skaters data from {skaters_path}...")
        self.skaters_data = pd.read_csv(skaters_path)

        logger.info(f"✓ Loaded {len(self.skaters_data):,} player records")
        logger.info(f"  Columns: {self.skaters_data.shape[1]}")
        seasons = sorted(self.skaters_data['season'].unique()) if 'season' in self.skaters_data.columns else 'Unknown'
        logger.info(f"  Seasons: {seasons}")

        return self.skaters_data

    def check_for_shot_data(self):
        """Check if we have shot-level data"""
        logger.info("\n🎯 Checking for shot data...")

        # Common shot file names
        shot_files = ["shots.csv", "shots_2024.csv", "shots_2025.csv", "allShots.csv", "shot_data.csv"]

        for filename in shot_files:
            path = os.path.join(self.data_dir, filename)
            if os.path.exists(path):
                logger.info(f"  ✓ Found shot data: {filename}")
                return path

        logger.info("  ❌ No shot-level data found")
        logger.info("\n  The skaters.csv file contains aggregated player stats, not individual shots.")
        logger.info("  For xG modeling, we need shot-level data.")
        return None

    def analyze_skaters_features(self):
        """Analyze what's in the skaters data"""
        if self.skaters_data is None:
            self.load_skaters_data()

        logger.info("\n" + "=" * 60)
        logger.info("SKATERS DATA ANALYSIS")
        logger.info("=" * 60)

        # Basic info
        if self.skaters_data is not None:
            logger.info(f"\nDataset shape: {self.skaters_data.shape}")
        else:
            logger.info("\nNo skaters data loaded")
            return

        # Show columns grouped by category
        columns = list(self.skaters_data.columns) if self.skaters_data is not None else []

        # Group columns by prefix
        column_groups = {
            "Basic Info": ["playerId", "name", "team", "position"],
            "Games & Time": ["games_played", "icetime", "shifts"],
            "Goals & Assists": [col for col in columns if "goal" in col.lower() or "assist" in col.lower()],
            "Shots": [col for col in columns if "shot" in col.lower()],
            "Expected Goals": [col for col in columns if "xgoal" in col.lower() or "xg" in col.lower()],
            "On-Ice Metrics": [col for col in columns if col.startswith("OnIce_")],
            "Individual Metrics": [col for col in columns if col.startswith("I_F_")],
        }

        for group, cols in column_groups.items():
            found_cols = [col for col in cols if col in columns]
            if found_cols:
                logger.info(f"\n{group}:")
                for col in found_cols[:10]:  # Show first 10
                    logger.info(f"  • {col}")
                if len(found_cols) > 10:
                    logger.info(f"  ... and {len(found_cols) - 10} more")

        # Check for shot metrics
        shot_columns = [col for col in columns if any(term in col.lower() for term in ["shot", "goal", "xgoal"])]
        logger.info(f"\n📊 Shot-related columns: {len(shot_columns)}")

        # Sample data
        if self.skaters_data is not None and "I_F_xGoals" in self.skaters_data.columns:
            logger.info("\n📈 Top 10 players by xGoals:")
            top_xg = self.skaters_data.nlargest(10, "I_F_xGoals")[["name", "team", "I_F_xGoals", "I_F_goals"]]
            for _, player in top_xg.iterrows():
                logger.info(
                    f"  {player['name']} ({player['team']}): {player['I_F_xGoals']:.1f} xG, {player['I_F_goals']} goals"
                )

    def create_player_quality_lookup(self) -> dict:
        """Create a lookup table for player shooting quality"""
        if self.skaters_data is None:
            self.load_skaters_data()

        logger.info("\n📊 Creating player quality lookup table...")

        # Calculate shooting percentage for each player
        if self.skaters_data is not None and "I_F_goals" in self.skaters_data.columns and "I_F_shotsOnGoal" in self.skaters_data.columns:
            self.skaters_data["shooting_pct"] = (
                self.skaters_data["I_F_goals"] / self.skaters_data["I_F_shotsOnGoal"].replace(0, np.nan)
            ).fillna(0)

            # Create lookup by player ID
            player_lookup = {}
            if self.skaters_data is not None:
                for _, player in self.skaters_data.iterrows():
                    if "playerId" in self.skaters_data.columns:
                        player_id = player["playerId"]
                        player_lookup[player_id] = {
                            "name": player.get("name", "Unknown"),
                            "shooting_pct": player.get("shooting_pct", 0.09),  # League average ~9%
                            "xgoals": player.get("I_F_xGoals", 0),
                            "goals": player.get("I_F_goals", 0),
                            "shots": player.get("I_F_shotsOnGoal", 0),
                        }

            logger.info(f"✓ Created lookup for {len(player_lookup)} players")
            return player_lookup
        else:
            logger.warning("Missing required columns for player quality lookup")
            return {}

    def get_download_instructions(self):
        """Instructions for getting shot-level data"""
        logger.info("\n" + "=" * 60)
        logger.info("HOW TO GET SHOT-LEVEL DATA FOR xG MODELING")
        logger.info("=" * 60)

        logger.info("\nYou have player-level data, but need shot-level data for xG modeling.")
        logger.info("\nOption 1: MoneyPuck Website")
        logger.info("1. Go to https://moneypuck.com/data.htm")
        logger.info("2. Scroll down to 'Download Shot Data'")
        logger.info("3. Download 'All Shots' (1.8M+ shots from 2007-2024)")
        logger.info("4. Extract the CSV to your data/ folder")

        logger.info("\nOption 2: Use NHL API Instead")
        logger.info("1. Use the fetch_enhanced_nhl_data.py script")
        logger.info("2. It will get play-by-play data with shot locations")
        logger.info("3. No shot speed, but includes all other features")

        logger.info("\nOption 3: Build with Player Data")
        logger.info("1. Create a player-quality-based model")
        logger.info("2. Use the skaters.csv for player shooting percentages")
        logger.info("3. Less accurate but still useful for demonstration")


def main():
    """Explore your MoneyPuck data"""
    loader = MoneyPuckDataLoader()

    # Check what files we have
    # files = loader.explore_available_files()  # Unused variable
    loader.explore_available_files()

    # Load skaters data
    try:
        loader.load_skaters_data()
        loader.analyze_skaters_features()

        # Check for shot data
        shot_path = loader.check_for_shot_data()

        if not shot_path:
            # We only have player data, not shot data
            loader.get_download_instructions()

            # But we can still create player quality metrics
            player_lookup = loader.create_player_quality_lookup()

            # Save player lookup for later use
            import json

            with open("data/player_quality_lookup.json", "w") as f:
                json.dump(player_lookup, f, indent=2)
            logger.info("\n✓ Saved player quality lookup to data/player_quality_lookup.json")

    except Exception as e:
        logger.error(f"Error loading data: {e}")


if __name__ == "__main__":
    main()
