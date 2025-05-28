# File: train_with_all_tables.py
"""
Train xG model using ALL available NHL data tables joined together
Includes: shots, goalies, skaters, teams, lines
"""

import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from train.train_xg_final import AdvancedXGModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveNHLDataPreparer:
    """Prepare NHL data by joining all available tables"""

    def __init__(self, data_dir: str = "data/nhl"):
        self.data_dir = Path(data_dir)
        self.shots_df = None
        self.goalies_df = None
        self.skaters_df = None
        self.teams_df = None
        self.lines_df = None

    def load_all_data(self):
        """Load all available data tables"""
        logger.info("Loading all NHL data tables...")

        # 1. Load shots data (from our collection)
        enhanced_dir = self.data_dir / "enhanced"
        shot_files = list(enhanced_dir.glob("nhl_shots_*.csv"))
        if shot_files:
            latest_shots = sorted(shot_files)[-1]
            self.shots_df = pd.read_csv(latest_shots)
            logger.info(f"Loaded {len(self.shots_df)} shots from {latest_shots.name}")

        # 2. Load MoneyPuck data if available
        # Goalies
        goalie_file = self.data_dir / "goalies.csv"
        if goalie_file.exists():
            self.goalies_df = pd.read_csv(goalie_file)
            logger.info(f"Loaded {len(self.goalies_df)} goalie records")

        # Skaters
        skater_file = self.data_dir / "skaters.csv"
        if skater_file.exists():
            self.skaters_df = pd.read_csv(skater_file)
            logger.info(f"Loaded {len(self.skaters_df)} skater records")

        # Teams
        team_file = self.data_dir / "teams.csv"
        if team_file.exists():
            self.teams_df = pd.read_csv(team_file)
            logger.info(f"Loaded {len(self.teams_df)} team records")

        # Lines
        line_file = self.data_dir / "lines.csv"
        if line_file.exists():
            self.lines_df = pd.read_csv(line_file)
            logger.info(f"Loaded {len(self.lines_df)} line records")

    def prepare_goalie_features(self) -> pd.DataFrame:
        """Prepare goalie features from MoneyPuck data"""
        if self.goalies_df is None:
            logger.warning("No goalie data available")
            return pd.DataFrame()

        # Select key goalie features
        goalie_features = [
            "playerId",
            "name",
            "team",
            "games_played",
            "xGoals",
            "goals",
            "xOnGoal",
            "ongoal",
            "lowDangerShots",
            "mediumDangerShots",
            "highDangerShots",
            "lowDangerxGoals",
            "mediumDangerxGoals",
            "highDangerxGoals",
            "lowDangerGoals",
            "mediumDangerGoals",
            "highDangerGoals",
            "xRebounds",
            "rebounds",
            "xFreeze",
            "freeze",
        ]

        # Filter to available columns
        available_features = [f for f in goalie_features if f in self.goalies_df.columns]
        goalie_df = self.goalies_df[available_features].copy()

        # Calculate save percentages by danger zone
        if all(col in goalie_df.columns for col in ["lowDangerShots", "lowDangerGoals"]):
            goalie_df["lowDanger_save_pct"] = 1 - (goalie_df["lowDangerGoals"] / goalie_df["lowDangerShots"]).fillna(0)
            goalie_df["mediumDanger_save_pct"] = 1 - (
                goalie_df["mediumDangerGoals"] / goalie_df["mediumDangerShots"]
            ).fillna(0)
            goalie_df["highDanger_save_pct"] = 1 - (goalie_df["highDangerGoals"] / goalie_df["highDangerShots"]).fillna(
                0
            )

        # Overall save percentage
        if "goals" in goalie_df.columns and "ongoal" in goalie_df.columns:
            goalie_df["overall_save_pct"] = 1 - (goalie_df["goals"] / goalie_df["ongoal"]).fillna(0)

        # Rename for merging
        goalie_df = goalie_df.rename(columns={"playerId": "goalie_id", "name": "goalie_name_mp"})

        # Prefix columns
        for col in goalie_df.columns:
            if col not in ["goalie_id", "goalie_name_mp"]:
                goalie_df = goalie_df.rename(columns={col: f"goalie_{col}"})

        return goalie_df

    def prepare_skater_features(self) -> pd.DataFrame:
        """Prepare skater features from MoneyPuck data"""
        if self.skaters_df is None:
            logger.warning("No skater data available")
            return pd.DataFrame()

        # Select key skater features
        skater_features = [
            "playerId",
            "name",
            "team",
            "position",
            "games_played",
            "I_F_goals",
            "I_F_xGoals",
            "I_F_shotsOnGoal",
            "I_F_shotAttempts",
            "I_F_rebounds",
            "I_F_xRebounds",
            "I_F_freeze",
            "I_F_lowDangerShots",
            "I_F_mediumDangerShots",
            "I_F_highDangerShots",
            "I_F_lowDangerGoals",
            "I_F_mediumDangerGoals",
            "I_F_highDangerGoals",
            "OnIce_F_xGoals",
            "OnIce_A_xGoals",
            "gameScore",
        ]

        # Filter to available columns
        available_features = [f for f in skater_features if f in self.skaters_df.columns]
        skater_df = self.skaters_df[available_features].copy()

        # Calculate shooting percentage
        if "I_F_goals" in skater_df.columns and "I_F_shotsOnGoal" in skater_df.columns:
            skater_df["shooting_pct"] = (skater_df["I_F_goals"] / skater_df["I_F_shotsOnGoal"]).fillna(0)

        # Goals above expected
        if "I_F_goals" in skater_df.columns and "I_F_xGoals" in skater_df.columns:
            skater_df["goals_above_expected"] = skater_df["I_F_goals"] - skater_df["I_F_xGoals"]

        # Rename for merging
        skater_df = skater_df.rename(columns={"playerId": "shooter_id", "name": "shooter_name_mp"})

        # Prefix columns
        for col in skater_df.columns:
            if col not in ["shooter_id", "shooter_name_mp"]:
                skater_df = skater_df.rename(columns={col: f"shooter_{col}"})

        return skater_df

    def prepare_team_features(self) -> pd.DataFrame:
        """Prepare team features"""
        if self.teams_df is None:
            logger.warning("No team data available")
            return pd.DataFrame()

        # Select key team features
        team_features = [
            "team",
            "xGoalsPercentage",
            "corsiPercentage",
            "fenwickPercentage",
            "xGoalsFor",
            "goalsFor",
            "xGoalsAgainst",
            "goalsAgainst",
            "shotsOnGoalFor",
            "shotAttemptsFor",
            "highDangerShotsFor",
            "mediumDangerShotsFor",
            "lowDangerShotsFor",
        ]

        # Filter to available columns
        available_features = [f for f in team_features if f in self.teams_df.columns]
        team_df = self.teams_df[available_features].copy()

        # Prefix columns
        for col in team_df.columns:
            if col != "team":
                team_df = team_df.rename(columns={col: f"team_{col}"})

        return team_df

    def join_all_data(self) -> pd.DataFrame | None:
        """Join all data tables together"""
        logger.info("\nJoining all data tables...")

        if self.shots_df is None:
            logger.error("No shot data available!")
            return None

        # Start with shots data
        df = self.shots_df.copy()
        initial_rows = len(df)

        # Extract player IDs from names if needed
        if "shooter_name" in df.columns and df["shooter_name"].str.contains("Player_").any():
            df["shooter_id"] = df["shooter_name"].str.extract(r"Player_(\d+)")[0].astype(float)

        if "goalie_name" in df.columns and df["goalie_name"].str.contains("Goalie_").any():
            df["goalie_id"] = df["goalie_name"].str.extract(r"Goalie_(\d+)")[0].astype(float)

        # 1. Join goalie data
        goalie_features = self.prepare_goalie_features()
        if not goalie_features.empty:
            logger.info(f"Joining {len(goalie_features)} goalie records...")
            df = df.merge(goalie_features, on="goalie_id", how="left", suffixes=("", "_goalie"))
            logger.info(f"After goalie join: {len(df)} rows")

        # 2. Join skater data
        skater_features = self.prepare_skater_features()
        if not skater_features.empty:
            logger.info(f"Joining {len(skater_features)} skater records...")
            df = df.merge(skater_features, on="shooter_id", how="left", suffixes=("", "_skater"))
            logger.info(f"After skater join: {len(df)} rows")

        # 3. Join team data
        team_features = self.prepare_team_features()
        if not team_features.empty and "shooting_team" in df.columns:
            logger.info(f"Joining {len(team_features)} team records...")
            # Map team names if needed
            df = df.merge(team_features, left_on="home_team", right_on="team", how="left", suffixes=("", "_team"))
            logger.info(f"After team join: {len(df)} rows")

        # Create composite features
        df = self.create_composite_features(df)

        # Ensure we didn't lose rows
        if len(df) != initial_rows:
            logger.warning(f"Row count changed from {initial_rows} to {len(df)} after joins!")

        return df

    def create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite features from joined data"""
        logger.info("Creating composite features...")

        # 1. Shooter vs Goalie matchup features
        if "shooter_shooting_pct" in df.columns and "goalie_overall_save_pct" in df.columns:
            df["shooter_vs_goalie_advantage"] = df["shooter_shooting_pct"] - (1 - df["goalie_overall_save_pct"])

        # 2. Danger zone matchup
        if "danger_zone" in df.columns:
            # Map danger zones to goalie save percentages
            danger_map = {
                "high": "goalie_highDanger_save_pct",
                "medium": "goalie_mediumDanger_save_pct",
                "low": "goalie_lowDanger_save_pct",
            }

            for zone, col in danger_map.items():
                if col in df.columns:
                    mask = df["danger_zone"] == zone
                    df.loc[mask, "goalie_zone_save_pct"] = df.loc[mask, col]

        # 3. Team momentum
        if "team_xGoalsPercentage" in df.columns:
            df["team_momentum"] = df["team_xGoalsPercentage"] - 50  # Center around 0

        # 4. Shooter hot/cold
        if "shooter_goals_above_expected" in df.columns:
            df["shooter_is_hot"] = (df["shooter_goals_above_expected"] > 2).astype(int)
            df["shooter_is_cold"] = (df["shooter_goals_above_expected"] < -2).astype(int)

        # 5. Fill missing values with reasonable defaults
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        return df

    def prepare_for_model(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Prepare final dataset for model training"""
        logger.info("Preparing data for model...")

        # Ensure we have all required base features
        required_features = ["shot_distance", "shot_angle", "x_coord", "y_coord", "danger_zone", "shot_type", "is_goal"]

        missing = [f for f in required_features if f not in df.columns]
        if missing:
            logger.error(f"Missing required features: {missing}")
            return None

        # Add any missing expected features with defaults
        if "is_rush" not in df.columns:
            df["is_rush"] = df.get("is_rush", 0)
        if "is_rebound" not in df.columns:
            df["is_rebound"] = df.get("is_rebound", 0)
        if "is_power_play" not in df.columns:
            df["is_power_play"] = df.get("is_power_play", 0)

        # Add placeholder features that model expects
        expected_features = [
            "is_one_timer",
            "traffic_score",
            "passes_before_shot",
            "time_since_zone_entry",
            "time_since_faceoff",
            "period",
            "time_remaining",
            "score_diff",
            "is_home_team",
            "momentum_score",
            "zone_time",
            "shot_attempts_sequence",
            "goalie_savePctg",
            "goalie_quality_rating",
            "goalie_shots_faced_period",
            "goalie_shots_faced_last_5min",
            "goalie_save_pct_last_10_shots",
            "goalie_save_pct_period",
            "time_since_last_shot",
            "consecutive_saves",
            "goalie_high_danger_save_pct",
            "goalie_fatigue_score",
            "goalie_cold_start",
            "goalie_recent_form_trend",
            "shooter_vs_goalie_career_shooting_pct",
        ]

        for feat in expected_features:
            if feat not in df.columns:
                # Use actual data if available from joins
                if feat == "goalie_savePctg" and "goalie_overall_save_pct" in df.columns:
                    df[feat] = df["goalie_overall_save_pct"]
                elif feat == "goalie_high_danger_save_pct" and "goalie_highDanger_save_pct" in df.columns:
                    df[feat] = df["goalie_highDanger_save_pct"]
                elif feat == "period" and "period" not in df.columns:
                    df[feat] = 2  # Default to 2nd period
                elif feat == "time_remaining":
                    df[feat] = 1200  # Default to full period
                elif feat == "is_home_team" and "shooting_team" in df.columns:
                    df[feat] = (df["shooting_team"] == "home").astype(int)
                else:
                    df[feat] = 0  # Default value

        # Ensure numeric danger zone encoding
        if df["danger_zone"].dtype == "object":
            df["danger_zone_encoded"] = pd.Categorical(df["danger_zone"]).codes

        logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

        return df


def main():
    """Main training pipeline with all tables joined"""

    logger.info("=" * 60)
    logger.info("NHL XG MODEL TRAINING - ALL TABLES JOINED")
    logger.info("=" * 60)

    # Initialize data preparer
    preparer = ComprehensiveNHLDataPreparer()

    # Load all data
    preparer.load_all_data()

    # Join everything together
    df = preparer.join_all_data()

    if df is None:
        logger.error("Failed to join data!")
        return

    # Prepare for model
    df_final = preparer.prepare_for_model(df)

    if df_final is None:
        logger.error("Failed to prepare data!")
        return

    # Show data summary
    logger.info("\n" + "=" * 40)
    logger.info("DATA SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Total shots: {len(df_final)}")
    logger.info(f"Goals: {df_final['is_goal'].sum()} ({df_final['is_goal'].mean():.3f})")
    logger.info(f"Total features: {len(df_final.columns)}")

    # Show feature categories
    goalie_features = [c for c in df_final.columns if "goalie_" in c]
    shooter_features = [c for c in df_final.columns if "shooter_" in c]
    team_features = [c for c in df_final.columns if "team_" in c]

    logger.info(f"\nFeature breakdown:")
    logger.info(f"  Goalie features: {len(goalie_features)}")
    logger.info(f"  Shooter features: {len(shooter_features)}")
    logger.info(f"  Team features: {len(team_features)}")

    # Save prepared data
    output_file = Path("data/nhl/enhanced") / f"nhl_all_tables_joined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_final.to_csv(output_file, index=False)
    logger.info(f"\nSaved joined data to: {output_file}")

    # Train the model
    logger.info("\n" + "=" * 40)
    logger.info("TRAINING XG MODEL")
    logger.info("=" * 40)

    model = AdvancedXGModel()

    try:
        final_auc = model.train_full_pipeline(str(output_file))

        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Model AUC: {final_auc:.4f}")
        logger.info(f"MoneyPuck AUC: 0.7781")
        logger.info(f"Difference: {final_auc - 0.7781:+.4f}")

        if final_auc > 0.7781:
            logger.info("\n🎉 WE BEAT MONEYPUCK! 🎉")
            logger.info("\nKey advantages from joined data:")
            logger.info("- Real goalie save percentages by danger zone")
            logger.info("- Shooter historical performance metrics")
            logger.info("- Team strength indicators")
            logger.info("- Shooter vs goalie matchup features")
        else:
            logger.info("\nGetting closer with the enriched data!")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
