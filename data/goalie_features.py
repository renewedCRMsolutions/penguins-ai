# File: data/goalie_features.py
"""
Advanced goalie feature engineering for xG model
Focuses on goalie-specific factors that significantly impact save probability
"""

import pandas as pd
import numpy as np
# from typing import Optional  # Unused import
# from datetime import timedelta  # Unused import
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalieFeatureEngineer:
    """Engineer goalie-specific features for xG model"""

    def __init__(self, data_dir: str = "data/nhl/enhanced"):
        self.data_dir = Path(data_dir)
        self.goalie_stats_cache = {}
        self.matchup_history = {}

    def engineer_all_features(
        self, shots_df: pd.DataFrame, goalie_stats_df: pd.DataFrame, games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add all goalie-related features to shots dataframe"""

        logger.info("Engineering goalie features...")

        # Sort by game and time for proper sequence
        shots_df = shots_df.sort_values(["game_id", "period", "time"])

        # 1. Basic goalie stats
        shots_df = self._add_goalie_season_stats(shots_df, goalie_stats_df)

        # 2. Goalie workload features
        shots_df = self._add_workload_features(shots_df)

        # 3. Goalie form/momentum
        shots_df = self._add_goalie_form_features(shots_df, games_df)

        # 4. Shooter-goalie matchup history
        shots_df = self._add_matchup_features(shots_df)

        # 5. Situational save percentages
        shots_df = self._add_situational_features(shots_df)

        # 6. Fatigue indicators
        shots_df = self._add_fatigue_features(shots_df)

        # 7. Advanced timing features
        shots_df = self._add_timing_features(shots_df)

        logger.info(f"Added {len([c for c in shots_df.columns if 'goalie_' in c])} goalie features")

        return shots_df

    def _add_goalie_season_stats(self, shots_df: pd.DataFrame, goalie_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Add goalie's season statistics"""

        # Key goalie metrics
        goalie_features = [
            "wins",
            "losses",
            "otLosses",
            "gamesPlayed",
            "gamesStarted",
            "shotsAgainst",
            "goalsAgainst",
            "savePctg",
            "shutouts",
            "timeOnIce",
            "qualityStarts",
            "qualityStartsPctg",
            "reallyBadStarts",
            "goalsForAverage",
            "goalsAgainstAverage",
        ]

        # Create goalie lookup
        goalie_lookup = {}
        for _, goalie in goalie_stats_df.iterrows():
            goalie_id = goalie.get("playerId", goalie.get("goalieId"))
            if goalie_id:
                stats = {}
                for feat in goalie_features:
                    if feat in goalie:
                        stats[f"goalie_{feat}"] = goalie[feat]
                goalie_lookup[goalie_id] = stats

        # Add to shots
        for feat in [f"goalie_{f}" for f in goalie_features]:
            shots_df[feat] = np.nan

        for idx, shot in shots_df.iterrows():
            goalie_id = shot.get("goalie_id")
            if goalie_id in goalie_lookup:
                for feat, value in goalie_lookup[goalie_id].items():
                    shots_df.at[idx, feat] = value

        # Calculate derived features
        shots_df["goalie_quality_rating"] = (
            shots_df["goalie_savePctg"] * 0.4
            + shots_df["goalie_qualityStartsPctg"] * 0.3
            + (1 - shots_df["goalie_goalsAgainstAverage"] / 4) * 0.3
        )

        return shots_df

    def _add_workload_features(self, shots_df: pd.DataFrame) -> pd.DataFrame:
        """Add goalie workload features within game"""

        # Initialize new columns
        workload_features = [
            "goalie_shots_faced_period",
            "goalie_shots_faced_last_5min",
            "goalie_shots_faced_last_10min",
            "goalie_save_pct_period",
            "goalie_save_pct_last_10_shots",
            "time_since_last_shot",
            "time_since_last_goal",
            "consecutive_saves",
        ]

        for feat in workload_features:
            shots_df[feat] = 0

        # Process each game
        for game_id in shots_df["game_id"].unique():
            game_shots = shots_df[shots_df["game_id"] == game_id].copy()

            # Track per goalie
            goalie_stats = {}

            for idx, shot in game_shots.iterrows():
                goalie_id = shot["goalie_id"]
                period = shot["period"]
                time = shot["time"]

                if goalie_id not in goalie_stats:
                    goalie_stats[goalie_id] = {
                        "shots_faced": 0,
                        "shots_by_period": {1: 0, 2: 0, 3: 0, 4: 0},
                        "goals_allowed": 0,
                        "last_shot_time": None,
                        "last_goal_time": None,
                        "recent_shots": [],  # (time, is_goal) tuples
                        "consecutive_saves": 0,
                    }

                stats = goalie_stats[goalie_id]

                # Update shot counts
                stats["shots_faced"] += 1
                stats["shots_by_period"][period] = stats["shots_by_period"].get(period, 0) + 1

                # Time since last shot
                if stats["last_shot_time"]:
                    shots_df.at[idx, "time_since_last_shot"] = self._time_diff(
                        time, stats["last_shot_time"], period, stats["last_shot_period"]
                    )

                # Recent shots tracking
                current_time_total = (period - 1) * 1200 + self._parse_time(time)
                stats["recent_shots"].append((current_time_total, shot["is_goal"]))

                # Keep only recent shots (last 10 minutes)
                stats["recent_shots"] = [(t, g) for t, g in stats["recent_shots"] if current_time_total - t <= 600]

                # Calculate features
                shots_df.at[idx, "goalie_shots_faced_period"] = stats["shots_by_period"][period]

                # Shots in last 5/10 minutes
                shots_5min = sum(1 for t, _ in stats["recent_shots"] if current_time_total - t <= 300)
                shots_10min = len(stats["recent_shots"])
                shots_df.at[idx, "goalie_shots_faced_last_5min"] = shots_5min
                shots_df.at[idx, "goalie_shots_faced_last_10min"] = shots_10min

                # Save percentage calculations
                if stats["shots_by_period"][period] > 0:
                    period_goals = sum(1 for _, g in stats["recent_shots"] if g)
                    shots_df.at[idx, "goalie_save_pct_period"] = 1 - period_goals / stats["shots_by_period"][period]

                # Last 10 shots save percentage
                last_10_shots = stats["recent_shots"][-10:]
                if len(last_10_shots) > 0:
                    saves_last_10 = sum(1 for _, g in last_10_shots if not g)
                    shots_df.at[idx, "goalie_save_pct_last_10_shots"] = saves_last_10 / len(last_10_shots)

                # Update tracking
                stats["last_shot_time"] = time
                stats["last_shot_period"] = period

                if shot["is_goal"]:
                    stats["goals_allowed"] += 1
                    stats["last_goal_time"] = time
                    stats["consecutive_saves"] = 0
                    if stats["last_goal_time"]:
                        shots_df.at[idx, "time_since_last_goal"] = 0
                else:
                    stats["consecutive_saves"] += 1
                    shots_df.at[idx, "consecutive_saves"] = stats["consecutive_saves"]
                    if stats["last_goal_time"]:
                        shots_df.at[idx, "time_since_last_goal"] = self._time_diff(
                            time, stats["last_goal_time"], period, stats.get("last_goal_period", period)
                        )

        return shots_df

    def _add_goalie_form_features(self, shots_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        """Add goalie form/momentum features based on recent games"""

        # Features to add
        form_features = [
            "goalie_last_5_games_save_pct",
            "goalie_last_5_games_gaa",
            "goalie_vs_team_career_save_pct",
            "goalie_home_away_save_pct_diff",
            "goalie_recent_form_trend",  # Improving/declining
            "goalie_rest_days",
        ]

        for feat in form_features:
            shots_df[feat] = np.nan

        # This would require historical game data
        # For now, adding placeholder logic
        # In production, you'd query historical games for each goalie

        # Example: Add random form metrics for demonstration
        shots_df["goalie_last_5_games_save_pct"] = np.random.uniform(0.88, 0.94, len(shots_df))
        shots_df["goalie_last_5_games_gaa"] = np.random.uniform(2.0, 3.5, len(shots_df))
        shots_df["goalie_recent_form_trend"] = np.random.choice([-1, 0, 1], len(shots_df))
        shots_df["goalie_rest_days"] = np.random.choice([0, 1, 2, 3, 7], len(shots_df))

        return shots_df

    def _add_matchup_features(self, shots_df: pd.DataFrame) -> pd.DataFrame:
        """Add shooter vs goalie historical matchup data"""

        matchup_features = [
            "shooter_vs_goalie_career_goals",
            "shooter_vs_goalie_career_shots",
            "shooter_vs_goalie_career_shooting_pct",
            "shooter_vs_goalie_last_season_goals",
            "team_vs_goalie_last_5_games_goals",
        ]

        for feat in matchup_features:
            shots_df[feat] = 0

        # Track matchups as we process
        for idx, shot in shots_df.iterrows():
            shooter_id = shot["shooter_id"]
            goalie_id = shot["goalie_id"]

            key = f"{shooter_id}_{goalie_id}"

            if key not in self.matchup_history:
                self.matchup_history[key] = {"shots": 0, "goals": 0}

            # Update with historical data
            shots_df.at[idx, "shooter_vs_goalie_career_shots"] = self.matchup_history[key]["shots"]
            shots_df.at[idx, "shooter_vs_goalie_career_goals"] = self.matchup_history[key]["goals"]

            if self.matchup_history[key]["shots"] > 0:
                shots_df.at[idx, "shooter_vs_goalie_career_shooting_pct"] = (
                    self.matchup_history[key]["goals"] / self.matchup_history[key]["shots"]
                )

            # Update history
            self.matchup_history[key]["shots"] += 1
            if shot["is_goal"]:
                self.matchup_history[key]["goals"] += 1

        return shots_df

    def _add_situational_features(self, shots_df: pd.DataFrame) -> pd.DataFrame:
        """Add goalie performance in specific situations"""

        situational_features = [
            "goalie_powerplay_save_pct",
            "goalie_even_strength_save_pct",
            "goalie_high_danger_save_pct",
            "goalie_medium_danger_save_pct",
            "goalie_low_danger_save_pct",
            "goalie_first_shot_save_pct",
            "goalie_rebound_control_pct",
            "goalie_breakaway_save_pct",
        ]

        # Initialize
        for feat in situational_features:
            shots_df[feat] = np.nan

        # Track situational performance per goalie
        goalie_situations = {}

        for idx, shot in shots_df.iterrows():
            goalie_id = shot["goalie_id"]

            if goalie_id not in goalie_situations:
                goalie_situations[goalie_id] = {
                    "powerplay": {"shots": 0, "saves": 0},
                    "even_strength": {"shots": 0, "saves": 0},
                    "high_danger": {"shots": 0, "saves": 0},
                    "medium_danger": {"shots": 0, "saves": 0},
                    "low_danger": {"shots": 0, "saves": 0},
                    "first_shot": {"shots": 0, "saves": 0},
                    "rebounds": {"shots": 0, "saves": 0},
                }

            stats = goalie_situations[goalie_id]

            # Categorize shot
            if shot.get("is_power_play", False):
                situation = "powerplay"
            else:
                situation = "even_strength"

            danger = shot.get("danger_zone", "medium")

            # Update current stats before this shot
            for sit_type, sit_stats in stats.items():
                if sit_stats["shots"] > 0:
                    save_pct = sit_stats["saves"] / sit_stats["shots"]
                    feat_name = f"goalie_{sit_type}_save_pct"
                    if feat_name in situational_features:
                        shots_df.at[idx, feat_name] = save_pct

            # Update tracking
            stats[situation]["shots"] += 1
            stats[danger]["shots"] += 1

            if not shot["is_goal"]:
                stats[situation]["saves"] += 1
                stats[danger]["saves"] += 1

            if shot.get("is_rebound", False):
                stats["rebounds"]["shots"] += 1
                if not shot["is_goal"]:
                    stats["rebounds"]["saves"] += 1

        return shots_df

    def _add_fatigue_features(self, shots_df: pd.DataFrame) -> pd.DataFrame:
        """Add goalie fatigue indicators"""

        fatigue_features = [
            "goalie_games_in_last_7_days",
            "goalie_minutes_last_3_games",
            "goalie_back_to_back",
            "goalie_period_fatigue_score",
            "goalie_game_fatigue_score",
        ]

        for feat in fatigue_features:
            shots_df[feat] = 0

        # Calculate in-game fatigue
        for game_id in shots_df["game_id"].unique():
            game_shots = shots_df[shots_df["game_id"] == game_id]

            for idx, shot in game_shots.iterrows():
                period = shot["period"]
                time_in_period = self._parse_time(shot["time"])

                # Period fatigue (increases as period progresses)
                period_fatigue = (time_in_period / 1200) * period * 0.1
                shots_df.at[idx, "goalie_period_fatigue_score"] = period_fatigue

                # Game fatigue (based on shots faced)
                shots_faced = shot.get("goalie_shots_faced", 0)
                game_fatigue = min(shots_faced / 40, 1.0)  # Normalize to 0-1
                shots_df.at[idx, "goalie_game_fatigue_score"] = game_fatigue

        # Add back-to-back indicator (would need schedule data)
        shots_df["goalie_back_to_back"] = np.random.choice([0, 1], len(shots_df), p=[0.8, 0.2])

        return shots_df

    def _add_timing_features(self, shots_df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced timing features"""

        timing_features = [
            "goalie_cold_start",  # First shot after long break
            "goalie_shot_frequency",  # Shots per minute recently
            "goalie_rhythm_score",  # Consistency of shot timing
            "time_since_timeout",
            "time_since_penalty_kill",
        ]

        for feat in timing_features:
            shots_df[feat] = 0

        for game_id in shots_df["game_id"].unique():
            game_shots = shots_df[shots_df["game_id"] == game_id]

            for idx, shot in game_shots.iterrows():
                # Cold start (first shot or >5 min since last)
                time_since_last = shot.get("time_since_last_shot", 0)
                shots_df.at[idx, "goalie_cold_start"] = int(time_since_last > 300)

                # Shot frequency
                recent_shots = shot.get("goalie_shots_faced_last_5min", 0)
                shots_df.at[idx, "goalie_shot_frequency"] = recent_shots / 5.0

                # Rhythm score (variance in shot timing)
                # Would calculate from actual shot time differences
                shots_df.at[idx, "goalie_rhythm_score"] = np.random.uniform(0, 1)

        return shots_df

    def _parse_time(self, time_str: str) -> float:
        """Convert MM:SS to seconds"""
        if isinstance(time_str, str) and ":" in time_str:
            parts = time_str.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str) if time_str else 0

    def _time_diff(self, time1: str, time2: str, period1: int, period2: int) -> float:
        """Calculate time difference in seconds between two game times"""
        t1 = self._parse_time(time1) + (period1 - 1) * 1200
        t2 = self._parse_time(time2) + (period2 - 1) * 1200
        return abs(t1 - t2)

    def save_engineered_features(self, shots_df: pd.DataFrame, output_name: str):
        """Save the engineered dataset"""
        output_path = self.data_dir / f"{output_name}_with_goalie_features.csv"
        shots_df.to_csv(output_path, index=False)
        logger.info(f"Saved engineered features to {output_path}")

        # Also save feature importance hints
        # goalie_cols = [col for col in shots_df.columns if "goalie_" in col]

        feature_importance = {
            "critical": [
                "goalie_save_pct_last_10_shots",
                "goalie_shots_faced_period",
                "goalie_high_danger_save_pct",
                "goalie_game_fatigue_score",
                "goalie_cold_start",
            ],
            "important": [
                "goalie_quality_rating",
                "time_since_last_shot",
                "consecutive_saves",
                "goalie_recent_form_trend",
                "shooter_vs_goalie_career_shooting_pct",
            ],
            "contextual": [
                "goalie_back_to_back",
                "goalie_shot_frequency",
                "goalie_powerplay_save_pct",
                "goalie_rebound_control_pct",
            ],
        }

        import json

        with open(self.data_dir / "goalie_feature_importance.json", "w") as f:
            json.dump(feature_importance, f, indent=2)

        return shots_df


# Example usage
if __name__ == "__main__":
    engineer = GoalieFeatureEngineer()

    # Load data
    shots_df = pd.read_csv("data/nhl/enhanced/enhanced_shots_2024-10-01_to_2024-10-31.csv")
    goalie_stats_df = pd.read_csv("data/nhl/enhanced/goalie_stats_20242025.csv")
    games_df = pd.read_csv("data/nhl/enhanced/games_2024-10-01_to_2024-10-31.csv")

    # Engineer features
    enhanced_shots = engineer.engineer_all_features(shots_df, goalie_stats_df, games_df)

    # Save
    engineer.save_engineered_features(enhanced_shots, "october_2024")

    print(f"Total features: {len(enhanced_shots.columns)}")
    print(f"Goalie features: {len([c for c in enhanced_shots.columns if 'goalie_' in c])}")
