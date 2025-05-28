# File: data/collect_nhl_fixed.py
"""
Fixed NHL data collection that actually works with the API
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path
from nhlpy import NHLClient
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHLDataCollectorFixed:
    """Fixed collector that properly extracts shot data"""

    def __init__(self, data_dir: str = "data/nhl/enhanced"):
        self.client = NHLClient()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._boxscore_cache = {}  # Cache for boxscore data
        self.game_id = None  # Current game being processed

    def collect_season_data(
        self,
        season: str = "20242025",
        team: Optional[str] = None,
        game_types: List[int] = [2, 3],
        max_games: Optional[int] = None,
    ):
        """
        Collect shot data for a season
        game_types: 2=regular season, 3=playoffs
        """
        logger.info(f"Collecting {season} season data...")

        # Get all game IDs for the season
        all_game_ids = self.client.helpers.get_gameids_by_season(season, game_types=game_types)

        if team:
            # Filter for specific team if requested
            logger.info(f"Filtering for {team} games...")
            team_games = []
            for game_id in all_game_ids:
                if team.upper() in str(game_id):  # Simple filter, could be improved
                    team_games.append(game_id)
            all_game_ids = team_games

        if max_games:
            all_game_ids = all_game_ids[:max_games]

        logger.info(f"Processing {len(all_game_ids)} games...")

        all_shots = []
        games_processed = 0

        for i, game_id in enumerate(all_game_ids):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(all_game_ids)} games...")

            shots = self.process_game_fixed(int(game_id))
            if shots:
                all_shots.extend(shots)
                games_processed += 1

            # Be nice to the API
            if i % 20 == 0 and i > 0:
                time.sleep(1)

        logger.info(f"Collected {len(all_shots)} shots from {games_processed} games")

        # Save data
        if all_shots:
            df = pd.DataFrame(all_shots)
            filename = f"nhl_shots_{season}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(self.data_dir / filename, index=False)
            logger.info(f"Saved to {self.data_dir / filename}")

        return all_shots

    def process_game_fixed(self, game_id: int) -> List[Dict]:
        """Process a single game and extract shot data"""
        try:
            self.game_id = game_id  # Store current game ID

            # Get play-by-play data
            pbp = self.client.game_center.play_by_play(game_id=str(game_id))

            if not pbp or "plays" not in pbp:
                return []

            # Also get boxscore for player names
            try:
                boxscore = self.client.game_center.boxscore(game_id=str(game_id))
                if boxscore:
                    self._boxscore_cache[game_id] = boxscore
            except Exception as e:
                logger.debug(f"Could not get boxscore for game {game_id}: {e}")

            # Get game info
            game_info = {
                "game_id": game_id,
                "game_date": pbp.get("gameDate", ""),
                "home_team": pbp.get("homeTeam", {}).get("abbrev", ""),
                "away_team": pbp.get("awayTeam", {}).get("abbrev", ""),
            }

            shots = []
            plays = pbp["plays"]

            # Track game state
            home_score = 0
            away_score = 0

            for i, play in enumerate(plays):
                event_type = play.get("typeDescKey", "")

                # Update score if goal
                if event_type == "goal":
                    details = play.get("details", {})
                    if details.get("eventOwnerTeamId") == pbp.get("homeTeam", {}).get("id"):
                        home_score += 1
                    else:
                        away_score += 1

                # Process shots and goals
                if event_type in ["shot-on-goal", "goal"]:
                    shot_data = self._extract_shot_features(play, plays, i, game_info, home_score, away_score, pbp)
                    if shot_data:
                        shots.append(shot_data)

            return shots

        except Exception as e:
            logger.debug(f"Error processing game {game_id}: {e}")
            return []

    def _extract_shot_features(
        self,
        shot_play: Dict,
        all_plays: List[Dict],
        shot_index: int,
        game_info: Dict,
        home_score: int,
        away_score: int,
        pbp: Dict,
    ) -> Optional[Dict]:
        """Extract features from a shot"""
        try:
            details = shot_play.get("details", {})
            period_desc = shot_play.get("periodDescriptor", {})

            # Basic shot info
            shot_data = {
                **game_info,  # Include game info
                # Shot basics
                "event_idx": shot_play.get("eventId"),
                "period": period_desc.get("number", 0),
                "period_type": period_desc.get("periodType", ""),
                "time_in_period": shot_play.get("timeInPeriod", "00:00"),
                "time_remaining": shot_play.get("timeRemaining", "00:00"),
                # Shot details
                "is_goal": 1 if shot_play.get("typeDescKey") == "goal" else 0,
                "shooter_id": details.get("shootingPlayerId"),
                "shooter_name": self._get_player_name(details.get("shootingPlayerId"), pbp),
                "shot_type": details.get("shotType", "Unknown"),
                "x_coord": details.get("xCoord", 0),
                "y_coord": details.get("yCoord", 0),
                "zone": details.get("zoneCode", ""),
                # Goalie info
                "goalie_id": details.get("goalieInNetId"),
                "goalie_name": self._get_player_name(details.get("goalieInNetId"), pbp),
                # Game state
                "home_score": home_score,
                "away_score": away_score,
                "score_diff": home_score - away_score,
                "shooting_team": (
                    "home" if details.get("eventOwnerTeamId") == pbp.get("homeTeam", {}).get("id") else "away"
                ),
                # Situation
                "situation": shot_play.get("situationCode", ""),
                "home_skaters": (
                    int(shot_play.get("situationCode", "0000")[:2]) if shot_play.get("situationCode") else 5
                ),
                "away_skaters": (
                    int(shot_play.get("situationCode", "0000")[2:]) if shot_play.get("situationCode") else 5
                ),
            }

            # Add calculated features
            shot_data["shot_distance"] = np.sqrt(shot_data["x_coord"] ** 2 + shot_data["y_coord"] ** 2)
            shot_data["shot_angle"] = np.degrees(
                np.arctan2(abs(shot_data["y_coord"]), abs(89 - abs(shot_data["x_coord"])))
            )

            # Determine power play
            if shot_data["shooting_team"] == "home":
                shot_data["is_power_play"] = shot_data["home_skaters"] > shot_data["away_skaters"]
            else:
                shot_data["is_power_play"] = shot_data["away_skaters"] > shot_data["home_skaters"]

            # Add context from previous events
            context = self._get_shot_context(all_plays, shot_index, shot_play)
            shot_data.update(context)

            return shot_data

        except Exception as e:
            logger.debug(f"Error extracting shot features: {e}")
            return None

    def _get_shot_context(self, plays: List[Dict], shot_index: int, shot_play: Dict) -> Dict:
        """Get context from events before the shot"""
        context = {
            "prev_event_type": "",
            "time_since_prev_event": 0,
            "is_rebound": False,
            "is_rush": False,
            "zone_time": 0,
        }

        if shot_index > 0:
            prev_play = plays[shot_index - 1]
            context["prev_event_type"] = prev_play.get("typeDescKey", "")

            # Check for rebound
            if context["prev_event_type"] in ["shot-on-goal", "missed-shot"]:
                prev_time = self._parse_time(prev_play.get("timeInPeriod", "00:00"))
                shot_time = self._parse_time(shot_play.get("timeInPeriod", "00:00"))
                time_diff = abs(shot_time - prev_time)

                context["time_since_prev_event"] = time_diff
                context["is_rebound"] = time_diff <= 3  # Within 3 seconds

            # Look for zone entry
            for i in range(max(0, shot_index - 10), shot_index):
                if plays[i].get("typeDescKey") == "zone-entry":
                    entry_time = self._parse_time(plays[i].get("timeInPeriod", "00:00"))
                    shot_time = self._parse_time(shot_play.get("timeInPeriod", "00:00"))
                    zone_time = abs(shot_time - entry_time)

                    context["zone_time"] = zone_time
                    context["is_rush"] = zone_time < 5  # Quick attack
                    break

        return context

    def _parse_time(self, time_str: str) -> float:
        """Convert MM:SS to seconds"""
        try:
            if ":" in time_str:
                minutes, seconds = time_str.split(":")
                return float(minutes) * 60 + float(seconds)
            return 0
        except Exception:
            return 0

    def _get_player_name(self, player_id: int, pbp: Dict) -> str:
        """Get player name from play details or roster"""
        try:
            # First check if we have boxscore data
            if hasattr(self, "_boxscore_cache") and self.game_id in self._boxscore_cache:
                boxscore = self._boxscore_cache[self.game_id]

                # Check player stats in boxscore
                for team in ["homeTeam", "awayTeam"]:
                    if team in boxscore:
                        # Check forwards, defense, and goalies
                        for position in ["forwards", "defense", "goalies"]:
                            players = boxscore[team].get(position, [])
                            for p_id in players:
                                player_info = boxscore[team].get("players", {}).get(f"ID{p_id}", {})
                                if player_info.get("playerId") == player_id:
                                    first = player_info.get("name", {}).get("firstName", "")
                                    last = player_info.get("name", {}).get("lastName", "")
                                    return f"{first} {last}".strip()

            # Try to get from play-by-play rosterSpots
            for team in ["homeTeam", "awayTeam"]:
                roster = pbp.get("rosterSpots", {}).get(team, {})
                for player_data in roster.values():
                    if player_data.get("playerId") == player_id:
                        first = player_data.get("firstName", "")
                        last = player_data.get("lastName", "")
                        return f"{first} {last}".strip()

            # If not found, return player ID
            return f"Player_{player_id}"

        except Exception as e:
            logger.debug(f"Error getting player name for {player_id}: {e}")
            return f"Player_{player_id}"


def main():
    """Collect NHL shot data"""
    logger.info("=" * 60)
    logger.info("NHL SHOT DATA COLLECTION (FIXED)")
    logger.info("=" * 60)

    collector = NHLDataCollectorFixed()

    # Option 1: Collect recent games (faster for testing)
    logger.info("\nCollecting last 100 games for testing...")
    shots = collector.collect_season_data(
        season="20242025", game_types=[2], max_games=100  # Regular season only  # Limit for testing
    )

    if shots:
        logger.info(f"\n✅ Success! Collected {len(shots)} shots")

        # Show sample of data
        df = pd.DataFrame(shots)
        logger.info(f"\nGoal rate: {df['is_goal'].mean():.3f}")
        logger.info(f"Power play shots: {df['is_power_play'].sum()}")
        logger.info(f"Rebound shots: {df['is_rebound'].sum()}")

        logger.info("\nSample shot data:")
        logger.info(df[["shooter_name", "shot_type", "shot_distance", "shot_angle", "is_goal"]].head())

    # For full season collection, use:
    # shots = collector.collect_season_data(season="20242025", game_types=[2, 3])

    # For Pittsburgh games only:
    # shots = collector.collect_season_data(season="20242025", team="PIT")


if __name__ == "__main__":
    main()
