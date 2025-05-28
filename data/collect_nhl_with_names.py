# File: collect_nhl_with_names.py
"""
Improved NHL data collection that properly gets player names
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


class ImprovedNHLCollector:
    """Collect NHL data with proper player names"""

    def __init__(self, data_dir: str = "data/nhl/enhanced"):
        self.client = NHLClient()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.player_cache = {}  # Cache player info

    def collect_with_details(self, season: str = "20242025", max_games: Optional[int] = None):
        """Collect games with full details including player names"""

        logger.info(f"Collecting {season} season with player details...")

        # Get game IDs
        game_ids = self.client.helpers.get_gameids_by_season(season, game_types=[2, 3])

        if max_games:
            game_ids = game_ids[:max_games]

        logger.info(f"Processing {len(game_ids)} games...")

        all_shots = []
        games_processed = 0

        for i, game_id in enumerate(game_ids):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(game_ids)} games...")

            shots = self.process_game_detailed(int(game_id))
            if shots:
                all_shots.extend(shots)
                games_processed += 1

            # Be nice to the API
            if i % 20 == 0 and i > 0:
                time.sleep(1)

        logger.info(f"Collected {len(all_shots)} shots from {games_processed} games")

        # Convert to DataFrame and save
        if all_shots:
            df = pd.DataFrame(all_shots)

            # Show sample of player names
            logger.info("\nSample of collected data:")
            sample_cols = ["shooter_name", "goalie_name", "shot_type", "shot_distance", "is_goal"]
            logger.info(df[sample_cols].head(10))

            # Save
            filename = f"nhl_shots_with_names_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(self.data_dir / filename, index=False)
            logger.info(f"\nSaved to {self.data_dir / filename}")

            # Summary stats
            logger.info("\nSummary:")
            logger.info(f"Total shots: {len(df)}")
            logger.info(f"Goals: {df['is_goal'].sum()} ({df['is_goal'].mean():.3f})")
            logger.info(f"Unique shooters: {df['shooter_name'].nunique()}")
            logger.info(f"Unique goalies: {df['goalie_name'].nunique()}")

        return all_shots

    def process_game_detailed(self, game_id: int) -> List[Dict]:
        """Process game with detailed player information"""
        try:
            # Get all game data
            pbp = self.client.game_center.play_by_play(game_id=str(game_id))
            boxscore = self.client.game_center.boxscore(game_id=str(game_id))

            if not pbp or "plays" not in pbp:
                return []

            # Build player lookup from boxscore
            player_lookup = self._build_player_lookup(boxscore)

            # Get game info
            game_info = {
                "game_id": game_id,
                "game_date": pbp.get("gameDate", ""),
                "season": pbp.get("season", ""),
                "game_type": pbp.get("gameType", ""),
                "home_team": pbp.get("homeTeam", {}).get("abbrev", ""),
                "away_team": pbp.get("awayTeam", {}).get("abbrev", ""),
                "venue": pbp.get("venue", {}).get("default", ""),
            }

            shots = []
            plays = pbp["plays"]

            # Track game state
            home_score = 0
            away_score = 0
            home_team_id = pbp.get("homeTeam", {}).get("id")

            for i, play in enumerate(plays):
                event_type = play.get("typeDescKey", "")

                # Update score
                if event_type == "goal":
                    details = play.get("details", {})
                    if details.get("eventOwnerTeamId") == home_team_id:
                        home_score += 1
                    else:
                        away_score += 1

                # Process shots and goals
                if event_type in ["shot-on-goal", "goal", "missed-shot"]:
                    shot_data = self._extract_detailed_shot(
                        play, plays, i, game_info, home_score, away_score, player_lookup, home_team_id
                    )
                    if shot_data:
                        shots.append(shot_data)

            return shots

        except Exception as e:
            logger.debug(f"Error processing game {game_id}: {e}")
            return []

    def _build_player_lookup(self, boxscore: Dict) -> Dict[int, Dict]:
        """Build lookup dictionary for player info"""
        lookup = {}

        if not boxscore:
            return lookup

        try:
            # Process both teams
            for team_key in ["homeTeam", "awayTeam"]:
                team_data = boxscore.get(team_key, {})

                # Get all player IDs from different positions
                all_players = []
                all_players.extend(team_data.get("forwards", []))
                all_players.extend(team_data.get("defense", []))
                all_players.extend(team_data.get("goalies", []))

                # Get player details
                players_data = team_data.get("players", {})

                for player_id in all_players:
                    player_key = f"ID{player_id}"
                    if player_key in players_data:
                        player_info = players_data[player_key]

                        # Extract name
                        name_info = player_info.get("name", {})
                        full_name = f"{name_info.get('firstName', '')} {name_info.get('lastName', '')}".strip()

                        # Store in lookup
                        lookup[player_id] = {
                            "name": full_name,
                            "position": player_info.get("position", ""),
                            "jersey": player_info.get("sweaterNumber", ""),
                            "team": team_data.get("abbrev", ""),
                        }

        except Exception as e:
            logger.debug(f"Error building player lookup: {e}")

        return lookup

    def _extract_detailed_shot(
        self,
        shot_play: Dict,
        all_plays: List[Dict],
        shot_index: int,
        game_info: Dict,
        home_score: int,
        away_score: int,
        player_lookup: Dict[int, Dict],
        home_team_id: int,
    ) -> Optional[Dict]:
        """Extract shot with detailed information"""
        try:
            details = shot_play.get("details", {})
            period_desc = shot_play.get("periodDescriptor", {})

            # Get player IDs
            shooter_id = details.get("shootingPlayerId", 0)
            goalie_id = details.get("goalieInNetId", 0)

            # Get player names from lookup
            shooter_info = player_lookup.get(shooter_id, {})
            goalie_info = player_lookup.get(goalie_id, {})

            # Determine shooting team
            shooting_team_id = details.get("eventOwnerTeamId")
            is_home_team = shooting_team_id == home_team_id

            # Basic shot data
            shot_data = {
                **game_info,
                # Event info
                "event_idx": shot_play.get("eventId"),
                "event_type": shot_play.get("typeDescKey"),
                "period": period_desc.get("number", 0),
                "period_type": period_desc.get("periodType", ""),
                "time_in_period": shot_play.get("timeInPeriod", "00:00"),
                "time_remaining": shot_play.get("timeRemaining", "00:00"),
                # Result
                "is_goal": 1 if shot_play.get("typeDescKey") == "goal" else 0,
                "is_shot_on_goal": 1 if shot_play.get("typeDescKey") in ["shot-on-goal", "goal"] else 0,
                # Shooter info
                "shooter_id": shooter_id,
                "shooter_name": shooter_info.get("name", f"Player_{shooter_id}"),
                "shooter_position": shooter_info.get("position", ""),
                "shooter_team": shooter_info.get("team", ""),
                # Shot details
                "shot_type": details.get("shotType", "Unknown"),
                "x_coord": details.get("xCoord", 0),
                "y_coord": details.get("yCoord", 0),
                "zone": details.get("zoneCode", ""),
                # Goalie info
                "goalie_id": goalie_id,
                "goalie_name": goalie_info.get("name", f"Goalie_{goalie_id}"),
                "goalie_team": goalie_info.get("team", ""),
                # Game state
                "home_score": home_score,
                "away_score": away_score,
                "score_diff": home_score - away_score if is_home_team else away_score - home_score,
                "shooting_team": "home" if is_home_team else "away",
                "is_home_team": int(is_home_team),
                # Situation
                "situation": shot_play.get("situationCode", "1551"),
                "home_skaters": (
                    int(shot_play.get("situationCode", "5555")[:2]) if shot_play.get("situationCode") else 5
                ),
                "away_skaters": (
                    int(shot_play.get("situationCode", "5555")[2:4]) if shot_play.get("situationCode") else 5
                ),
            }

            # Calculate derived features
            shot_data["shot_distance"] = np.sqrt(shot_data["x_coord"] ** 2 + shot_data["y_coord"] ** 2)
            shot_data["shot_angle"] = np.degrees(
                np.arctan2(abs(shot_data["y_coord"]), abs(89 - abs(shot_data["x_coord"])))
            )

            # Power play
            if is_home_team:
                shot_data["is_power_play"] = int(shot_data["home_skaters"] > shot_data["away_skaters"])
            else:
                shot_data["is_power_play"] = int(shot_data["away_skaters"] > shot_data["home_skaters"])

            # Add context
            context = self._get_shot_context(all_plays, shot_index, shot_play)
            shot_data.update(context)

            return shot_data

        except Exception as e:
            logger.debug(f"Error extracting shot: {e}")
            return None

    def _get_shot_context(self, plays: List[Dict], shot_index: int, shot_play: Dict) -> Dict:
        """Get context from previous events"""
        context = {
            "prev_event_type": "",
            "time_since_prev_event": 0,
            "is_rebound": 0,
            "is_rush": 0,
            "zone_time": 0,
        }

        if shot_index > 0:
            prev_play = plays[shot_index - 1]
            context["prev_event_type"] = prev_play.get("typeDescKey", "")

            # Check for rebound
            if context["prev_event_type"] in ["shot-on-goal", "missed-shot"]:
                # Simple time calculation
                context["is_rebound"] = 1  # Would need actual time calc

            # Look for zone entry (simplified)
            for i in range(max(0, shot_index - 5), shot_index):
                if plays[i].get("typeDescKey") == "zone-entry":
                    context["zone_time"] = shot_index - i  # Simplified
                    context["is_rush"] = 1 if (shot_index - i) < 3 else 0
                    break

        return context


def main():
    """Collect NHL data with player names"""
    logger.info("=" * 60)
    logger.info("NHL DATA COLLECTION WITH PLAYER NAMES")
    logger.info("=" * 60)

    collector = ImprovedNHLCollector()

    # Collect last 50 games to test
    collector.collect_with_details(season="20242025", max_games=50)  # Start smaller to test

    logger.info("\nCollection complete!")

    # If successful, you can run again with more games:
    # shots = collector.collect_with_details(season="20242025", max_games=500)


if __name__ == "__main__":
    main()
