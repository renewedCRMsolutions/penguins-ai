# File: data/collect_live_nhl_data.py
"""
Enhanced NHL data collection pipeline using nhl-api-py
Collects rich shot context data including pre-shot events, goalie workload, and game flow
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
from nhlpy import NHLClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHLDataCollector:
    """Collect enhanced shot data with full context from NHL API"""

    def __init__(self, data_dir: str = "data/nhl/enhanced"):
        self.client = NHLClient(verbose=False)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache for player/goalie stats to avoid repeated API calls
        self.player_cache = {}
        self.goalie_cache = {}

    def collect_date_range(self, start_date: str, end_date: str, game_types: List[int] = [2]):
        """
        Collect all games in date range
        game_types: 1=preseason, 2=regular season, 3=playoffs
        """
        all_shots = []
        all_games = []

        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            logger.info(f"Processing games for {date_str}")

            try:
                # Get schedule for date
                schedule = self.client.schedule.get_schedule(date=date_str)

                if schedule and "gameWeek" in schedule:
                    for day in schedule["gameWeek"]:
                        if "games" in day:
                            for game in day["games"]:
                                if game.get("gameType") in game_types:
                                    game_shots = self.process_game(game["id"])
                                    if game_shots:
                                        all_shots.extend(game_shots)
                                        all_games.append(game)

            except Exception as e:
                logger.error(f"Error processing {date_str}: {e}")

            current += timedelta(days=1)

        # Save collected data
        if all_shots:
            shots_df = pd.DataFrame(all_shots)
            shots_df.to_csv(self.data_dir / f"enhanced_shots_{start_date}_to_{end_date}.csv", index=False)
            logger.info(f"Saved {len(shots_df)} shots to file")

            games_df = pd.DataFrame(all_games)
            games_df.to_csv(self.data_dir / f"games_{start_date}_to_{end_date}.csv", index=False)

        return all_shots

    def process_game(self, game_id: int) -> List[Dict]:
        """Process a single game and extract enhanced shot data"""
        try:
            # Get play-by-play data
            pbp = self.client.game_center.play_by_play(game_id=str(game_id))
            boxscore = self.client.game_center.boxscore(game_id=str(game_id))

            if not pbp or "plays" not in pbp:
                return []

            plays = pbp["plays"]
            shots = []

            # Track game state
            game_state = {
                "home_goals": 0,
                "away_goals": 0,
                "period": 1,
                "time_remaining": 1200,  # 20 minutes in seconds
                "home_goalie_shots": 0,
                "away_goalie_shots": 0,
                "last_10_events": [],
                "momentum": 0,  # Track momentum swings
            }

            # Get goalie info from boxscore
            goalie_info = self._extract_goalie_info(boxscore)

            for i, play in enumerate(plays):
                event_type = play.get("typeDescKey", "")

                # Update game state
                self._update_game_state(play, game_state)

                # Process shot events
                if event_type in ["shot-on-goal", "goal"]:
                    shot_data = self._process_shot(play, plays, i, game_state, goalie_info, game_id)
                    if shot_data:
                        shots.append(shot_data)

            return shots

        except Exception as e:
            logger.error(f"Error processing game {game_id}: {e}")
            return []

    def _process_shot(
        self, shot_play: Dict, all_plays: List[Dict], shot_index: int, game_state: Dict, goalie_info: Dict, game_id: int
    ) -> Optional[Dict]:
        """Extract enhanced features for a single shot"""
        try:
            shot_data = {
                "game_id": game_id,
                "event_id": shot_play.get("eventId"),
                "period": shot_play.get("periodDescriptor", {}).get("number", 0),
                "time": shot_play.get("timeInPeriod", "00:00"),
                "is_goal": shot_play.get("typeDescKey") == "goal",
                # Basic shot info
                "shooter_id": shot_play.get("details", {}).get("shootingPlayerId"),
                "shooter_name": self._get_player_name(shot_play),
                "shot_type": shot_play.get("details", {}).get("shotType", "Unknown"),
                "x_coord": shot_play.get("details", {}).get("xCoord", 0),
                "y_coord": shot_play.get("details", {}).get("yCoord", 0),
                # Enhanced features
                "score_differential": self._get_score_diff(shot_play, game_state),
                "time_remaining": self._get_time_remaining(shot_play),
                "is_home_team": shot_play.get("details", {}).get("eventOwnerTeamId") == shot_play.get("homeTeamId"),
                # Goalie state
                "goalie_id": shot_play.get("details", {}).get("goalieInNetId"),
                "goalie_shots_faced": self._get_goalie_workload(shot_play, game_state),
                "goalie_time_since_last_shot": 0,  # Will calculate
                # Pre-shot context (last 30 seconds)
                **self._extract_pre_shot_context(all_plays, shot_index, shot_play),
                # Game flow
                "momentum_score": game_state.get("momentum", 0),
                "is_power_play": self._check_power_play(shot_play),
                "is_rush": False,  # Will calculate
                "zone_time": 0,  # Will calculate
            }

            # Add advanced metrics
            shot_data.update(self._calculate_advanced_metrics(shot_data, all_plays, shot_index))

            return shot_data

        except Exception as e:
            logger.error(f"Error processing shot: {e}")
            return None

    def _extract_pre_shot_context(self, plays: List[Dict], shot_index: int, shot_play: Dict) -> Dict:
        """Extract events in 30-second window before shot"""
        context = {
            "passes_before_shot": 0,
            "zone_entries": 0,
            "hits_for": 0,
            "hits_against": 0,
            "faceoff_win": None,
            "time_since_zone_entry": 0,
            "time_since_faceoff": 0,
            "shot_attempts_sequence": 0,
            "rebounds": 0,
        }

        shot_time = self._parse_time(shot_play.get("timeInPeriod", "00:00"))
        shot_period = shot_play.get("periodDescriptor", {}).get("number", 1)
        shooting_team = shot_play.get("details", {}).get("eventOwnerTeamId")

        # Look back through previous plays (up to 30 seconds)
        for i in range(shot_index - 1, max(0, shot_index - 50), -1):
            play = plays[i]
            play_time = self._parse_time(play.get("timeInPeriod", "00:00"))
            play_period = play.get("periodDescriptor", {}).get("number", 1)

            # Stop if different period or more than 30 seconds ago
            if play_period != shot_period:
                break
            if shot_time - play_time > 30:
                break

            event_type = play.get("typeDescKey", "")
            event_team = play.get("details", {}).get("eventOwnerTeamId")

            # Count relevant events
            if event_team == shooting_team:
                if "pass" in event_type.lower():
                    context["passes_before_shot"] += 1
                elif event_type == "zone-entry":
                    context["zone_entries"] += 1
                    context["time_since_zone_entry"] = shot_time - play_time
                elif event_type == "hit":
                    context["hits_for"] += 1
                elif event_type in ["shot-on-goal", "missed-shot", "blocked-shot"]:
                    context["shot_attempts_sequence"] += 1
                elif event_type == "faceoff":
                    # win_team = play.get("details", {}).get("winningPlayerId")
                    context["faceoff_win"] = event_team == shooting_team
                    context["time_since_faceoff"] = shot_time - play_time
            else:
                if event_type == "hit":
                    context["hits_against"] += 1

        # Determine if rush chance
        context["is_rush"] = context["time_since_zone_entry"] < 5 and context["passes_before_shot"] < 2

        return context

    def _calculate_advanced_metrics(self, shot_data: Dict, plays: List[Dict], shot_index: int) -> Dict:
        """Calculate advanced shot quality metrics"""
        metrics = {}

        # Shot distance and angle
        x, y = shot_data["x_coord"], shot_data["y_coord"]

        # Assuming offensive zone extends from x=25 to x=89 (NHL rink coordinates)
        # Goal line is at x=89
        goal_x = 89 if x > 0 else -89

        distance = np.sqrt((goal_x - x) ** 2 + y**2)
        angle = np.arctan2(abs(y), abs(goal_x - x)) * 180 / np.pi

        metrics["shot_distance"] = distance
        metrics["shot_angle"] = angle

        # Danger zone classification
        if distance < 15 and angle < 30:
            metrics["danger_zone"] = "high"
        elif distance < 30 and angle < 45:
            metrics["danger_zone"] = "medium"
        else:
            metrics["danger_zone"] = "low"

        # Shot quality factors
        metrics["is_one_timer"] = shot_data["passes_before_shot"] == 1
        metrics["is_rebound"] = self._check_rebound(plays, shot_index)
        metrics["traffic_score"] = self._calculate_traffic(plays, shot_index)

        return metrics

    def _get_goalie_workload(self, shot_play: Dict, game_state: Dict) -> int:
        """Get number of shots goalie has faced"""
        is_home = shot_play.get("details", {}).get("eventOwnerTeamId") == shot_play.get("homeTeamId")
        return game_state["away_goalie_shots"] if is_home else game_state["home_goalie_shots"]

    def _update_game_state(self, play: Dict, game_state: Dict):
        """Update running game state"""
        event_type = play.get("typeDescKey", "")

        # Update period and time
        game_state["period"] = play.get("periodDescriptor", {}).get("number", 1)
        time_str = play.get("timeInPeriod", "20:00")
        game_state["time_remaining"] = self._parse_time_remaining(time_str)

        # Update goals
        if event_type == "goal":
            is_home = play.get("details", {}).get("eventOwnerTeamId") == play.get("homeTeamId")
            if is_home:
                game_state["home_goals"] += 1
                game_state["momentum"] += 2
            else:
                game_state["away_goals"] += 1
                game_state["momentum"] -= 2

        # Update shot counts for goalie workload
        if event_type in ["shot-on-goal", "goal"]:
            is_home = play.get("details", {}).get("eventOwnerTeamId") == play.get("homeTeamId")
            if is_home:
                game_state["away_goalie_shots"] += 1
            else:
                game_state["home_goalie_shots"] += 1

        # Track momentum (decay over time)
        game_state["momentum"] *= 0.98

        # Update last 10 events for pattern detection
        game_state["last_10_events"].append(event_type)
        if len(game_state["last_10_events"]) > 10:
            game_state["last_10_events"].pop(0)

    def _parse_time(self, time_str: str) -> float:
        """Convert MM:SS to seconds"""
        if ":" in time_str:
            minutes, seconds = time_str.split(":")
            return float(minutes) * 60 + float(seconds)
        return 0

    def _parse_time_remaining(self, time_str: str) -> float:
        """Convert time to seconds remaining in period"""
        elapsed = self._parse_time(time_str)
        return max(0, 1200 - elapsed)  # 20 minutes per period

    def _check_rebound(self, plays: List[Dict], shot_index: int) -> bool:
        """Check if shot is a rebound (within 3 seconds of previous shot)"""
        if shot_index <= 0:
            return False

        current_shot = plays[shot_index]
        current_time = self._parse_time(current_shot.get("timeInPeriod", "00:00"))
        current_period = current_shot.get("periodDescriptor", {}).get("number", 1)

        # Check previous play
        prev_play = plays[shot_index - 1]
        if prev_play.get("typeDescKey") in ["shot-on-goal", "missed-shot"]:
            prev_time = self._parse_time(prev_play.get("timeInPeriod", "00:00"))
            prev_period = prev_play.get("periodDescriptor", {}).get("number", 1)

            if current_period == prev_period and (current_time - prev_time) <= 3:
                return True

        return False

    def _calculate_traffic(self, plays: List[Dict], shot_index: int) -> int:
        """Estimate traffic in front of net based on recent hits/battles"""
        traffic_score = 0
        shot_time = self._parse_time(plays[shot_index].get("timeInPeriod", "00:00"))
        shot_period = plays[shot_index].get("periodDescriptor", {}).get("number", 1)

        # Look at last 10 seconds of play
        for i in range(shot_index - 1, max(0, shot_index - 20), -1):
            play = plays[i]
            play_time = self._parse_time(play.get("timeInPeriod", "00:00"))
            play_period = play.get("periodDescriptor", {}).get("number", 1)

            if play_period != shot_period or shot_time - play_time > 10:
                break

            event_type = play.get("typeDescKey", "")
            if event_type in ["hit", "penalty", "stoppage"]:
                traffic_score += 1

        return min(traffic_score, 5)  # Cap at 5

    def _check_power_play(self, play: Dict) -> bool:
        """Check if shot occurred during power play"""
        situation = play.get("situationCode", "1551")
        # Situation codes: first two digits are home skaters, last two are away
        # 5v4, 5v3, 4v3 are power plays
        if len(situation) == 4:
            home_skaters = int(situation[:2])
            away_skaters = int(situation[2:])
            return abs(home_skaters - away_skaters) >= 1
        return False

    def _get_score_diff(self, play: Dict, game_state: Dict) -> int:
        """Get score differential from shooting team's perspective"""
        is_home = play.get("details", {}).get("eventOwnerTeamId") == play.get("homeTeamId")
        if is_home:
            return game_state["home_goals"] - game_state["away_goals"]
        else:
            return game_state["away_goals"] - game_state["home_goals"]

    def _get_time_remaining(self, play: Dict) -> float:
        """Get time remaining in game (seconds)"""
        period = play.get("periodDescriptor", {}).get("number", 1)
        time_in_period = self._parse_time_remaining(play.get("timeInPeriod", "20:00"))

        if period <= 3:
            return (3 - period) * 1200 + time_in_period
        else:
            # Overtime
            return time_in_period

    def _extract_goalie_info(self, boxscore: Dict) -> Dict:
        """Extract goalie information from boxscore"""
        goalie_info = {}

        try:
            for team in ["homeTeam", "awayTeam"]:
                if team in boxscore:
                    goalies = boxscore[team].get("goalies", [])
                    for goalie_id in goalies:
                        # Store goalie info for quick lookup
                        goalie_info[goalie_id] = {"team": team, "shots_faced": 0, "saves": 0}
        except Exception:
            pass

        return goalie_info

    def _get_player_name(self, play: Dict) -> str:
        """Extract player name from play data"""
        details = play.get("details", {})

        # Try different fields where player name might be
        for field in ["shootingPlayerName", "playerName", "eventOwnerName"]:
            if field in details:
                return details[field]

        return "Unknown"

    def collect_goalie_season_stats(self, season: str = "20242025"):
        """Collect goalie stats for the season"""
        logger.info(f"Collecting goalie stats for season {season}")

        try:
            # Get all goalies stats
            goalie_stats = self.client.stats.goalie_stats_summary_simple(
                start_season=season, end_season=season, stats_type="summary"
            )

            if goalie_stats and isinstance(goalie_stats, dict) and "data" in goalie_stats:
                df = pd.DataFrame(goalie_stats.get("data", []))
                df.to_csv(self.data_dir / f"goalie_stats_{season}.csv", index=False)
                logger.info(f"Saved {len(df)} goalie records")

            return goalie_stats

        except Exception as e:
            logger.error(f"Error collecting goalie stats: {e}")
            return None

    def collect_shooter_recent_performance(self, player_id: str, season: str = "20242025", last_n_games: int = 10):
        """Get shooter's recent performance metrics"""
        try:
            # Get player game log
            game_log = self.client.stats.player_game_log(player_id=player_id, season_id=season, game_type=2)

            if game_log and len(game_log) >= last_n_games:
                recent_games = game_log[-last_n_games:]

                metrics = {
                    "goals_last_10": sum(g.get("goals", 0) for g in recent_games),
                    "shots_last_10": sum(g.get("shots", 0) for g in recent_games),
                    "shooting_pct_last_10": 0,
                    "hot_streak": False,
                }

                if metrics["shots_last_10"] > 0:
                    metrics["shooting_pct_last_10"] = metrics["goals_last_10"] / metrics["shots_last_10"]

                # Hot streak if 3+ goals in last 5 games
                metrics["hot_streak"] = sum(g.get("goals", 0) for g in recent_games[-5:]) >= 3

                return metrics

        except Exception as e:
            logger.error(f"Error getting shooter performance: {e}")

        return None


# Example usage
if __name__ == "__main__":
    collector = NHLDataCollector()

    # Collect last 30 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    shots = collector.collect_date_range(
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), game_types=[2]  # Regular season only
    )

    # Also collect goalie season stats
    collector.collect_goalie_season_stats()

    logger.info(f"Collection complete. Total shots: {len(shots)}")
