# File: data/collect_all_nhl_data.py
"""
Comprehensive NHL data collector that gets EVERYTHING:
- Shots with all players on ice
- Complete pre-shot sequences
- Player stats for everyone
- Proper relationships between all entities
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveNHLCollector:
    """Collect ALL NHL data with proper relationships"""

    def __init__(self, data_dir: str = "data/nhl/complete"):
        self.base_url = "https://api-web.nhle.com/v1"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Storage for all data
        self.all_shots = []
        self.all_players = {}
        self.all_goalies = {}
        self.all_games = {}
        self.player_game_stats = {}
        self.line_combinations = {}
        self.player_relationships = []

        # Track unique entities
        self.unique_player_ids = set()
        self.unique_goalie_ids = set()

        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def collect_complete_season(self, start_date: str, end_date: str, max_games: Optional[int] = None):
        """Main entry point - collect everything for a date range"""
        logger.info(f"Starting comprehensive NHL data collection: {start_date} to {end_date}")

        # Step 1: Get all game IDs
        game_ids = await self.get_all_game_ids(start_date, end_date)

        if max_games:
            game_ids = game_ids[:max_games]

        logger.info(f"Found {len(game_ids)} games to process")

        # Step 2: Process each game completely
        for i, game_id in enumerate(game_ids):
            logger.info(f"\nProcessing game {i + 1}/{len(game_ids)}: {game_id}")
            try:
                await self.process_game_complete(game_id)
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                continue

            # Rate limiting
            await asyncio.sleep(0.5)

        # Step 3: Collect all player season stats
        logger.info("\nCollecting season stats for all players...")
        await self.collect_all_player_stats()

        # Step 4: Save everything
        self.save_complete_dataset()

        return len(self.all_shots)

    async def get_all_game_ids(self, start_date: str, end_date: str) -> List[str]:
        """Get all game IDs for date range"""
        game_ids = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"{self.base_url}/schedule/{date_str}"

            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        if "gameWeek" in data:
                            for day in data["gameWeek"]:
                                for game in day.get("games", []):
                                    if game.get("gameState") in ["OFF", "FINAL"]:
                                        game_ids.append(str(game["id"]))

            except Exception as e:
                logger.error(f"Error fetching games for {date_str}: {e}")

            current_date += timedelta(days=1)

        return game_ids

    async def process_game_complete(self, game_id: str):
        """Process a single game - get EVERYTHING"""

        # Get all game data endpoints
        pbp_data = await self.fetch_play_by_play(game_id)
        boxscore_data = await self.fetch_boxscore(game_id)
        shifts_data = await self.fetch_shifts(game_id)

        if not pbp_data or not boxscore_data:
            logger.warning(f"Missing data for game {game_id}")
            return

        # Store game info
        self.all_games[game_id] = {
            "game_id": game_id,
            "date": pbp_data.get("gameDate", ""),
            "home_team": pbp_data.get("homeTeam", {}).get("abbrev", ""),
            "away_team": pbp_data.get("awayTeam", {}).get("abbrev", ""),
        }

        # Build player roster for this game
        game_roster = self.build_game_roster(boxscore_data)

        # Process each play
        plays = pbp_data.get("plays", [])

        for i, play in enumerate(plays):
            # Process shots and goals
            if play.get("typeDescKey") in ["shot-on-goal", "goal", "missed-shot"]:
                shot_data = await self.extract_complete_shot_data(play, plays, game_id, game_roster, shifts_data, i)
                if shot_data:
                    self.all_shots.append(shot_data)

    async def extract_complete_shot_data(
        self, shot_play: Dict, all_plays: List, game_id: str, roster: Dict, shifts: Optional[Dict], play_idx: int
    ) -> Optional[Dict]:
        """Extract COMPLETE shot data with all context"""

        details = shot_play.get("details", {})

        # Basic shot info
        shot_data = {
            "shot_id": f"{game_id}_{play_idx}",
            "game_id": game_id,
            "period": shot_play.get("periodDescriptor", {}).get("number", 0),
            "time_in_period": self.parse_time(shot_play.get("timeInPeriod", "0:00")),
            "event_type": shot_play.get("typeDescKey", ""),
            "is_goal": 1 if shot_play.get("typeDescKey") == "goal" else 0,
        }

        # Location data
        shot_data["x_coord"] = details.get("xCoord", 0)
        shot_data["y_coord"] = details.get("yCoord", 0)
        shot_data["shot_distance"] = self.calculate_distance(shot_data["x_coord"], shot_data["y_coord"])
        shot_data["shot_angle"] = self.calculate_angle(shot_data["x_coord"], shot_data["y_coord"])
        shot_data["shot_type"] = details.get("shotType", "")

        # CRITICAL: Get actual player IDs
        shooter_id = details.get("shootingPlayerId")
        goalie_id = details.get("goalieInNetId")

        if not shooter_id:
            return None

        shot_data["shooter_id"] = shooter_id
        shot_data["goalie_id"] = goalie_id

        # Track unique players
        self.unique_player_ids.add(shooter_id)
        if goalie_id:
            self.unique_goalie_ids.add(goalie_id)

        # Get player names from roster
        shot_data["shooter_name"] = self.get_player_name(shooter_id, roster)
        shot_data["goalie_name"] = self.get_player_name(goalie_id, roster) if goalie_id else None

        # CRITICAL: Get who was on ice
        on_ice = self.get_players_on_ice(shot_play, shifts, roster)
        shot_data.update(on_ice)

        # Track all players we've seen
        for player_id in on_ice.get("home_players_on_ice_ids", []):
            self.unique_player_ids.add(player_id)
        for player_id in on_ice.get("away_players_on_ice_ids", []):
            self.unique_player_ids.add(player_id)

        # Get pre-shot sequence
        pre_shot = self.extract_pre_shot_sequence(all_plays, play_idx)
        shot_data.update(pre_shot)

        # Game state
        shot_data["home_score"] = details.get("homeScore", 0)
        shot_data["away_score"] = details.get("awayScore", 0)
        shot_data["is_home_team"] = details.get("eventOwnerTeamId") == self.all_games[game_id]["home_team"]

        # Situation
        situation = shot_play.get("situationCode", "1551")
        shot_data["home_skaters"] = int(situation[0])
        shot_data["away_skaters"] = int(situation[1])
        shot_data["is_power_play"] = shot_data["home_skaters"] != shot_data["away_skaters"]

        return shot_data

    def get_players_on_ice(self, play: Dict, shifts: Optional[Dict], roster: Dict) -> Dict:
        """Extract who was actually on ice during the shot"""

        on_ice_data = {}

        # Method 1: From shift data (most accurate)
        if shifts:
            period = play.get("periodDescriptor", {}).get("number", 0)
            time = self.parse_time(play.get("timeInPeriod", "0:00"))

            home_players = []
            away_players = []

            for shift in shifts.get("data", []):
                # Check if player was on ice at this time
                if shift.get("period") == period and shift.get("startTime") <= time <= shift.get("endTime"):

                    player_id = shift.get("playerId")
                    team_id = shift.get("teamId")

                    if team_id == shifts.get("homeTeamId"):
                        home_players.append(player_id)
                    else:
                        away_players.append(player_id)

            on_ice_data["home_players_on_ice_ids"] = home_players[:6]  # Max 6 including goalie
            on_ice_data["away_players_on_ice_ids"] = away_players[:6]

        # Method 2: From play details (backup)
        else:
            # Try to extract from situation code or other fields
            on_ice_data["home_players_on_ice_ids"] = []
            on_ice_data["away_players_on_ice_ids"] = []

        # Get player names
        on_ice_data["home_players_on_ice_names"] = [
            self.get_player_name(pid, roster) for pid in on_ice_data.get("home_players_on_ice_ids", [])
        ]
        on_ice_data["away_players_on_ice_names"] = [
            self.get_player_name(pid, roster) for pid in on_ice_data.get("away_players_on_ice_ids", [])
        ]

        # Create individual columns for easier analysis
        for i in range(6):
            home_ids = on_ice_data.get("home_players_on_ice_ids", [])
            away_ids = on_ice_data.get("away_players_on_ice_ids", [])

            on_ice_data[f"home_player_{i}_id"] = home_ids[i] if i < len(home_ids) else None
            on_ice_data[f"away_player_{i}_id"] = away_ids[i] if i < len(away_ids) else None

        return on_ice_data

    def extract_pre_shot_sequence(self, all_plays: List, shot_idx: int) -> Dict:
        """Extract the sequence of events before the shot"""

        sequence_data = {
            "last_event_type": None,
            "last_event_time_diff": 0,
            "last_event_distance": 0,
            "zone_entries_last_30s": 0,
            "passes_last_30s": 0,
            "shots_last_30s": 0,
            "hits_last_30s": 0,
            "is_rush": 0,
            "is_rebound": 0,
            "is_off_turnover": 0,
            "play_sequence": [],
        }

        if shot_idx == 0:
            return sequence_data

        shot_time = self.parse_time(all_plays[shot_idx].get("timeInPeriod", "0:00"))
        shot_period = all_plays[shot_idx].get("periodDescriptor", {}).get("number", 0)

        # Look back through previous plays
        for i in range(max(0, shot_idx - 20), shot_idx):
            play = all_plays[i]
            play_type = play.get("typeDescKey", "")
            play_time = self.parse_time(play.get("timeInPeriod", "0:00"))
            play_period = play.get("periodDescriptor", {}).get("number", 0)

            # Only look at same period
            if play_period != shot_period:
                continue

            time_diff = shot_time - play_time

            # Add to sequence
            sequence_data["play_sequence"].append(
                {"type": play_type, "time_diff": time_diff, "details": play.get("details", {})}
            )

            # Count events in last 30 seconds
            if time_diff <= 30:
                if play_type == "zone-entry":
                    sequence_data["zone_entries_last_30s"] += 1
                elif play_type in ["pass", "indirect-pass"]:
                    sequence_data["passes_last_30s"] += 1
                elif play_type in ["shot-on-goal", "missed-shot", "blocked-shot"]:
                    sequence_data["shots_last_30s"] += 1
                elif play_type == "hit":
                    sequence_data["hits_last_30s"] += 1

            # Check last event
            if i == shot_idx - 1:
                sequence_data["last_event_type"] = play_type
                sequence_data["last_event_time_diff"] = time_diff

                # Calculate distance
                last_x = play.get("details", {}).get("xCoord", 0)
                last_y = play.get("details", {}).get("yCoord", 0)
                shot_x = all_plays[shot_idx].get("details", {}).get("xCoord", 0)
                shot_y = all_plays[shot_idx].get("details", {}).get("yCoord", 0)

                sequence_data["last_event_distance"] = np.sqrt((shot_x - last_x) ** 2 + (shot_y - last_y) ** 2)

        # Determine shot types
        if sequence_data["zone_entries_last_30s"] > 0 and sequence_data["last_event_time_diff"] < 10:
            sequence_data["is_rush"] = 1

        if (
            sequence_data["last_event_type"] in ["shot-on-goal", "missed-shot"]
            and sequence_data["last_event_time_diff"] < 3
        ):
            sequence_data["is_rebound"] = 1

        if sequence_data["last_event_type"] in ["turnover", "takeaway"] and sequence_data["last_event_time_diff"] < 5:
            sequence_data["is_off_turnover"] = 1

        return sequence_data

    async def collect_all_player_stats(self):
        """Collect season stats for all unique players"""

        all_player_ids = self.unique_player_ids.union(self.unique_goalie_ids)
        logger.info(f"Collecting stats for {len(all_player_ids)} unique players")

        for i, player_id in enumerate(all_player_ids):
            if i % 50 == 0:
                logger.info(f"  Progress: {i}/{len(all_player_ids)}")

            try:
                stats = await self.fetch_player_season_stats(player_id)
                if stats:
                    self.all_players[player_id] = stats

                    # Separate goalie stats
                    if player_id in self.unique_goalie_ids:
                        self.all_goalies[player_id] = stats

            except Exception as e:
                logger.error(f"Error fetching stats for player {player_id}: {e}")

            await asyncio.sleep(0.1)  # Rate limiting

    def save_complete_dataset(self):
        """Save all collected data in organized format"""

        logger.info("\nSaving complete dataset...")

        # 1. Shots with all context
        shots_df = pd.DataFrame(self.all_shots)
        shots_file = self.data_dir / f"complete_shots_{datetime.now().strftime('%Y%m%d')}.csv"
        shots_df.to_csv(shots_file, index=False)
        logger.info(f"Saved {len(shots_df)} shots to {shots_file}")

        # 2. Player stats
        if self.all_players:
            players_df = pd.DataFrame.from_dict(self.all_players, orient="index")
            players_file = self.data_dir / f"all_players_{datetime.now().strftime('%Y%m%d')}.csv"
            players_df.to_csv(players_file, index=False)
            logger.info(f"Saved {len(players_df)} player stats to {players_file}")

        # 3. Goalie-specific stats
        if self.all_goalies:
            goalies_df = pd.DataFrame.from_dict(self.all_goalies, orient="index")
            goalies_file = self.data_dir / f"all_goalies_{datetime.now().strftime('%Y%m%d')}.csv"
            goalies_df.to_csv(goalies_file, index=False)
            logger.info(f"Saved {len(goalies_df)} goalie stats to {goalies_file}")

        # 4. Game metadata
        games_df = pd.DataFrame.from_dict(self.all_games, orient="index")
        games_file = self.data_dir / f"games_metadata_{datetime.now().strftime('%Y%m%d')}.csv"
        games_df.to_csv(games_file, index=False)

        # 5. Create player relationships (who plays with whom)
        self.save_player_relationships(shots_df)

        # 6. Save summary statistics
        self.save_collection_summary()

    def save_player_relationships(self, shots_df: pd.DataFrame):
        """Create and save player relationship data"""

        relationships = []

        # For each shot, create relationships between players on ice
        for _, shot in shots_df.iterrows():
            # Get all home players
            home_players = []
            for i in range(6):
                player_id = shot.get(f"home_player_{i}_id")
                if player_id and pd.notna(player_id):
                    home_players.append(player_id)

            # Create pairwise relationships
            for i, p1 in enumerate(home_players):
                for p2 in home_players[i + 1 :]:
                    relationships.append(
                        {
                            "player_1": p1,
                            "player_2": p2,
                            "team": "home",
                            "game_id": shot["game_id"],
                            "situation": f"{shot['home_skaters']}v{shot['away_skaters']}",
                        }
                    )

        relationships_df = pd.DataFrame(relationships)
        relationships_file = self.data_dir / f"player_relationships_{datetime.now().strftime('%Y%m%d')}.csv"
        relationships_df.to_csv(relationships_file, index=False)
        logger.info(f"Saved {len(relationships_df)} player relationships")

    def save_collection_summary(self):
        """Save summary of what was collected"""

        summary = {
            "collection_date": datetime.now().isoformat(),
            "total_games": len(self.all_games),
            "total_shots": len(self.all_shots),
            "total_goals": sum(1 for s in self.all_shots if s.get("is_goal")),
            "total_players": len(self.all_players),
            "total_goalies": len(self.all_goalies),
            "unique_shooters": len(self.unique_player_ids),
            "shot_types": pd.DataFrame(self.all_shots)["shot_type"].value_counts().to_dict() if self.all_shots else {},
        }

        summary_file = self.data_dir / f"collection_summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\nCollection Summary:")
        for key, value in summary.items():
            if key != "shot_types":
                logger.info(f"  {key}: {value}")

    # Helper methods for API calls
    async def fetch_play_by_play(self, game_id: str) -> Optional[Dict]:
        """Fetch play-by-play data"""
        url = f"{self.base_url}/gamecenter/{game_id}/play-by-play"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error fetching PBP for {game_id}: {e}")
        return None

    async def fetch_boxscore(self, game_id: str) -> Optional[Dict]:
        """Fetch boxscore data"""
        url = f"{self.base_url}/gamecenter/{game_id}/boxscore"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error fetching boxscore for {game_id}: {e}")
        return None

    async def fetch_shifts(self, game_id: str) -> Optional[Dict]:
        """Fetch shift data"""
        url = f"{self.base_url}/gamecenter/{game_id}/shifts"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception:
            # Shifts might not be available for all games
            pass
        return None

    async def fetch_player_season_stats(self, player_id: str) -> Optional[Dict]:
        """Fetch player season stats"""
        url = f"{self.base_url}/player/{player_id}/landing"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract relevant stats
                    stats = {
                        "player_id": player_id,
                        "name": data.get("fullName", ""),
                        "position": data.get("position", ""),
                        "team": data.get("currentTeamAbbrev", ""),
                    }

                    # Get current season stats
                    if "featuredStats" in data:
                        season_stats = data["featuredStats"].get("regularSeason", {}).get("career", {})
                        stats.update(
                            {
                                "games_played": season_stats.get("gamesPlayed", 0),
                                "goals": season_stats.get("goals", 0),
                                "assists": season_stats.get("assists", 0),
                                "shots": season_stats.get("shots", 0),
                                "shooting_pct": season_stats.get("shootingPctg", 0),
                                "save_pct": season_stats.get("savePctg", 0),  # For goalies
                                "goals_against_avg": season_stats.get("goalsAgainstAverage", 0),
                            }
                        )

                    return stats

        except Exception as e:
            logger.error(f"Error fetching stats for {player_id}: {e}")
        return None

    def build_game_roster(self, boxscore: Dict) -> Dict:
        """Build roster mapping from boxscore"""
        roster = {}

        for team in ["homeTeam", "awayTeam"]:
            if team in boxscore:
                # Players are usually under 'players' key
                players = boxscore[team].get("players", {})

                for player_key, player_data in players.items():
                    # Extract player ID (might be in different formats)
                    player_id = None
                    if "playerId" in player_data:
                        player_id = str(player_data["playerId"])
                    elif player_key.startswith("ID"):
                        player_id = player_key[2:]  # Remove 'ID' prefix
                    else:
                        player_id = player_key

                    if player_id:
                        roster[player_id] = {
                            "name": player_data.get("name", {}).get("default", ""),
                            "position": player_data.get("position", ""),
                            "team": boxscore[team].get("abbrev", ""),
                        }

        return roster

    def get_player_name(self, player_id: str, roster: Dict) -> Optional[str]:
        """Get player name from roster"""
        if player_id and str(player_id) in roster:
            return roster[str(player_id)].get("name", f"Player_{player_id}")
        return f"Player_{player_id}" if player_id else None

    @staticmethod
    def parse_time(time_str: str) -> float:
        """Convert MM:SS to seconds"""
        try:
            parts = time_str.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return 0

    @staticmethod
    def calculate_distance(x: float, y: float) -> float:
        """Calculate distance from net"""
        goal_x = 89 if x > 0 else -89
        return np.sqrt((goal_x - x) ** 2 + y**2)

    @staticmethod
    def calculate_angle(x: float, y: float) -> float:
        """Calculate shot angle"""
        goal_x = 89 if x > 0 else -89
        return abs(np.degrees(np.arctan2(y, goal_x - x)))


async def main():
    """Run the complete data collection"""

    # Configuration
    start_date = "2024-10-01"  # Season start
    end_date = "2025-04-15"  # Regular season end
    max_games = None  # Set to small number for testing

    logger.info("=" * 60)
    logger.info("NHL COMPREHENSIVE DATA COLLECTION")
    logger.info("=" * 60)

    async with ComprehensiveNHLCollector() as collector:
        total_shots = await collector.collect_complete_season(start_date, end_date, max_games)

    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Total shots collected: {total_shots}")
    logger.info("Check data/nhl/complete/ for all files")


if __name__ == "__main__":
    asyncio.run(main())
