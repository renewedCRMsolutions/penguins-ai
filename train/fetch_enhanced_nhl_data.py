# File: train/fetch_enhanced_nhl_data.py
"""
Fetch NHL data with enhanced features including shot speed, player tracking, and pre-shot events
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHLEnhancedDataFetcher:
    """Fetch enhanced NHL data including shot speeds and tracking data"""

    def __init__(self):
        self.base_url = "https://api-web.nhle.com/v1"
        self.session = None
        self.all_shots = []
        self.player_cache = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_game_ids(self, start_date: str, end_date: str) -> List[int]:
        """Fetch all game IDs between dates"""
        logger.info(f"Fetching games from {start_date} to {end_date}")

        game_ids = []
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"{self.base_url}/score/{date_str}"

            try:
                async with self.session.get(url) as response:  # type: ignore
                    if response.status == 200:
                        data = await response.json()

                        for game in data.get("games", []):
                            if game.get("gameType") == 2:  # Regular season only
                                game_ids.append(game["id"])

            except Exception as e:
                logger.error(f"Error fetching games for {date_str}: {e}")

            current_date += timedelta(days=1)

        logger.info(f"Found {len(game_ids)} games")
        return game_ids

    async def fetch_enhanced_play_by_play(self, game_id: int) -> Dict:
        """Fetch enhanced play-by-play data with shot speeds"""
        url = f"{self.base_url}/gamecenter/{game_id}/play-by-play"

        try:
            async with self.session.get(url) as response:  # type: ignore
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error fetching PBP for game {game_id}: {e}")

        return {}

    def extract_shot_features(self, play_data: Dict, game_id: int) -> List[Dict]:
        """Extract shot features including speed from play-by-play data"""
        shots = []

        if not play_data or "plays" not in play_data:
            return shots

        # Get game info
        game_info = play_data.get("gameDate", {})
        home_team = play_data.get("homeTeam", {}).get("abbrev", "")
        away_team = play_data.get("awayTeam", {}).get("abbrev", "")

        # Process each play
        for play in play_data.get("plays", []):
            if play.get("typeDescKey") in ["shot-on-goal", "goal"]:
                shot_data = self._extract_single_shot(play, game_id, home_team, away_team)
                if shot_data:
                    shots.append(shot_data)

        return shots

    def _extract_single_shot(self, play: Dict, game_id: int, home_team: str, away_team: str) -> Optional[Dict]:
        """Extract features from a single shot"""
        try:
            # Basic info
            shot = {
                "game_id": game_id,
                "event_idx": play.get("eventId", 0),
                "period": play.get("periodDescriptor", {}).get("number", 1),
                "time_in_period": self._parse_time(play.get("timeInPeriod", "00:00")),
                "shot_type": play.get("typeDescKey", ""),
                "is_goal": 1 if play.get("typeDescKey") == "goal" else 0,
            }

            # Location data
            details = play.get("details", {})
            shot["x_coord"] = details.get("xCoord", 0)
            shot["y_coord"] = details.get("yCoord", 0)
            shot["zone"] = details.get("zoneCode", "")

            # Calculate distance and angle
            shot["distance"] = self._calculate_distance(shot["x_coord"], shot["y_coord"])
            shot["angle"] = self._calculate_angle(shot["x_coord"], shot["y_coord"])

            # Shot details
            shot["shot_type_detail"] = details.get("shotType", "")

            # ENHANCED FEATURES - This is where shot speed might be
            shot["shot_speed"] = details.get("shotSpeed", None)
            shot["shot_velocity"] = details.get("velocity", None)

            # Player info
            shooter = details.get("shootingPlayer", {})
            shot["shooter_id"] = shooter.get("playerId", 0)
            shot["shooter_name"] = shooter.get("name", {}).get("default", "")
            shot["shooter_position"] = shooter.get("position", "")

            goalie = details.get("goalie", {})
            shot["goalie_id"] = goalie.get("playerId", 0)
            shot["goalie_name"] = goalie.get("name", {}).get("default", "")

            # Team info
            shot["shooting_team"] = details.get("eventOwnerTeamId", "")
            shot["home_team"] = home_team
            shot["away_team"] = away_team
            shot["is_home_team"] = 1 if shot["shooting_team"] == home_team else 0

            # Situation
            situation = play.get("situationCode", "0000")
            shot["home_skaters"] = int(situation[0])
            shot["away_skaters"] = int(situation[1])
            shot["strength"] = self._determine_strength(
                shot["home_skaters"], shot["away_skaters"], bool(shot["is_home_team"])
            )

            # Score state
            shot["home_score"] = details.get("homeScore", 0)
            shot["away_score"] = details.get("awayScore", 0)
            shot["score_diff"] = (
                shot["home_score"] - shot["away_score"]
                if shot["is_home_team"]
                else shot["away_score"] - shot["home_score"]
            )

            return shot

        except Exception as e:
            logger.error(f"Error extracting shot: {e}")
            return None

    def _parse_time(self, time_str: str) -> float:
        """Convert MM:SS to seconds"""
        try:
            mins, secs = map(int, time_str.split(":"))
            return mins * 60 + secs
        except Exception:
            return 0

    def _calculate_distance(self, x: float, y: float) -> float:
        """Calculate distance from net (assuming net at x=89)"""
        net_x = 89
        return np.sqrt((net_x - abs(x)) ** 2 + y**2)

    def _calculate_angle(self, x: float, y: float) -> float:
        """Calculate shot angle"""
        net_x = 89
        return abs(np.arctan2(y, net_x - abs(x)) * 180 / np.pi)

    def _determine_strength(self, home_skaters: int, away_skaters: int, is_home: bool) -> str:
        """Determine strength situation"""
        if home_skaters == away_skaters:
            return "EV"
        elif (is_home and home_skaters > away_skaters) or (not is_home and away_skaters > home_skaters):
            return "PP"
        else:
            return "SH"

    async def fetch_pre_shot_events(self, play_data: Dict, shot_event_id: int) -> Dict:
        """Extract events leading up to the shot"""
        pre_shot_features = {
            "last_event_type": None,
            "time_since_last_event": None,
            "zone_entries_last_minute": 0,
            "passes_last_30_seconds": 0,
            "is_rush": 0,
            "is_rebound": 0,
            "is_off_turnover": 0,
        }

        # Find shot index
        shot_idx = None
        plays = play_data.get("plays", [])

        for i, play in enumerate(plays):
            if play.get("eventId") == shot_event_id:
                shot_idx = i
                break

        if shot_idx is None:
            return pre_shot_features

        # Look at previous 10 events
        lookback = min(shot_idx, 10)
        recent_events = plays[shot_idx - lookback : shot_idx]

        if recent_events:
            # Last event
            last_event = recent_events[-1]
            pre_shot_features["last_event_type"] = last_event.get("typeDescKey", "")

            # Check for rush (zone entry within 10 seconds)
            for event in reversed(recent_events):
                if event.get("typeDescKey") == "zone-entry":
                    entry_time = self._parse_time(event.get("timeInPeriod", "00:00"))
                    shot_time = self._parse_time(plays[shot_idx].get("timeInPeriod", "00:00"))
                    if abs(shot_time - entry_time) < 10:
                        pre_shot_features["is_rush"] = 1
                    break

            # Check for rebound
            if pre_shot_features["last_event_type"] in ["shot-on-goal", "blocked-shot"]:
                pre_shot_features["is_rebound"] = 1

            # Check for turnover
            if pre_shot_features["last_event_type"] in ["turnover", "takeaway"]:
                pre_shot_features["is_off_turnover"] = 1

        return pre_shot_features

    async def fetch_player_stats(self, player_id: int, season: str = "20242025") -> Dict:
        """Fetch player statistics for quality metrics"""
        if player_id in self.player_cache:
            return self.player_cache[player_id]

        url = f"{self.base_url}/player/{player_id}/landing"

        try:
            async with self.session.get(url) as response:  # type: ignore
                if response.status == 200:
                    data = await response.json()

                    # Extract relevant stats
                    stats = {
                        "shooting_pct": data.get("careerTotals", {}).get("regularSeason", {}).get("shootingPct", 0),
                        "goals_per_game": data.get("careerTotals", {}).get("regularSeason", {}).get("goalsPerGame", 0),
                        "shots_per_game": data.get("careerTotals", {}).get("regularSeason", {}).get("shotsPerGame", 0),
                    }

                    self.player_cache[player_id] = stats
                    return stats

        except Exception as e:
            logger.error(f"Error fetching player stats for {player_id}: {e}")

        return {"shooting_pct": 0, "goals_per_game": 0, "shots_per_game": 0}

    async def process_game(self, game_id: int) -> List[Dict]:
        """Process a single game and extract all shots with features"""
        logger.info(f"Processing game {game_id}")

        # Fetch play-by-play
        pbp_data = await self.fetch_enhanced_play_by_play(game_id)
        if not pbp_data:
            return []

        # Extract shots
        shots = self.extract_shot_features(pbp_data, game_id)

        # Enhance with pre-shot events
        for shot in shots:
            pre_shot = await self.fetch_pre_shot_events(pbp_data, shot["event_idx"])
            shot.update(pre_shot)

            # Add player quality metrics
            if shot["shooter_id"]:
                player_stats = await self.fetch_player_stats(shot["shooter_id"])
                shot["shooter_career_shooting_pct"] = player_stats["shooting_pct"]
                shot["shooter_goals_per_game"] = player_stats["goals_per_game"]

        return shots

    async def fetch_all_data(self, start_date: str, end_date: str, max_games: Optional[int] = None):
        """Main method to fetch all data"""
        # Get game IDs
        game_ids = await self.fetch_game_ids(start_date, end_date)

        if max_games:
            game_ids = game_ids[:max_games]

        logger.info(f"Processing {len(game_ids)} games")

        # Process games in batches
        batch_size = 10
        all_shots = []

        for i in range(0, len(game_ids), batch_size):
            batch = game_ids[i : i + batch_size]
            tasks = [self.process_game(game_id) for game_id in batch]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for shots in results:
                if isinstance(shots, list):
                    all_shots.extend(shots)

            logger.info(f"Processed {min(i + batch_size, len(game_ids))}/{len(game_ids)} games")

        self.all_shots = all_shots
        logger.info(f"Total shots collected: {len(all_shots)}")

        return all_shots

    def save_data(self, output_dir: str = "data/nhl"):
        """Save collected data"""
        os.makedirs(output_dir, exist_ok=True)

        if self.all_shots:
            # Save raw data
            df = pd.DataFrame(self.all_shots)

            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save as CSV
            csv_path = os.path.join(output_dir, f"shots_enhanced_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} shots to {csv_path}")

            # Save as JSON for inspection
            json_path = os.path.join(output_dir, f"shots_enhanced_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump(self.all_shots[:10], f, indent=2)  # Sample for inspection
            logger.info(f"Saved sample to {json_path}")

            # Print summary
            self.print_summary(df)

    def print_summary(self, df: pd.DataFrame):
        """Print data summary"""
        logger.info("\n" + "=" * 60)
        logger.info("DATA COLLECTION SUMMARY")
        logger.info("=" * 60)

        logger.info(f"Total shots: {len(df)}")
        logger.info(f"Goals: {df['is_goal'].sum()} ({df['is_goal'].mean() * 100:.1f}%)")

        # Check for shot speed data
        if "shot_speed" in df.columns and df["shot_speed"].notna().any():
            logger.info("\n✅ SHOT SPEED DATA FOUND!")
            logger.info(f"Shots with speed: {df['shot_speed'].notna().sum()}")
            logger.info(f"Avg shot speed: {df['shot_speed'].mean():.1f}")
            logger.info(f"Max shot speed: {df['shot_speed'].max():.1f}")
        else:
            logger.warning("\n❌ No shot speed data found in API response")

        # Feature availability
        logger.info("\nFeature availability:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            logger.info(f"  {col}: {non_null}/{len(df)} ({non_null / len(df) * 100:.1f}%)")


async def main():
    """Run the enhanced data fetcher"""
    # Configure dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    logger.info("🏒 NHL ENHANCED DATA FETCHER")
    logger.info("=" * 60)
    logger.info(f"Fetching data from {start_date} to {end_date}")

    async with NHLEnhancedDataFetcher() as fetcher:
        # Fetch all data
        await fetcher.fetch_all_data(start_date, end_date, max_games=50)  # Start with 50 games

        # Save results
        fetcher.save_data()


if __name__ == "__main__":
    asyncio.run(main())
