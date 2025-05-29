# api/game_tracker.py
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict
import pandas as pd


class GameTracker:
    def __init__(self, model, features):
        self.model = model
        self.features = features
        self.player_cache = {}

    async def fetch_game_data(self, game_id: str) -> Dict:
        """Fetch play-by-play data from NHL API"""
        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()

    def calculate_from_pbp(self, play: Dict, pbp_data: Dict) -> float:
        """Calculate time since last faceoff"""
        play_idx = next(i for i, p in enumerate(pbp_data["plays"]) if p == play)
        for i in range(play_idx - 1, -1, -1):
            if pbp_data["plays"][i]["typeDescKey"] == "faceoff":
                return play["timeInPeriod"] - pbp_data["plays"][i]["timeInPeriod"]
        return 30.0

    async def get_player_stats(self, player_id: str) -> Dict:
        """Get cached or fetch player statistics"""
        if player_id in self.player_cache:
            return self.player_cache[player_id]

        # Fetch from NHL API or return defaults
        return {"shooting_talent": 1.0, "goals_per_xgoals": 1.0, "shot_quality_ratio": 0.3}

    async def track_game(self, game_id: str):
        """Main game tracking loop"""
        while True:
            pbp_data = await self.fetch_game_data(game_id)

            for play in pbp_data.get("plays", []):
                if play.get("typeDescKey") == "shot":
                    time_since_faceoff = self.calculate_from_pbp(play, pbp_data)
                    shooter_stats = await self.get_player_stats(play.get("shooterId", ""))

                    # Prepare features for model
                    shot_features = self.prepare_shot_features(play, time_since_faceoff, shooter_stats)

                    # Make prediction
                    shot_df = pd.DataFrame([shot_features])
                    prediction = float(self.model.predict_proba(shot_df[self.features])[0, 1])

                    yield {
                        "event": "shot",
                        "xG": prediction,
                        "location": {"x": play.get("xCoord"), "y": play.get("yCoord")},
                        "shooter": play.get("shooterId"),
                        "timestamp": datetime.now().isoformat(),
                    }

            await asyncio.sleep(5)

    def prepare_shot_features(self, play: Dict, time_since_faceoff: float, shooter_stats: Dict) -> Dict:
        """Convert play data to model features"""
        return {
            "arenaAdjustedShotDistance": play.get("shotDistance", 30),
            "shotAngleAdjusted": play.get("shotAngle", 0),
            "shooting_talent": shooter_stats["shooting_talent"],
            "timeSinceFaceoff": time_since_faceoff,
            # Add all other required features with defaults
        }
