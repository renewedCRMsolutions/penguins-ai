# File: test/test_nhl_endpoints.py
"""
Test all NHL API endpoints to discover available data and shot speed locations
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHLEndpointTester:
    """Test NHL API endpoints to find all available data"""

    def __init__(self):
        self.base_url = "https://api-web.nhle.com/v1"
        self.session: Optional[aiohttp.ClientSession] = None
        self.test_game_id = "2024020100"  # Example game ID
        self.test_player_id = "8478402"  # Connor McDavid
        self.test_team = "EDM"  # Edmonton Oilers
        self.test_date = "2024-10-15"
        self.results = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_endpoint(self, name: str, url: str) -> Dict:
        """Test a single endpoint"""
        logger.info(f"Testing {name}: {url}")

        try:
            if not self.session:
                raise RuntimeError("Session not initialized")

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    result = {
                        "status": "success",
                        "url": url,
                        "sample_data": self._extract_sample(data),
                        "keys": list(data.keys()) if isinstance(data, dict) else "list",
                        "has_shot_data": self._check_for_shot_data(data),
                    }

                    logger.info(f"  ✅ Success - Keys: {result['keys']}")

                    # Save full response for interesting endpoints
                    if result["has_shot_data"]:
                        with open(f"data/endpoint_samples/{name.replace('/', '_')}.json", "w") as f:
                            json.dump(data, f, indent=2)

                    return result
                else:
                    logger.warning(f"  ❌ Failed - Status: {response.status}")
                    return {"status": "failed", "code": response.status, "url": url}

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            return {"status": "error", "error": str(e), "url": url}

    def _extract_sample(self, data: Any) -> Any:
        """Extract a sample of the data for inspection"""
        if isinstance(data, dict):
            # For dicts, show structure
            sample = {}
            for key, value in list(data.items())[:5]:
                if isinstance(value, (dict, list)):
                    sample[key] = f"<{type(value).__name__}>"
                else:
                    sample[key] = value
            return sample
        elif isinstance(data, list) and data:
            # For lists, show first item
            return self._extract_sample(data[0])
        return data

    def _check_for_shot_data(self, data: Any, depth: int = 0) -> bool:
        """Recursively check for shot-related data"""
        if depth > 5:  # Prevent infinite recursion
            return False

        shot_keywords = [
            "shot",
            "speed",
            "velocity",
            "mph",
            "kph",
            "shotSpeed",
            "shotVelocity",
            "shotType",
            "distance",
            "angle",
        ]

        if isinstance(data, dict):
            # Check keys
            for key in data.keys():
                if any(keyword in str(key).lower() for keyword in shot_keywords):
                    return True
            # Check values
            for value in data.values():
                if self._check_for_shot_data(value, depth + 1):
                    return True

        elif isinstance(data, list) and data:
            # Check first few items
            for item in data[:3]:
                if self._check_for_shot_data(item, depth + 1):
                    return True

        return False

    async def test_all_endpoints(self):
        """Test all known NHL API endpoints"""
        logger.info("\n🏒 TESTING NHL API ENDPOINTS")
        logger.info("=" * 60)

        # Create sample directory
        import os

        os.makedirs("data/endpoint_samples", exist_ok=True)

        endpoints = {
            # Schedule endpoints
            "schedule/now": f"{self.base_url}/schedule/now",
            "schedule/date": f"{self.base_url}/schedule/{self.test_date}",
            # Team schedule
            "club-schedule/month": f"{self.base_url}/club-schedule/{self.test_team}/month/now",
            "club-schedule/week": f"{self.base_url}/club-schedule/{self.test_team}/week/now",
            "club-schedule-season": f"{self.base_url}/club-schedule-season/{self.test_team}/20242025",
            # Scores
            "score/now": f"{self.base_url}/score/now",
            "score/date": f"{self.base_url}/score/{self.test_date}",
            # Game details - THESE ARE KEY FOR SHOT DATA
            "gamecenter/landing": f"{self.base_url}/gamecenter/{self.test_game_id}/landing",
            "gamecenter/boxscore": f"{self.base_url}/gamecenter/{self.test_game_id}/boxscore",
            "gamecenter/play-by-play": f"{self.base_url}/gamecenter/{self.test_game_id}/play-by-play",
            "gamecenter/right-rail": f"{self.base_url}/gamecenter/{self.test_game_id}/right-rail",
            # WSC endpoints - MIGHT HAVE TRACKING DATA
            "wsc/play-by-play": f"{self.base_url}/wsc/play-by-play/{self.test_game_id}",
            "wsc/game-story": f"{self.base_url}/wsc/game-story/{self.test_game_id}",
            # Player data
            "player/landing": f"{self.base_url}/player/{self.test_player_id}/landing",
            "player/game-log": f"{self.base_url}/player/{self.test_player_id}/game-log/20242025/2",
            # Stats leaders - MIGHT HAVE AGGREGATED SHOT SPEEDS
            "skater-stats-leaders": (
                f"{self.base_url}/skater-stats-leaders/current?" "categories=goals,shots,points&limit=10"
            ),
            "goalie-stats-leaders": (
                f"{self.base_url}/goalie-stats-leaders/current?" "categories=wins,saves,savePercentage&limit=10"
            ),
            # Team stats
            "club-stats": f"{self.base_url}/club-stats/{self.test_team}/20242025/2",
            "club-stats-season": f"{self.base_url}/club-stats-season/{self.test_team}",
            # Standings
            "standings/now": f"{self.base_url}/standings/now",
            # Roster
            "roster/current": f"{self.base_url}/roster/{self.test_team}/current",
            # Draft
            "draft/picks": f"{self.base_url}/draft/picks/2024/1",
        }

        # Test each endpoint
        for name, url in endpoints.items():
            self.results[name] = await self.test_endpoint(name, url)
            await asyncio.sleep(0.5)  # Be nice to the API

        # Analyze results
        self.analyze_results()

    def analyze_results(self):
        """Analyze test results to find shot data"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 ENDPOINT ANALYSIS RESULTS")
        logger.info("=" * 60)

        # Count successes
        successful = sum(1 for r in self.results.values() if r.get("status") == "success")
        logger.info(f"\nTotal endpoints tested: {len(self.results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(self.results) - successful}")

        # Find endpoints with shot data
        shot_endpoints = [name for name, result in self.results.items() if result.get("has_shot_data")]

        if shot_endpoints:
            logger.info("\n🎯 ENDPOINTS WITH POTENTIAL SHOT DATA:")
            for endpoint in shot_endpoints:
                logger.info(f"  - {endpoint}")
        else:
            logger.warning("\n❌ No endpoints found with obvious shot data")

        # Save full results
        with open("data/endpoint_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("\nFull results saved to data/endpoint_test_results.json")

        # Specific endpoint insights
        self.check_play_by_play_structure()

    def check_play_by_play_structure(self):
        """Deep dive into play-by-play structure"""
        pbp_result = self.results.get("gamecenter/play-by-play", {})

        if pbp_result.get("status") == "success":
            logger.info("\n🔍 PLAY-BY-PLAY STRUCTURE ANALYSIS")
            logger.info("=" * 40)

            # Load the saved file
            try:
                with open("data/endpoint_samples/gamecenter_play-by-play.json", "r") as f:
                    pbp_data = json.load(f)

                # Look for plays
                if "plays" in pbp_data:
                    plays = pbp_data["plays"]
                    logger.info(f"Total plays: {len(plays)}")

                    # Find shot plays
                    shot_plays = [p for p in plays if p.get("typeDescKey") in ["shot-on-goal", "goal"]]
                    logger.info(f"Shot plays: {len(shot_plays)}")

                    if shot_plays:
                        # Analyze first shot
                        first_shot = shot_plays[0]
                        logger.info("\nFirst shot structure:")
                        logger.info(f"  Type: {first_shot.get('typeDescKey')}")
                        logger.info(f"  Period: {first_shot.get('periodDescriptor', {}).get('number')}")

                        # Check details
                        details = first_shot.get("details", {})
                        logger.info("\n  Details keys:")
                        for key in details.keys():
                            logger.info(f"    - {key}: {type(details[key]).__name__}")

                        # Look for speed data
                        speed_keys = [k for k in details.keys() if "speed" in k.lower() or "velocity" in k.lower()]
                        if speed_keys:
                            logger.info(f"\n  🎯 FOUND SPEED KEYS: {speed_keys}")
                            for key in speed_keys:
                                logger.info(f"    {key}: {details[key]}")

            except Exception as e:
                logger.error(f"Error analyzing PBP structure: {e}")


async def test_nhl_api_py():
    """Test the nhl-api-py package for additional data"""
    try:
        from nhlpy import NHLClient

        logger.info("\n🏒 TESTING NHL-API-PY PACKAGE")
        logger.info("=" * 60)

        client = NHLClient(verbose=True)

        # Test key endpoints
        tests = {
            "Game PBP": lambda: client.game_center.play_by_play(game_id="2024020100"),
            "Player Stats": lambda: client.stats.player_career_stats(player_id="8478402"),
            "Skater Summary": lambda: client.stats.skater_stats_summary_simple(
                start_season="20242025", end_season="20242025"
            ),
        }

        for name, test_func in tests.items():
            try:
                logger.info(f"\nTesting {name}...")
                result = test_func()

                # Check for shot speed data
                if isinstance(result, dict):
                    check_nested_for_speed(result, name)

            except Exception as e:
                logger.error(f"Error testing {name}: {e}")

    except ImportError:
        logger.warning("nhl-api-py not installed. Run: pip install nhl-api-py")


def check_nested_for_speed(data: Any, context: str, depth: int = 0):
    """Recursively check for speed-related fields"""
    if depth > 3:
        return

    if isinstance(data, dict):
        for key, value in data.items():
            if any(term in str(key).lower() for term in ["speed", "velocity", "mph", "kph"]):
                logger.info(f"  🎯 Found in {context}: {key} = {value}")
            elif isinstance(value, (dict, list)):
                check_nested_for_speed(value, f"{context}.{key}", depth + 1)
    elif isinstance(data, list) and data:
        check_nested_for_speed(data[0], f"{context}[0]", depth + 1)


async def main():
    """Run all tests"""
    # Test NHL API endpoints
    async with NHLEndpointTester() as tester:
        await tester.test_all_endpoints()

    # Test nhl-api-py package
    await test_nhl_api_py()

    logger.info("\n✅ TESTING COMPLETE")
    logger.info("Check data/endpoint_samples/ for full JSON responses")
    logger.info("Check data/endpoint_test_results.json for summary")


if __name__ == "__main__":
    asyncio.run(main())
