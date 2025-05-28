# File: test_nhl_api.py
"""
Test NHL API directly to debug issues
"""

from nhlpy import NHLClient
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_direct_api():
    """Test the NHL API with direct requests"""

    logger.info("Testing NHL API endpoints directly...")

    # Test 1: Direct HTTP request to NHL API
    logger.info("\n1. Testing direct HTTP request:")

    test_dates = [
        "2025-05-27",
        "2025-05-15",
        "2025-04-15",
        "2025-03-15",
        "2025-01-15",
        "2024-11-15",
    ]

    for date in test_dates:
        url = f"https://api-web.nhle.com/v1/schedule/{date}"
        try:
            response = requests.get(url)
            logger.info(f"\nDate: {date}")
            logger.info(f"Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()

                # Check if there's game data
                if "gameWeek" in data:
                    total_games = 0
                    for day in data["gameWeek"]:
                        if "games" in day:
                            total_games += len(day["games"])
                    logger.info(f"Games found: {total_games}")

                    # Show first game details if any
                    if total_games > 0:
                        for day in data["gameWeek"]:
                            if "games" in day and day["games"]:
                                game = day["games"][0]
                                away = game.get('awayTeam', {}).get('abbrev')
                                home = game.get('homeTeam', {}).get('abbrev')
                                logger.info(f"Sample game: {away} @ {home}")
                                logger.info(f"Game ID: {game.get('id')}")
                                logger.info(f"Game Type: {game.get('gameType')}")
                                break
                else:
                    logger.info("No gameWeek data in response")

        except Exception as e:
            logger.error(f"Error: {e}")


def test_nhlpy_client():
    """Test using nhl-api-py client"""

    logger.info("\n\n2. Testing nhl-api-py client:")

    client = NHLClient(verbose=True)

    # Test different methods
    try:
        # Test 1: Get current scores
        logger.info("\nTesting score_now endpoint:")
        scores = client.game_center.score_now()
        if scores:
            logger.info(f"Score data retrieved: {type(scores)}")
            if isinstance(scores, dict) and "games" in scores:
                logger.info(f"Current games: {len(scores['games'])}")

    except Exception as e:
        logger.error(f"score_now error: {e}")

    try:
        # Test 2: Get season games
        logger.info("\nTesting season games:")
        games = client.schedule.get_season_schedule(team_abbr="PIT", season="20242025")
        if games:
            logger.info(f"Found Pittsburgh games: {len(games) if isinstance(games, list) else 'Unknown'}")

    except Exception as e:
        logger.error(f"Season schedule error: {e}")

    try:
        # Test 3: Get specific game IDs for testing
        logger.info("\nTrying to get game IDs by season:")
        game_ids = client.helpers.get_gameids_by_season("20242025", game_types=[2, 3])
        if game_ids:
            logger.info(f"Found {len(game_ids)} game IDs")
            logger.info(f"Sample game IDs: {game_ids[:5]}")

    except Exception as e:
        logger.error(f"Game IDs error: {e}")


def test_game_data():
    """Try to get actual game data"""

    logger.info("\n\n3. Testing with known game IDs:")

    client = NHLClient()

    # Try some standard game ID formats
    # NHL game IDs typically follow pattern: YYYY02XXXX for regular season
    # or YYYY03XXXX for playoffs

    test_game_ids = [
        "2024020001",  # First game of 2024-25 regular season
        "2024021000",  # Mid-season game
        "2025020800",  # Later regular season
        "2025030111",  # Potential playoff game
    ]

    for game_id in test_game_ids:
        try:
            logger.info(f"\nTrying game ID: {game_id}")
            pbp = client.game_center.play_by_play(game_id=game_id)

            if pbp and "plays" in pbp:
                shot_types = ["shot-on-goal", "goal"]
                shots = sum(1 for play in pbp["plays"] if play.get("typeDescKey") in shot_types)
                logger.info(f"SUCCESS! Found {shots} shots in game {game_id}")

                # Get some game info
                if "gameDate" in pbp:
                    logger.info(f"Game date: {pbp['gameDate']}")

                return game_id  # Return a working game ID

        except Exception:
            logger.info(f"No data for {game_id}")


def suggest_fix():
    """Suggest how to fix the data collection"""

    logger.info("\n\n" + "=" * 60)
    logger.info("SUGGESTED FIXES")
    logger.info("=" * 60)

    logger.info("\nThe issue might be:")
    logger.info("1. We're in the off-season (late May)")
    logger.info("2. Need to use different game type codes")
    logger.info("3. Need to go back further for regular season data")

    logger.info("\nTry collecting data from earlier in the season:")
    logger.info("\n# Regular season (October 2024 - April 2025):")
    logger.info("python run_full_pipeline.py --mode full --days 180")

    logger.info("\n# Or modify the collect script to use specific dates:")
    logger.info("# In collect_live_nhl_data.py, change to:")
    logger.info("# start_date = '2024-10-01'")
    logger.info("# end_date = '2025-04-15'")


def main():
    """Run all tests"""

    logger.info("=" * 60)
    logger.info("NHL API DIAGNOSTIC TEST")
    logger.info("=" * 60)

    # Run tests
    test_direct_api()
    test_nhlpy_client()
    working_game = test_game_data()

    if working_game:
        logger.info(f"\n✅ Found working game: {working_game}")
        logger.info("The API is working, just need to use the right date range!")

    suggest_fix()


if __name__ == "__main__":
    main()
