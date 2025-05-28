# File: check_nhl_schedule.py
"""
Check what NHL games are available to collect
"""

from nhlpy import NHLClient
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_available_games():
    """Check what games are available in different date ranges"""

    client = NHLClient()

    # Current date
    today = datetime(2025, 5, 28)

    logger.info("Checking NHL game availability...")
    logger.info(f"Current date: {today.strftime('%Y-%m-%d')}")

    # Check different date ranges
    date_ranges = [
        # Recent games (last 7 days)
        (today - timedelta(days=7), today, "Last 7 days"),
        # Last month
        (today - timedelta(days=30), today, "Last 30 days"),
        # Earlier in playoffs (April-May)
        (datetime(2025, 4, 1), datetime(2025, 5, 28), "Playoffs (April-May)"),
        # Regular season end (March-April)
        (datetime(2025, 3, 1), datetime(2025, 4, 15), "Regular season end"),
        # Mid-season (January-February)
        (datetime(2025, 1, 1), datetime(2025, 2, 28), "Mid-season"),
        # Season start (October 2024)
        (datetime(2024, 10, 1), datetime(2024, 11, 30), "Season start"),
    ]

    for start_date, end_date, period_name in date_ranges:
        logger.info(f"\n{period_name}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        game_count = 0
        current = start_date

        while current <= end_date:
            try:
                schedule = client.schedule.get_schedule(date=current.strftime("%Y-%m-%d"))

                if schedule and "gameWeek" in schedule:
                    for day in schedule["gameWeek"]:
                        if "games" in day:
                            daily_games = len(day["games"])
                            if daily_games > 0:
                                game_count += daily_games

            except Exception:
                pass  # Silent fail for dates with no games

            current += timedelta(days=1)

        logger.info(f"  Found {game_count} games")

    # Get specific info about recent games
    logger.info("\nChecking most recent games with play-by-play data:")

    # Try to find the most recent games
    for days_back in range(1, 60):
        check_date = today - timedelta(days=days_back)

        try:
            schedule = client.schedule.get_schedule(date=check_date.strftime("%Y-%m-%d"))

            if schedule and "gameWeek" in schedule:
                for day in schedule["gameWeek"]:
                    if "games" in day and len(day["games"]) > 0:
                        logger.info(f"\nFound games on {check_date.strftime('%Y-%m-%d')}:")

                        for game in day["games"][:3]:  # Show first 3 games
                            game_id = game.get("id")
                            away = game.get("awayTeam", {}).get("abbrev", "UNK")
                            home = game.get("homeTeam", {}).get("abbrev", "UNK")
                            state = game.get("gameState", "Unknown")

                            logger.info(f"  {away} @ {home} - Game ID: {game_id} - State: {state}")

                            # Try to get play-by-play to see if shot data exists
                            if state in ["FINAL", "OFF"]:
                                try:
                                    pbp = client.game_center.play_by_play(game_id=str(game_id))
                                    if pbp and "plays" in pbp:
                                        shot_count = sum(
                                            1
                                            for play in pbp["plays"]
                                            if play.get("typeDescKey") in ["shot-on-goal", "goal"]
                                        )
                                        logger.info(f"    -> Contains {shot_count} shots/goals")
                                except Exception:
                                    pass

                        return check_date  # Return the most recent date with games

        except Exception:
            pass

    return None


def suggest_best_collection_strategy():
    """Suggest the best strategy for collecting data"""

    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDED DATA COLLECTION STRATEGY")
    logger.info("=" * 60)

    logger.info("\nSince it's late May 2025, we're in the playoffs/off-season.")
    logger.info("Here are your best options for collecting shot data:")

    logger.info("\n1. PLAYOFFS 2025 (Recommended):")
    logger.info("   python run_full_pipeline.py --mode full --days 60")
    logger.info("   -> Gets playoff games (higher quality, more intense)")

    logger.info("\n2. RECENT REGULAR SEASON:")
    logger.info("   python data/collect_live_nhl_data.py")
    logger.info("   -> Then modify to use date range: 2025-01-01 to 2025-04-15")

    logger.info("\n3. FULL SEASON DATA:")
    logger.info("   Collect from October 2024 to April 2025")
    logger.info("   -> Most data, but takes longer")

    logger.info("\n4. SPECIFIC TEAM FOCUS (for Penguins):")
    logger.info("   Modify collector to filter for Pittsburgh games only")


def main():
    """Check NHL data availability"""

    logger.info("=" * 60)
    logger.info("NHL DATA AVAILABILITY CHECK")
    logger.info("=" * 60)

    # Check what's available
    most_recent = check_available_games()

    if most_recent:
        logger.info(f"\nMost recent games found: {most_recent.strftime('%Y-%m-%d')}")

    # Suggest strategy
    suggest_best_collection_strategy()


if __name__ == "__main__":
    main()
