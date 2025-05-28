# File: debug_player_names_full.py
"""
Full debug script to find player names in NHL API
"""

from nhlpy import NHLClient
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_full_game_data():
    """Save complete game data to understand structure"""

    client = NHLClient()
    game_id = "2024020001"

    logger.info("Fetching complete game data...")

    # 1. Get boxscore
    boxscore = client.game_center.boxscore(game_id=game_id)

    # Save FULL boxscore
    with open("data/nhl/debug_boxscore_FULL.json", "w") as f:
        json.dump(boxscore, f, indent=2)
    logger.info("Saved full boxscore to debug_boxscore_FULL.json")

    # 2. Get play-by-play
    pbp = client.game_center.play_by_play(game_id=game_id)

    # Save just the roster spots
    if "rosterSpots" in pbp:
        with open("data/nhl/debug_rosterSpots.json", "w") as f:
            json.dump(pbp["rosterSpots"][:5], f, indent=2)  # First 5 players
        logger.info(f"Found {len(pbp['rosterSpots'])} roster spots")

    # 3. Check for playerByGameStats
    if boxscore and "playerByGameStats" in boxscore:
        logger.info("\n✅ Found playerByGameStats!")

        # Save player stats structure
        with open("data/nhl/debug_playerByGameStats.json", "w") as f:
            # Just save structure, not all data
            stats = boxscore["playerByGameStats"]
            sample = {}

            for team in ["homeTeam", "awayTeam"]:
                if team in stats:
                    sample[team] = {}
                    team_data = stats[team]

                    # Check what keys exist
                    logger.info(f"\n{team} player stat keys: {list(team_data.keys())}")

                    # Sample each position group
                    for key in team_data.keys():
                        if isinstance(team_data[key], list) and len(team_data[key]) > 0:
                            # Just save first player as example
                            sample[team][key] = team_data[key][0]
                            logger.info(f"  {key}: {len(team_data[key])} players")

            json.dump(sample, f, indent=2)

        logger.info("\nSaved player stats sample to debug_playerByGameStats.json")

    # 4. Try to find a specific player
    shooter_id = "8483495"  # From your debug output
    logger.info(f"\nLooking for player {shooter_id} in all data...")

    # Search in boxscore
    found = search_for_player_in_dict(boxscore, shooter_id)
    if found:
        logger.info(f"Found player {shooter_id}!")
        logger.info(f"Path: {found['path']}")
        logger.info(f"Data: {json.dumps(found['data'], indent=2)}")

    # 5. Get player directly from API
    logger.info(f"\nFetching player {shooter_id} directly...")

    # Use the player landing endpoint
    try:
        # Direct URL approach since nhlpy might not have player.get_player
        import requests

        url = f"https://api-web.nhle.com/v1/player/{shooter_id}/landing"
        response = requests.get(url)

        if response.status_code == 200:
            player_data = response.json()

            with open(f"data/nhl/debug_player_{shooter_id}.json", "w") as f:
                json.dump(player_data, f, indent=2)

            # Try to extract name
            name = extract_player_name(player_data)
            logger.info(f"Player name: {name}")
        else:
            logger.error(f"Failed to fetch player {shooter_id}: {response.status_code}")

    except Exception as e:
        logger.error(f"Error fetching player: {e}")


def search_for_player_in_dict(data, player_id, path=""):
    """Recursively search for player ID in nested dict"""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "playerId" and str(value) == str(player_id):
                # Found it! Return the parent object
                return {"path": path, "data": data}

            # Recurse
            result = search_for_player_in_dict(value, player_id, f"{path}.{key}")
            if result:
                return result

    elif isinstance(data, list):
        for i, item in enumerate(data):
            result = search_for_player_in_dict(item, player_id, f"{path}[{i}]")
            if result:
                return result

    return None


def extract_player_name(player_data):
    """Extract name from various possible locations"""
    # Try different paths
    paths = [
        lambda d: f"{d.get('firstName', {}).get('default', '')} {d.get('lastName', {}).get('default', '')}",
        lambda d: d.get("fullName", ""),
        lambda d: d.get("name", {}).get("default", ""),
        lambda d: f"{d.get('name', {}).get('first', '')} {d.get('name', {}).get('last', '')}",
    ]

    for path in paths:
        try:
            name = path(player_data).strip()
            if name:
                return name
        except Exception:
            continue

    return "Unknown"


def check_landing_structure():
    """Check the landing endpoint structure"""
    client = NHLClient()
    game_id = "2024020001"

    landing = client.game_center.landing(game_id=game_id)

    logger.info("\nChecking landing endpoint structure...")

    # Look for player rosters
    if landing:
        # Save summary
        with open("data/nhl/debug_landing_structure.json", "w") as f:
            # Just save keys to understand structure
            structure = {
                "top_level_keys": list(landing.keys()),
            }

            # Check for roster/player keys
            for key in ["homeTeam", "awayTeam", "summary", "gameInfo"]:
                if key in landing:
                    if isinstance(landing[key], dict):
                        structure[f"{key}_keys"] = list(landing[key].keys())
                    else:
                        # Store the type name as a simple value, not trying to replace the list
                        structure[f"{key}_type"] = type(landing[key]).__name__

            json.dump(structure, f, indent=2)

        logger.info("Saved landing structure to debug_landing_structure.json")


def main():
    """Run all debugging"""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE NHL API PLAYER NAME DEBUG")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs("data/nhl", exist_ok=True)

    # Run all checks
    save_full_game_data()
    check_landing_structure()

    logger.info("\n" + "=" * 60)
    logger.info("DEBUGGING COMPLETE!")
    logger.info("Check these files:")
    logger.info("  - data/nhl/debug_boxscore_FULL.json")
    logger.info("  - data/nhl/debug_playerByGameStats.json")
    logger.info("  - data/nhl/debug_rosterSpots.json")
    logger.info("  - data/nhl/debug_player_8483495.json")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
