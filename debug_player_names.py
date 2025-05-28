# File: debug_player_names.py
"""
Debug script to figure out how to get player names from NHL API
"""

from nhlpy import NHLClient
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def explore_game_data():
    """Explore different API endpoints to find player names"""

    client = NHLClient()

    # Use a known game ID
    game_id = "2024020001"

    logger.info("=" * 60)
    logger.info("EXPLORING NHL API DATA STRUCTURE")
    logger.info("=" * 60)

    # 1. Check play-by-play structure
    logger.info(f"\n1. Checking play-by-play for game {game_id}...")
    pbp = client.game_center.play_by_play(game_id=game_id)

    if pbp:
        # Look for roster information
        logger.info("\nChecking for roster in play-by-play:")

        # Check different possible locations
        possible_roster_keys = ["rosterSpots", "roster", "players", "boxscore"]
        for key in possible_roster_keys:
            if key in pbp:
                logger.info(f"  Found '{key}' in pbp")
                # Show structure
                if isinstance(pbp[key], dict):
                    logger.info(f"    Keys: {list(pbp[key].keys())[:5]}...")

        # Check team structure
        for team in ["homeTeam", "awayTeam"]:
            if team in pbp:
                logger.info(f"\n  {team} structure:")
                team_data = pbp[team]
                logger.info(f"    Keys: {list(team_data.keys())}")

                # Look for roster/players
                if "roster" in team_data:
                    logger.info(f"    Found roster in {team}")
                if "players" in team_data:
                    logger.info(f"    Found players in {team}")

        # Look at a shot play
        logger.info("\nLooking for shot with player info:")
        for play in pbp.get("plays", [])[:50]:  # First 50 plays
            if play.get("typeDescKey") in ["shot-on-goal", "goal"]:
                logger.info(f"\n  Found {play['typeDescKey']}:")
                details = play.get("details", {})

                # Show all details keys
                logger.info(f"    Details keys: {list(details.keys())}")

                # Check for player info
                shooter_id = details.get("shootingPlayerId")
                if shooter_id:
                    logger.info(f"    Shooter ID: {shooter_id}")

                    # Check if name is in details
                    for key in details:
                        if "name" in key.lower() or "player" in key.lower():
                            logger.info(f"    Found potential name field: {key} = {details[key]}")

                break

    # 2. Check boxscore structure
    logger.info(f"\n2. Checking boxscore for game {game_id}...")
    boxscore = client.game_center.boxscore(game_id=game_id)

    if boxscore:
        logger.info("\nBoxscore structure:")
        logger.info(f"  Top-level keys: {list(boxscore.keys())}")

        # Check team structure in boxscore
        for team in ["homeTeam", "awayTeam"]:
            if team in boxscore:
                team_data = boxscore[team]
                logger.info(f"\n  {team} in boxscore:")
                logger.info(f"    Keys: {list(team_data.keys())[:10]}...")

                # Look for player data
                if "players" in team_data:
                    players = team_data["players"]
                    logger.info(f"    Players type: {type(players)}")
                    if isinstance(players, dict) and len(players) > 0:
                        # Show first player
                        first_key = list(players.keys())[0]
                        first_player = players[first_key]
                        logger.info(f"    First player key: {first_key}")
                        logger.info(f"    First player data keys: {list(first_player.keys())[:10]}...")

                        # Look for name
                        if "name" in first_player:
                            logger.info(f"    Name structure: {first_player['name']}")

                        # Show player ID
                        if "playerId" in first_player:
                            logger.info(f"    Player ID: {first_player['playerId']}")

                # Check roster spots
                if "forwards" in team_data:
                    logger.info(f"    Found forwards: {len(team_data['forwards'])} players")
                    logger.info(f"    Forward IDs: {team_data['forwards'][:3]}...")

    # 3. Try landing endpoint
    logger.info(f"\n3. Checking landing endpoint for game {game_id}...")
    try:
        landing = client.game_center.landing(game_id=game_id)
        if landing:
            logger.info("Landing endpoint available")
            # Check for roster/player info
            if "roster" in landing:
                logger.info("  Found roster in landing")
    except Exception as e:
        logger.info(f"  Landing endpoint error: {e}")

    # 4. Show how to get a specific player
    logger.info("\n4. Testing player lookup...")

    # Try to get player 8476460 (from your data)
    player_id = "8476460"
    logger.info(f"\nLooking for player {player_id}...")

    # Check if we can get player info directly
    try:
        # Some APIs have player endpoints
        logger.info("Checking for player-specific endpoints...")
        # This would be where we'd check for player endpoints if they exist
    except Exception as e:
        logger.info(f"No direct player endpoint: {e}")

    # 5. Save sample data for inspection
    logger.info("\n5. Saving sample data for manual inspection...")

    # Save boxscore sample
    if boxscore and "homeTeam" in boxscore:
        sample_file = "data/nhl/debug_boxscore_sample.json"
        with open(sample_file, "w") as f:
            # Just save home team data
            sample = {"homeTeam": boxscore["homeTeam"]}
            json.dump(sample, f, indent=2)
        logger.info(f"  Saved boxscore sample to {sample_file}")
        logger.info("  You can manually inspect this to find player names")


def test_working_approach():
    """Test the approach that should work"""

    logger.info("\n" + "=" * 60)
    logger.info("TESTING PLAYER NAME EXTRACTION")
    logger.info("=" * 60)

    client = NHLClient()
    game_id = "2024020001"

    # Get both endpoints
    pbp = client.game_center.play_by_play(game_id=game_id)
    boxscore = client.game_center.boxscore(game_id=game_id)

    # Find a shot
    for play in pbp.get("plays", [])[:100]:
        if play.get("typeDescKey") == "shot-on-goal":
            details = play.get("details", {})
            shooter_id = details.get("shootingPlayerId")

            if shooter_id:
                logger.info(f"\nFound shot by player {shooter_id}")

                # Try to find this player in boxscore
                for team in ["homeTeam", "awayTeam"]:
                    if team in boxscore:
                        players = boxscore[team].get("players", {})

                        # Check different ID formats
                        for key_format in [f"ID{shooter_id}", str(shooter_id), f"player_{shooter_id}"]:
                            if key_format in players:
                                player_data = players[key_format]
                                logger.info(f"  Found player in {team} boxscore!")

                                # Extract name
                                if "name" in player_data:
                                    name = player_data["name"]
                                    if isinstance(name, dict):
                                        full_name = f"{name.get('firstName', '')} {name.get('lastName', '')}"
                                    else:
                                        full_name = str(name)
                                    logger.info(f"  Player name: {full_name}")
                                    return

                # If not found in boxscore, check pbp roster
                logger.info("  Not found in boxscore, checking play-by-play...")

                # Check rosterSpots
                for team in ["homeTeam", "awayTeam"]:
                    roster = pbp.get("rosterSpots", [])
                    logger.info(f"  Checking {len(roster)} roster spots...")

                break


def main():
    """Run all debugging"""
    explore_game_data()
    test_working_approach()

    logger.info("\n" + "=" * 60)
    logger.info("DEBUGGING COMPLETE")
    logger.info("Check data/nhl/debug_boxscore_sample.json for full structure")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
