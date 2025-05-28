# File: test/check_pbp_structure.py
"""
Deep dive into play-by-play data to find all available fields
"""

import json
import os
from typing import Set, Any
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_pbp_structure():
    """Analyze the play-by-play JSON structure in detail"""

    # Check if we have the sample file
    pbp_file = "data/endpoint_samples/gamecenter_play-by-play.json"

    if not os.path.exists(pbp_file):
        logger.error(f"File not found: {pbp_file}")
        logger.info("Run test_nhl_endpoints.py first to generate sample data")
        return

    # Load the data
    with open(pbp_file, "r") as f:
        data = json.load(f)

    logger.info("🏒 ANALYZING PLAY-BY-PLAY STRUCTURE")
    logger.info("=" * 60)

    # Get all plays
    plays = data.get("plays", [])
    logger.info(f"Total plays: {len(plays)}")

    # Analyze different play types
    play_types = defaultdict(int)
    for play in plays:
        play_types[play.get("typeDescKey", "unknown")] += 1

    logger.info("\nPlay type distribution:")
    for ptype, count in sorted(play_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  {ptype}: {count}")

    # Focus on shot plays
    shot_plays = [p for p in plays if p.get("typeDescKey") in ["shot-on-goal", "goal", "missed-shot", "blocked-shot"]]
    logger.info(f"\nTotal shot-related plays: {len(shot_plays)}")

    # Analyze all unique fields in shot plays
    all_fields = set()
    detail_fields = set()

    for play in shot_plays:
        # Top level fields
        all_fields.update(play.keys())

        # Detail fields
        if "details" in play:
            detail_fields.update(play["details"].keys())

    logger.info("\nTop-level fields in shot plays:")
    for field in sorted(all_fields):
        logger.info(f"  - {field}")

    logger.info("\nDetail fields in shot plays:")
    for field in sorted(detail_fields):
        logger.info(f"  - {field}")

    # Look for any field containing speed/velocity keywords
    logger.info("\n🔍 SEARCHING FOR SPEED-RELATED FIELDS...")
    speed_keywords = ["speed", "velocity", "mph", "kph", "fast", "hard", "power"]

    def search_for_keywords(obj: Any, path: str = "", depth: int = 0) -> Set[str]:
        """Recursively search for speed-related keywords"""
        found_paths = set()

        if depth > 5:  # Prevent infinite recursion
            return found_paths

        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # Check if key contains any speed keyword
                if any(keyword in str(key).lower() for keyword in speed_keywords):
                    found_paths.add(f"{current_path} = {value}")

                # Recurse into value
                found_paths.update(search_for_keywords(value, current_path, depth + 1))

        elif isinstance(obj, list) and obj:
            # Check first few items
            for i, item in enumerate(obj[:3]):
                found_paths.update(search_for_keywords(item, f"{path}[{i}]", depth + 1))

        return found_paths

    speed_fields = search_for_keywords(data)

    if speed_fields:
        logger.info("✅ Found speed-related fields:")
        for field in sorted(speed_fields):
            logger.info(f"  {field}")
    else:
        logger.info("❌ No speed-related fields found in play-by-play data")

    # Sample shot details
    if shot_plays:
        logger.info("\n📊 SAMPLE SHOT DETAILS:")
        for i, play in enumerate(shot_plays[:3]):
            logger.info(f"\nShot {i + 1} ({play.get('typeDescKey')}):")
            details = play.get("details", {})

            # Key fields
            logger.info(f"  Location: ({details.get('xCoord')}, {details.get('yCoord')})")
            logger.info(f"  Shot type: {details.get('shotType')}")
            logger.info(f"  Zone: {details.get('zoneCode')}")
            logger.info(f"  Period: {play.get('periodDescriptor', {}).get('number')}")
            logger.info(f"  Time: {play.get('timeInPeriod')}")

            # Check for any additional fields
            extra_fields = [k for k in details.keys() if k not in ["xCoord", "yCoord", "shotType", "zoneCode"]]
            if extra_fields:
                logger.info(f"  Additional fields: {extra_fields}")

    # Check other endpoints for shot data
    logger.info("\n🔍 CHECKING OTHER ENDPOINTS FOR SHOT DATA...")

    # Check right-rail data
    right_rail_file = "data/endpoint_samples/gamecenter_right-rail.json"
    if os.path.exists(right_rail_file):
        with open(right_rail_file, "r") as f:
            rr_data = json.load(f)

        logger.info("\nRight-rail data keys:")
        for key in rr_data.keys():
            logger.info(f"  - {key}")

        # Check shotsByPeriod
        if "shotsByPeriod" in rr_data:
            logger.info("\n  shotsByPeriod structure:")
            shots_data = rr_data["shotsByPeriod"]
            if isinstance(shots_data, list) and shots_data:
                logger.info(f"    Sample: {shots_data[0]}")

    # Check WSC data
    wsc_file = "data/endpoint_samples/wsc_play-by-play.json"
    if os.path.exists(wsc_file):
        with open(wsc_file, "r") as f:
            wsc_data = json.load(f)

        logger.info("\nWSC play-by-play structure:")
        if isinstance(wsc_data, list) and wsc_data:
            logger.info(f"  Number of events: {len(wsc_data)}")

            # Find shot events
            wsc_shots = [e for e in wsc_data if "shot" in str(e).lower()]
            if wsc_shots:
                logger.info(f"  Shot events found: {len(wsc_shots)}")
                logger.info(f"  Sample: {wsc_shots[0]}")

    # Summary and recommendations
    logger.info("\n" + "=" * 60)
    logger.info("📋 SUMMARY")
    logger.info("=" * 60)
    logger.info("\n1. NHL API play-by-play does NOT include shot speed")
    logger.info("2. Available shot features:")
    logger.info("   - Location (x,y coordinates)")
    logger.info("   - Shot type (wrist, slap, snap, etc.)")
    logger.info("   - Zone (offensive, neutral, defensive)")
    logger.info("   - Game situation")
    logger.info("\n3. To get shot speed, you'll need to:")
    logger.info("   - Use NHL Edge website scraping (different approach needed)")
    logger.info("   - Find alternative data sources (MoneyPuck, etc.)")
    logger.info("   - Use shot type as a proxy for shot quality")


if __name__ == "__main__":
    analyze_pbp_structure()
