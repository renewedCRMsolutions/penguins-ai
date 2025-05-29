# scripts/collect_nhl_season.py
import asyncio
import pandas as pd
from pathlib import Path
from nhlpy import NHLClient


class NHLSeasonCollector:
    def __init__(self):
        self.client = NHLClient()
        self.output_dir = Path("data/nhl/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def time_to_seconds(self, time_str):
        """Convert time string to seconds"""
        if isinstance(time_str, (int, float)):
            return time_str
        if isinstance(time_str, str) and ":" in time_str:
            minutes, seconds = time_str.split(":")
            return int(minutes) * 60 + int(seconds)
        return 0

    async def collect_season(self, start_date: str, end_date: str):
        print("Fetching game IDs for 2024-25 season...")
        game_ids = self.client.helpers.get_gameids_by_season("20242025", game_types=[2])
        print(f"Found {len(game_ids)} games")

        all_shots = []
        for i, game_id in enumerate(game_ids):
            if i % 10 == 0:
                print(f"Processing game {i + 1}/{len(game_ids)} - {len(all_shots)} shots collected")

            try:
                shots = await self.collect_game_shots(game_id)
                all_shots.extend(shots)
            except Exception as e:
                print(f"Error on game {game_id}: {e}")

            if i % 3 == 0:  # Rate limiting
                await asyncio.sleep(0.5)

        # Save data
        df = pd.DataFrame(all_shots)
        output_file = self.output_dir / f"nhl_shots_{start_date}_to_{end_date}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} shots to {output_file}")

    async def collect_game_shots(self, game_id: str):
        pbp = self.client.game_center.play_by_play(game_id)
        shots = []
        plays = pbp.get("plays", [])

        for idx, play in enumerate(plays):
            if play.get("typeDescKey") in ["shot-on-goal", "goal", "missed-shot", "blocked-shot"]:
                shot_data = self.extract_shot_features(play, pbp, idx)
                shots.append(shot_data)

        return shots

    def extract_shot_features(self, play, pbp, play_idx):
        plays = pbp.get("plays", [])

        # Convert play time
        play_time = self.time_to_seconds(play.get("timeInPeriod", 0))

        # Find previous plays for context
        prev_shot = None
        prev_event = None

        for i in range(play_idx - 1, -1, -1):
            if plays[i].get("typeDescKey") in ["shot-on-goal", "goal", "missed-shot", "blocked-shot"]:
                prev_shot = plays[i]
                break
            if plays[i].get("typeDescKey") and prev_event is None:
                prev_event = plays[i]

        # Calculate time since various events
        time_since_faceoff = 60.0  # default
        time_since_zone_entry = 30.0
        faceoff_location = None

        for i in range(play_idx - 1, -1, -1):
            event = plays[i]
            event_time = self.time_to_seconds(event.get("timeInPeriod", 0))

            if event.get("typeDescKey") == "faceoff":
                time_since_faceoff = play_time - event_time
                faceoff_location = event.get("details", {})
                break
            if event.get("typeDescKey") == "zone-entry":
                time_since_zone_entry = play_time - event_time

        # Get all play details
        details = play.get("details", {}) or {}
        period_desc = play.get("periodDescriptor", {}) or {}
        situation = play.get("situationCode", "")

        # Convert previous event times
        prev_event_time = self.time_to_seconds(prev_event.get("timeInPeriod", 0)) if prev_event else 0
        prev_shot_time = self.time_to_seconds(prev_shot.get("timeInPeriod", 0)) if prev_shot else 0

        # Extract all features
        return {
            # Game info
            "game_id": pbp.get("id"),
            "season": pbp.get("season"),
            "game_type": pbp.get("gameType"),
            "game_date": pbp.get("gameDate"),
            "venue": pbp.get("venue", {}).get("default"),
            # Time info
            "period": period_desc.get("number"),
            "period_type": period_desc.get("periodType"),
            "time_in_period": play_time,
            "time_remaining": play.get("timeRemaining", 0),
            "game_seconds": (period_desc.get("number", 1) - 1) * 1200 + play_time,
            # Shot details
            "event_type": play.get("typeDescKey"),
            "x_coord": details.get("xCoord", 0),
            "y_coord": details.get("yCoord", 0),
            "zone_code": details.get("zoneCode"),
            "shot_type": details.get("shotType"),
            "is_goal": 1 if play.get("typeDescKey") == "goal" else 0,
            "reason": details.get("reason"),
            # Calculated shot metrics
            "shot_distance": self.calculate_shot_distance(
                details.get("xCoord", 0), details.get("yCoord", 0), details.get("attackingTeamId")
            ),
            "shot_angle": self.calculate_shot_angle(
                details.get("xCoord", 0), details.get("yCoord", 0), details.get("attackingTeamId")
            ),
            # Players
            "shooter_id": details.get("shootingPlayerId"),
            "shooter_name": f"{details.get('shootingPlayerFirstName', '')} "
            f"{details.get('shootingPlayerLastName', '')}".strip(),
            "goalie_id": details.get("goalieInNetId"),
            "goalie_name": f"{details.get('goalieInNetFirstName', '')} "
            f"{details.get('goalieInNetLastName', '')}".strip(),
            "assist1_id": details.get("assist1PlayerId"),
            "assist1_name": f"{details.get('assist1FirstName', '')} " f"{details.get('assist1LastName', '')}".strip(),
            "assist2_id": details.get("assist2PlayerId"),
            "assist2_name": f"{details.get('assist2FirstName', '')} " f"{details.get('assist2LastName', '')}".strip(),
            "blocker_id": details.get("blockingPlayerId"),
            # Team info
            "shooting_team_id": details.get("eventOwnerTeamId"),
            "shooting_team": (
                pbp.get("homeTeam", {}).get("abbrev")
                if details.get("eventOwnerTeamId") == pbp.get("homeTeam", {}).get("id")
                else pbp.get("awayTeam", {}).get("abbrev")
            ),
            "home_team_id": pbp.get("homeTeam", {}).get("id"),
            "home_team": pbp.get("homeTeam", {}).get("abbrev"),
            "away_team_id": pbp.get("awayTeam", {}).get("id"),
            "away_team": pbp.get("awayTeam", {}).get("abbrev"),
            "is_home_team": 1 if details.get("eventOwnerTeamId") == pbp.get("homeTeam", {}).get("id") else 0,
            # Score state
            "home_score": details.get("homeScore", 0),
            "away_score": details.get("awayScore", 0),
            "score_differential": self.calculate_score_diff(details, pbp),
            "is_tied": 1 if details.get("homeScore", 0) == details.get("awayScore", 0) else 0,
            "is_leading": self.is_leading(details, pbp),
            # Game state
            "home_skaters": int(situation[0]) if situation else 5,
            "away_skaters": int(situation[1]) if len(situation) > 1 else 5,
            "strength_state": situation if situation else "55",
            "is_powerplay": 1 if situation and situation[0] != situation[1] else 0,
            "is_shorthanded": self.is_shorthanded(situation, details, pbp),
            "empty_net": 1 if details.get("emptyNet") else 0,
            "is_penalty_shot": 1 if details.get("isPenaltyShot") else 0,
            # Previous event context
            "prev_event_type": prev_event.get("typeDescKey") if prev_event else None,
            "prev_event_x": prev_event.get("details", {}).get("xCoord") if prev_event else None,
            "prev_event_y": prev_event.get("details", {}).get("yCoord") if prev_event else None,
            "prev_event_team": prev_event.get("details", {}).get("eventOwnerTeamId") if prev_event else None,
            "time_since_prev_event": play_time - prev_event_time if prev_event else 0,
            "distance_from_prev_event": self.calculate_distance(play, prev_event) if prev_event else 0,
            # Shot context
            "time_since_prev_shot": play_time - prev_shot_time if prev_shot else 60,
            "prev_shot_result": prev_shot.get("typeDescKey") if prev_shot else None,
            "shots_in_sequence": self.count_shot_sequence(plays, play_idx),
            # Faceoff context
            "time_since_faceoff": time_since_faceoff,
            "faceoff_win": self.check_faceoff_win(plays, play_idx, details.get("eventOwnerTeamId")),
            "faceoff_zone": faceoff_location.get("zoneCode") if faceoff_location else None,
            "is_off_zone_faceoff": (
                1 if faceoff_location and faceoff_location.get("zoneCode") == "O" and time_since_faceoff < 5 else 0
            ),
            # Rush/Rebound indicators
            "is_rebound": self.is_rebound(play, prev_shot, play_time, prev_shot_time),
            "is_rush": self.is_rush(play, plays, play_idx, play_time),
            "is_one_timer": (
                1 if prev_event and prev_event.get("typeDescKey") == "pass" and play_time - prev_event_time < 2 else 0
            ),
            "speed_from_prev": self.calculate_speed(play, prev_event, play_time - prev_event_time) if prev_event else 0,
            # Zone time
            "time_since_zone_entry": time_since_zone_entry,
            "offensive_zone_time": self.calculate_oz_time(plays, play_idx),
            # Additional flags
            "is_backhand": 1 if details.get("shotType") == "backhand" else 0,
            "is_deflection": 1 if details.get("shotType") in ["deflected", "tip-in"] else 0,
            "is_wraparound": 1 if details.get("shotType") == "wrap-around" else 0,
            "playoff_game": 1 if pbp.get("gameType") == 3 else 0,
        }

    def is_rebound(self, play, prev_shot, play_time, prev_shot_time):
        if not prev_shot:
            return 0
        time_diff = play_time - prev_shot_time
        return 1 if time_diff < 3 and prev_shot.get("typeDescKey") in ["shot-on-goal", "goal"] else 0

    def is_rush(self, play, plays, play_idx, play_time):
        # Check if shot came quickly after neutral zone event
        for i in range(max(0, play_idx - 10), play_idx):
            event = plays[i]
            if event.get("details", {}).get("zoneCode") == "N":
                event_time = self.time_to_seconds(event.get("timeInPeriod", 0))
                time_diff = play_time - event_time
                if time_diff < 5:
                    return 1
        return 0

    def calculate_speed(self, play, prev_event, time_diff):
        if not prev_event or time_diff <= 0:
            return 0
        distance = self.calculate_distance(play, prev_event)
        return distance / time_diff

    def calculate_shot_distance(self, x, y, attacking_team):
        # NHL rink: goals at x=89 or x=-89
        goal_x = 89 if x > 0 else -89
        return ((x - goal_x) ** 2 + y**2) ** 0.5

    def calculate_shot_angle(self, x, y, attacking_team):
        import math

        goal_x = 89 if x > 0 else -89
        return abs(math.degrees(math.atan2(y, goal_x - x)))

    def calculate_distance(self, play1, play2):
        if not play2:
            return 0
        x1 = play1.get("details", {}).get("xCoord", 0)
        y1 = play1.get("details", {}).get("yCoord", 0)
        x2 = play2.get("details", {}).get("xCoord", 0)
        y2 = play2.get("details", {}).get("yCoord", 0)
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def calculate_score_diff(self, details, pbp):
        home_score = details.get("homeScore", 0)
        away_score = details.get("awayScore", 0)
        if details.get("eventOwnerTeamId") == pbp.get("homeTeam", {}).get("id"):
            return home_score - away_score
        return away_score - home_score

    def is_leading(self, details, pbp):
        score_diff = self.calculate_score_diff(details, pbp)
        return 1 if score_diff > 0 else 0

    def is_shorthanded(self, situation, details, pbp):
        if not situation or len(situation) < 2:
            return 0
        home_skaters = int(situation[0])
        away_skaters = int(situation[1])
        if details.get("eventOwnerTeamId") == pbp.get("homeTeam", {}).get("id"):
            return 1 if home_skaters < away_skaters else 0
        return 1 if away_skaters < home_skaters else 0

    def count_shot_sequence(self, plays, play_idx):
        count = 1
        play_time = self.time_to_seconds(plays[play_idx].get("timeInPeriod", 0))
        for i in range(play_idx - 1, max(0, play_idx - 10), -1):
            if plays[i].get("typeDescKey") in ["shot-on-goal", "goal", "missed-shot", "blocked-shot"]:
                event_time = self.time_to_seconds(plays[i].get("timeInPeriod", 0))
                if play_time - event_time < 10:
                    count += 1
                else:
                    break
        return count

    def check_faceoff_win(self, plays, play_idx, shooting_team):
        play_time = self.time_to_seconds(plays[play_idx].get("timeInPeriod", 0))
        for i in range(play_idx - 1, -1, -1):
            if plays[i].get("typeDescKey") == "faceoff":
                winning_player = plays[i].get("details", {}).get("winningPlayerId")
                return 1 if winning_player == shooting_team else 0  # Fixed: compare int to int
            event_time = self.time_to_seconds(plays[i].get("timeInPeriod", 0))
            if event_time < play_time - 60:
                break
        return 0

    def calculate_oz_time(self, plays, play_idx):
        oz_time = 0
        for i in range(play_idx - 1, max(0, play_idx - 50), -1):
            if plays[i].get("details", {}).get("zoneCode") == "O":
                if i + 1 < len(plays):
                    time1 = self.time_to_seconds(plays[i].get("timeInPeriod", 0))
                    time2 = self.time_to_seconds(plays[i + 1].get("timeInPeriod", 0))
                    oz_time += time2 - time1
                else:
                    oz_time += 1
            if plays[i].get("details", {}).get("zoneCode") in ["N", "D"]:
                break
        return oz_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    collector = NHLSeasonCollector()
    asyncio.run(collector.collect_season(args.start, args.end))
