# File: scripts/check_nhl_features.py

import requests

# Check what data is available
game_id = "2024020500"

# 1. Enhanced play-by-play
pbp = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play").json()
print("Play-by-play fields:", pbp['plays'][0].keys())

# 2. Shift charts for player positions
shifts = requests.get(f"https://api-web.nhle.com/v1/shift-charts/{game_id}").json()
print("Shift data available:", shifts.keys())

# 3. Game boxscore for context
boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore").json()
print("Boxscore data:", boxscore.keys())