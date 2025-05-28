# File: scripts/fetch_all_nhl_endpoints.py

import requests
import json
import os
# from datetime import datetime, timedelta  # Unused imports
import time

class NHLAPIExplorer:
    """Fetch JSON from every NHL API endpoint"""
    
    def __init__(self):
        self.base_url = "https://api-web.nhle.com/v1"
        self.output_dir = "data/nhl_api_samples"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fetch_all_endpoints(self):
        """Fetch data from every endpoint in the NHL API Reference"""
        
        print("🏒 NHL API COMPLETE ENDPOINT EXPLORATION")
        print("=" * 80)
        
        # Sample IDs for testing
        game_id = "2024020500"
        player_id = "8478402"  # Connor McDavid
        team = "PIT"
        date = "2024-01-15"
        season = "20232024"
        
        # ALL ENDPOINTS FROM THE REFERENCE
        endpoints = {
            # GAME ENDPOINTS
            "scoreboard": f"/score/{date}",
            "game_landing": f"/gamecenter/{game_id}/landing",
            "game_boxscore": f"/gamecenter/{game_id}/boxscore",
            "game_playbyplay": f"/gamecenter/{game_id}/play-by-play",
            "game_shiftcharts": f"/shift-charts/{game_id}",
            "game_video": f"/gameVideo/{game_id}",
            
            # PLAYER ENDPOINTS
            "player_landing": f"/player/{player_id}/landing",
            "player_gamelog": f"/player/{player_id}/game-log/{season}/2",
            "player_specific_stats": f"/player/{player_id}/specific-stats?season={season}",
            "player_career_stats": f"/player-stats/{player_id}/summary?seasonType=career",
            
            # TEAM ENDPOINTS
            "team_roster": f"/roster/{team}/current",
            "team_prospects": f"/prospects/{team}",
            "team_roster_season": f"/roster-season/{team}/{season}",
            "team_schedule": f"/club-schedule-season/{team}/{season}",
            "team_stats": f"/club-stats-season/{team}",
            
            # STANDINGS & STATS
            "standings": f"/standings/{date}",
            "standings_season": f"/standings-season",
            "playoff_bracket": f"/playoff-bracket/{season}",
            
            # LEAGUE STATS
            "skater_leaders": f"/leaders/skaters/points?limit=10",
            "goalie_leaders": f"/leaders/goalies/wins?limit=10",
            "team_stats_leaders": f"/stats/team?limit=10",
            
            # DRAFT
            "draft_rankings": f"/draft/rankings/now",
            "draft_prospects": f"/draft/prospects/2024",
            
            # SCHEDULE
            "schedule_calendar": f"/schedule-calendar/{date}",
            "season_schedule": f"/schedule/{season}",
            
            # CONTENT
            "game_story": f"/game-story/{game_id}",
            "game_recap": f"/recap/{game_id}",
            
            # CONFIGURATION
            "glossary": "/glossary",
            "config": "/config",
            "countries": "/countries",
            "franchises": "/franchises",
            "teams": "/teams",
            "drafts": "/drafts",
            "seasons": "/seasons",
            "trophy": "/trophy",
            "awards": "/awards",
            "venues": "/venues"
        }
        
        # Fetch each endpoint
        for name, endpoint in endpoints.items():
            print(f"\n📊 Fetching: {name}")
            print(f"   URL: {self.base_url}{endpoint}")
            
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Save to file
                    filename = f"{self.output_dir}/{name}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"   ✅ Saved to {filename}")
                    
                    # Show sample of what we got
                    self.show_data_structure(name, data)
                    
                else:
                    print(f"   ❌ Error: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Failed: {str(e)}")
            
            # Rate limiting
            time.sleep(0.5)
    
    def show_data_structure(self, name, data):
        """Show the structure of the JSON response"""
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())[:10]}")
            
            # Special handling for play-by-play
            if name == "game_playbyplay" and 'plays' in data:
                print(f"   Total plays: {len(data['plays'])}")
                # Find shot events
                shots = [p for p in data['plays'] if 'shot' in p.get('typeDescKey', '')]
                if shots:
                    print(f"   Shot example fields: {list(shots[0].keys())}")
                    if 'details' in shots[0]:
                        print(f"   Shot details: {list(shots[0]['details'].keys())}")
        
        elif isinstance(data, list):
            print(f"   List with {len(data)} items")
            if data:
                print(f"   First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
    
    def analyze_shot_features(self):
        """Analyze all shot-related features from the data"""
        print("\n\n🎯 ANALYZING SHOT FEATURES FOR XG MODEL")
        print("=" * 80)
        
        # Load play-by-play data
        pbp_file = f"{self.output_dir}/game_playbyplay.json"
        if os.path.exists(pbp_file):
            with open(pbp_file, 'r') as f:
                data = json.load(f)
            
            all_features = set()
            shot_features = set()
            
            for play in data.get('plays', []):
                # Collect all play features
                all_features.update(play.keys())
                
                # For shots, collect detailed features
                if 'shot' in play.get('typeDescKey', '') or play.get('typeDescKey') == 'goal':
                    shot_features.update(play.keys())
                    if 'details' in play:
                        shot_features.update([f"details.{k}" for k in play['details'].keys()])
            
            print("\nALL AVAILABLE SHOT FEATURES:")
            for feature in sorted(shot_features):
                print(f"  - {feature}")
            
            # Save feature list
            with open(f"{self.output_dir}/available_features.txt", 'w') as f:
                f.write("SHOT FEATURES AVAILABLE IN NHL API:\n")
                f.write("=" * 50 + "\n")
                for feature in sorted(shot_features):
                    f.write(f"{feature}\n")

if __name__ == "__main__":
    explorer = NHLAPIExplorer()
    
    # Fetch all endpoints
    explorer.fetch_all_endpoints()
    
    # Analyze what we got
    explorer.analyze_shot_features()
    
    print("\n\n✅ COMPLETE! Check data/nhl_api_samples/ for all JSON files")
    print("💡 Now we can see EVERY feature available to improve our model!")