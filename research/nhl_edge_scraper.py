# File: research/nhl_edge_scraper.py
# Scrape shot speed and advanced tracking data from NHL Edge

import asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
import json
import time
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHLEdgeScraper:
    """Scrape NHL Edge for shot speed and tracking data"""

    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    def __init__(self, headless=True):
        self.base_url = "https://edge.nhl.com"
        self.headless = headless
        self.driver = None
        self.scraped_data = {"shot_speeds": [], "player_speeds": [], "zone_times": [], "tracking_data": []}

    def setup_driver(self):
        """Initialize Chrome driver with optimal settings"""
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        # Disable images for faster loading
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)

        self.driver = webdriver.Chrome(options=options)
        logger.info("✅ Chrome driver initialized")

    def close_driver(self):
        """Close the driver"""
        if self.driver:
            self.driver.quit()
            logger.info("Driver closed")

    def scrape_shot_speeds(self, season="20242025", limit=None):
        """Scrape shot speed data for all players"""
        logger.info(f"🎯 Scraping shot speeds for season {season}")

        try:
            # Navigate to shot speed stats
            url = f"{self.base_url}/en/stats/nhl/skaters/shots?season={season}&seasonType=REG"
            if self.driver is None:
                raise RuntimeError("Driver not initialized. Call setup_driver() first.")
            self.driver.get(url)

            # Wait for data to load
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "rt-tbody")))
            time.sleep(2)  # Extra wait for JavaScript rendering

            # Extract player shot data
            players_data = []

            # Find all player rows
            player_rows = self.driver.find_elements(By.CSS_SELECTOR, ".rt-tr-group")

            for i, row in enumerate(player_rows):
                if limit and i >= limit:
                    break

                try:
                    # Extract player data
                    cells = row.find_elements(By.CSS_SELECTOR, ".rt-td")

                    if len(cells) > 10:  # Ensure we have enough data
                        player_data = {
                            "player_name": cells[1].text.strip(),
                            "team": cells[2].text.strip(),
                            "position": cells[3].text.strip(),
                            "games_played": self._parse_number(cells[4].text),
                            "goals": self._parse_number(cells[5].text),
                            "shots": self._parse_number(cells[6].text),
                            "avg_shot_speed": self._parse_float(cells[7].text),
                            "max_shot_speed": self._parse_float(cells[8].text),
                            "shot_speed_above_95": self._parse_number(cells[9].text),
                            "shot_speed_above_100": self._parse_number(cells[10].text),
                            "season": season,
                        }

                        players_data.append(player_data)
                        logger.info(
                            f"  ✓ {player_data['player_name']}: "
                            f"Avg {player_data['avg_shot_speed']} mph, "
                            f"Max {player_data['max_shot_speed']} mph"
                        )

                except Exception as e:
                    logger.error(f"Error parsing row {i}: {e}")
                    continue

            self.scraped_data["shot_speeds"] = players_data
            logger.info(f"✅ Scraped {len(players_data)} players' shot speed data")

            # Try to get more detailed shot-by-shot data
            self._scrape_detailed_shots(season)

            return players_data

        except Exception as e:
            logger.error(f"Error scraping shot speeds: {e}")
            return []

    def _scrape_detailed_shots(self, season):
        """Try to get individual shot data with speeds"""
        logger.info("🎯 Attempting to scrape detailed shot data...")

        try:
            # Navigate to shots section
            url = f"{self.base_url}/en/stats/nhl/shots?season={season}"
            if self.driver is None:
                raise RuntimeError("Driver not initialized. Call setup_driver() first.")
            self.driver.get(url)
            time.sleep(3)

            # Look for shot details
            shot_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-shot-speed]")

            detailed_shots = []
            for element in shot_elements[:100]:  # Limit for testing
                shot_data = {
                    "shot_id": element.get_attribute("data-shot-id"),
                    "player": element.get_attribute("data-player"),
                    "speed_mph": self._parse_float(element.get_attribute("data-shot-speed")),
                    "distance": self._parse_float(element.get_attribute("data-shot-distance")),
                    "angle": self._parse_float(element.get_attribute("data-shot-angle")),
                    "result": element.get_attribute("data-shot-result"),
                }
                detailed_shots.append(shot_data)

            if detailed_shots:
                self.scraped_data["detailed_shots"] = detailed_shots
                logger.info(f"✅ Found {len(detailed_shots)} detailed shots")

        except Exception as e:
            logger.info(f"Could not find detailed shot data: {e}")

    def scrape_player_speeds(self, season="20242025", limit=50):
        """Scrape player skating speed data"""
        logger.info(f"⚡ Scraping player speeds for season {season}")

        try:
            url = f"{self.base_url}/en/stats/nhl/skaters/skating?season={season}"
            if self.driver is None:
                raise RuntimeError("Driver not initialized. Call setup_driver() first.")
            self.driver.get(url)

            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "rt-tbody")))
            time.sleep(2)

            player_speeds = []
            player_rows = self.driver.find_elements(By.CSS_SELECTOR, ".rt-tr-group")

            for i, row in enumerate(player_rows):
                if limit and i >= limit:
                    break

                try:
                    cells = row.find_elements(By.CSS_SELECTOR, ".rt-td")

                    if len(cells) > 8:
                        speed_data = {
                            "player_name": cells[1].text.strip(),
                            "team": cells[2].text.strip(),
                            "avg_speed": self._parse_float(cells[4].text),
                            "max_speed": self._parse_float(cells[5].text),
                            "speed_bursts_18plus": self._parse_number(cells[6].text),
                            "speed_bursts_20plus": self._parse_number(cells[7].text),
                            "speed_bursts_22plus": self._parse_number(cells[8].text),
                            "season": season,
                        }

                        player_speeds.append(speed_data)
                        logger.info(f"  ✓ {speed_data['player_name']}: Max speed {speed_data['max_speed']} mph")

                except Exception as e:
                    logger.error(f"Error parsing speed row {i}: {e}")
                    continue

            self.scraped_data["player_speeds"] = player_speeds
            logger.info(f"✅ Scraped {len(player_speeds)} players' speed data")

            return player_speeds

        except Exception as e:
            logger.error(f"Error scraping player speeds: {e}")
            return []

    def scrape_zone_time(self, season="20242025"):
        """Scrape offensive zone time percentages"""
        logger.info(f"📍 Scraping zone time data for season {season}")

        try:
            url = f"{self.base_url}/en/stats/nhl/teams/puck-possession?season={season}"
            if self.driver is None:
                raise RuntimeError("Driver not initialized. Call setup_driver() first.")
            self.driver.get(url)

            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "rt-tbody")))
            time.sleep(2)

            zone_data = []
            team_rows = self.driver.find_elements(By.CSS_SELECTOR, ".rt-tr-group")

            for row in team_rows:
                try:
                    cells = row.find_elements(By.CSS_SELECTOR, ".rt-td")

                    if len(cells) > 6:
                        team_data = {
                            "team": cells[1].text.strip(),
                            "offensive_zone_time_pct": self._parse_float(cells[3].text),
                            "neutral_zone_time_pct": self._parse_float(cells[4].text),
                            "defensive_zone_time_pct": self._parse_float(cells[5].text),
                            "offensive_zone_time_per_game": cells[6].text.strip(),
                            "season": season,
                        }

                        zone_data.append(team_data)
                        logger.info(f"  ✓ {team_data['team']}: OZ Time {team_data['offensive_zone_time_pct']}%")

                except Exception as e:
                    logger.error(f"Error parsing zone row: {e}")
                    continue

            self.scraped_data["zone_times"] = zone_data
            logger.info(f"✅ Scraped {len(zone_data)} teams' zone time data")

            return zone_data

        except Exception as e:
            logger.error(f"Error scraping zone times: {e}")
            return []

    def scrape_all_data(self, season="20242025"):
        """Scrape all available NHL Edge data"""
        logger.info("\n🏒 STARTING COMPREHENSIVE NHL EDGE SCRAPE")
        logger.info(f"Season: {season}")
        logger.info("=" * 60)

        self.setup_driver()

        try:
            # Scrape all data types
            self.scrape_shot_speeds(season, limit=100)  # Top 100 shooters
            self.scrape_player_speeds(season, limit=100)  # Top 100 skaters
            self.scrape_zone_time(season)

            # Save all data
            self.save_scraped_data()

            # Generate summary
            self.generate_summary()

        finally:
            self.close_driver()

    def save_scraped_data(self):
        """Save all scraped data to files"""
        os.makedirs("data/nhl_edge", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        with open(f"data/nhl_edge/scraped_data_{timestamp}.json", "w") as f:
            json.dump(self.scraped_data, f, indent=2)

        # Save as CSV files
        for data_type, data in self.scraped_data.items():
            if data:
                df = pd.DataFrame(data)
                df.to_csv(f"data/nhl_edge/{data_type}_{timestamp}.csv", index=False)
                logger.info(f"💾 Saved {data_type} to CSV")

        # Create a merged dataset for training
        self.create_training_dataset()

    def create_training_dataset(self):
        """Create a merged dataset ready for model training"""
        logger.info("\n🔧 Creating training dataset...")

        # Convert to DataFrames
        shot_speeds_df = pd.DataFrame(self.scraped_data.get("shot_speeds", []))
        player_speeds_df = pd.DataFrame(self.scraped_data.get("player_speeds", []))

        if not shot_speeds_df.empty and not player_speeds_df.empty:
            # Merge player data
            merged_df = pd.merge(shot_speeds_df, player_speeds_df, on=["player_name", "team", "season"], how="outer")

            # Save merged data
            merged_df.to_csv("data/nhl_edge/player_tracking_features.csv", index=False)
            logger.info(f"✅ Created training dataset with {len(merged_df)} players")

            # Create feature lookup table
            feature_lookup = merged_df.set_index("player_name")[
                ["avg_shot_speed", "max_shot_speed", "avg_speed", "max_speed"]
            ].to_dict("index")

            # Save lookup table for easy access during training
            with open("data/nhl_edge/player_feature_lookup.json", "w") as f:
                json.dump(feature_lookup, f, indent=2)

            logger.info("✅ Created player feature lookup table")

    def generate_summary(self):
        """Generate summary of scraped data"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 NHL EDGE SCRAPING SUMMARY")
        logger.info("=" * 60)

        # Shot speed insights
        if self.scraped_data["shot_speeds"]:
            shot_df = pd.DataFrame(self.scraped_data["shot_speeds"])
            logger.info("\n🎯 SHOT SPEED INSIGHTS:")
            logger.info(f"  Players scraped: {len(shot_df)}")
            logger.info(f"  Avg shot speed: {shot_df['avg_shot_speed'].mean():.1f} mph")
            logger.info(f"  Max shot speed: {shot_df['max_shot_speed'].max():.1f} mph")
            logger.info(f"  Players with 100+ mph shots: {shot_df['shot_speed_above_100'].sum()}")

            # Top 5 hardest shooters
            top_shooters = shot_df.nlargest(5, "max_shot_speed")[["player_name", "max_shot_speed"]]
            logger.info("\n  Top 5 Hardest Shooters:")
            for _, player in top_shooters.iterrows():
                logger.info(f"    {player['player_name']}: {player['max_shot_speed']} mph")

        # Player speed insights
        if self.scraped_data["player_speeds"]:
            speed_df = pd.DataFrame(self.scraped_data["player_speeds"])
            logger.info("\n⚡ PLAYER SPEED INSIGHTS:")
            logger.info(f"  Players scraped: {len(speed_df)}")
            logger.info(f"  Avg max speed: {speed_df['max_speed'].mean():.1f} mph")
            fastest_idx = speed_df['max_speed'].idxmax()
            fastest_name = speed_df.loc[fastest_idx, 'player_name']
            fastest_speed = speed_df['max_speed'].max()
            logger.info(
                f"  Fastest skater: {fastest_name} ({fastest_speed:.1f} mph)"
            )

        # Zone time insights
        if self.scraped_data["zone_times"]:
            zone_df = pd.DataFrame(self.scraped_data["zone_times"])
            logger.info("\n📍 ZONE TIME INSIGHTS:")
            logger.info(f"  Teams scraped: {len(zone_df)}")
            logger.info(f"  Avg offensive zone time: {zone_df['offensive_zone_time_pct'].mean():.1f}%")

            # Best offensive teams
            top_offensive = zone_df.nlargest(3, "offensive_zone_time_pct")[["team", "offensive_zone_time_pct"]]
            logger.info("\n  Top 3 Offensive Zone Teams:")
            for _, team in top_offensive.iterrows():
                logger.info(f"    {team['team']}: {team['offensive_zone_time_pct']}%")

        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Merge this data with your existing shot data")
        logger.info("2. Add shot_speed as a feature in your XGBoost model")
        logger.info("3. Use player_speed data for rush detection")
        logger.info("4. Incorporate zone_time for possession quality")
        logger.info("\n" + "=" * 60)

    def _parse_number(self, text):
        """Parse number from text, handling errors"""
        try:
            return int(text.replace(",", "").strip())
        except (ValueError, AttributeError):
            return 0

    def _parse_float(self, text):
        """Parse float from text, handling errors"""
        try:
            return float(text.replace(",", "").strip())
        except (ValueError, AttributeError):
            return 0.0


async def main():
    """Run the NHL Edge scraper"""
    scraper = NHLEdgeScraper(headless=False)  # Set to True for headless mode

    try:
        # Scrape all available data
        scraper.scrape_all_data(season="20242025")

        # Show sample of data for verification
        if scraper.scraped_data["shot_speeds"]:
            print("\n🎯 Sample Shot Speed Data:")
            for player in scraper.scraped_data["shot_speeds"][:5]:
                print(f"  {player['player_name']}: {player['avg_shot_speed']} mph average")

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
    finally:
        scraper.close_driver()


if __name__ == "__main__":
    # Note: You need to have Chrome and ChromeDriver installed
    # Install: pip install selenium pandas

    print("🏒 NHL EDGE SCRAPER")
    print("=" * 60)
    print("This will scrape:")
    print("  - Shot speeds (avg, max, 95+ mph, 100+ mph)")
    print("  - Player skating speeds")
    print("  - Zone time percentages")
    print("=" * 60)

    asyncio.run(main())
