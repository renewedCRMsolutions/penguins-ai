<NHL_AI_Implementation_Prompt version="1.0" created="2025-05-27">
<!-- PROJECT CONTEXT -->
<ProjectOverview>
    <name>Pittsburgh Penguins AI Analytics</name>
    <current_state>Frontend complete, API ready, needs real NHL data training</current_state>
    <immediate_goal>Train production xG model with real NHL API data</immediate_goal>
    <hardware_optimized>ThinkPad P16 Gen 2 (20 threads, 47GB RAM)</hardware_optimized>
</ProjectOverview>

<!-- CRITICAL PATH - DO THIS FIRST -->
<ImmediateImplementation>
    <Step1_DataCollection>
        <task>Fix and run train_nhl_optimized.py</task>
        <issues_to_fix>
            <issue>NHL API endpoints need proper error handling</issue>
            <issue>Date formatting for API calls</issue>
            <issue>Rate limiting to avoid API blocks</issue>
        </issues_to_fix>
        <modern_approach>
            <!-- Use async/await for API calls -->
            <!-- Implement exponential backoff -->
            <!-- Cache responses locally to avoid re-fetching -->
        </modern_approach>
    </Step1_DataCollection>

    <Step2_FeatureEngineering>
        <required_features>
            <feature name="shot_distance">sqrt((x-goal_x)^2 + (y-goal_y)^2)</feature>
            <feature name="shot_angle">atan2(y-goal_y, x-goal_x)</feature>
            <feature name="shot_type">wrist/slap/snap/backhand/tip/deflection</feature>
            <feature name="game_state">even/powerplay/shorthanded</feature>
            <feature name="period">1/2/3/OT</feature>
            <feature name="score_differential">team_score - opponent_score</feature>
            <feature name="time_elapsed">seconds since period start</feature>
            <feature name="is_rebound">previous_event was shot within 3 seconds</feature>
            <feature name="is_rush">zone entry within 4 seconds</feature>
        </required_features>
    </Step2_FeatureEngineering>

    <Step3_ModelTraining>
        <xgboost_config>
            <param name="n_estimators">1000</param>
            <param name="max_depth">6</param>
            <param name="learning_rate">0.01</param>
            <param name="n_jobs">20</param>
            <param name="tree_method">hist</param>
            <param name="predictor">cpu_predictor</param>
            <param name="early_stopping_rounds">50</param>
        </xgboost_config>
        <validation>
            <method>TimeSeriesSplit (5 folds)</method>
            <metrics>AUC, Brier Score, Log Loss</metrics>
            <target>AUC > 0.75</target>
        </validation>
    </Step3_ModelTraining>
</ImmediateImplementation>

<!-- WORKING CODE STRUCTURE -->
<CodeImplementation>
    
    <File name="train/train_nhl_optimized.py">
        <purpose>Single training script that handles everything</purpose>
        <structure>
pythonimport asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NHLDataCollector:
    """Collect NHL shot data from official API"""
    BASE_URL = "https://api-web.nhle.com/v1"
    
    def __init__(self):
        self.shots_data = []
        
    async def fetch_with_retry(self, session, url, retries=3):
        """Fetch with exponential backoff"""
        for attempt in range(retries):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt == retries - 1:
                    raise
        return None
    
    async def get_games_for_date(self, session, date_str):
        """Get all games for a specific date"""
        url = f"{self.BASE_URL}/score/{date_str}"
        data = await self.fetch_with_retry(session, url)
        return data.get('games', []) if data else []
    
    async def process_game(self, session, game_id):
        """Extract shot data from a single game"""
        url = f"{self.BASE_URL}/gamecenter/{game_id}/play-by-play"
        data = await self.fetch_with_retry(session, url)
        
        if not data:
            return
        
        shots = []
        for play in data.get('plays', []):
            if play['typeDescKey'] in ['shot-on-goal', 'goal']:
                # Extract all available features
                details = play.get('details', {})
                shot = {
                    'game_id': game_id,
                    'period': play.get('period'),
                    'time_in_period': play.get('timeInPeriod'),
                    'x_coord': details.get('xCoord', 0),
                    'y_coord': details.get('yCoord', 0),
                    'shot_type': details.get('shotType', 'unknown'),
                    'is_goal': play['typeDescKey'] == 'goal',
                    'situation_code': play.get('situationCode', ''),
                    'zone_code': details.get('zoneCode', ''),
                    'player_id': details.get('shootingPlayerId'),
                    'goalie_id': details.get('goalieInNetId'),
                }
                shots.append(shot)
        
        self.shots_data.extend(shots)
        logger.info(f"Processed game {game_id}: {len(shots)} shots")
    
    async def collect_data(self, days_back=30):
        """Collect data for the last N days"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for day in range(days_back):
                date = datetime.now() - timedelta(days=day)
                date_str = date.strftime("%Y-%m-%d")
                
                games = await self.get_games_for_date(session, date_str)
                
                for game in games:
                    if game.get('gameState') == 'OFF':  # Completed games
                        task = self.process_game(session, game['id'])
                        tasks.append(task)
                
                # Process in batches to avoid overwhelming API
                if len(tasks) >= 10:
                    await asyncio.gather(*tasks)
                    tasks = []
                    await asyncio.sleep(1)  # Rate limiting
            
            # Process remaining tasks
            if tasks:
                await asyncio.gather(*tasks)
        
        return pd.DataFrame(self.shots_data)

class FeatureEngineer:
    """Create xG model features"""
    
    @staticmethod
    def add_features(df):
        """Add all engineered features"""
        # Goal location (varies by rink end)
        df['goal_x'] = 89  # Assuming offensive zone
        df['goal_y'] = 0   # Center of net
        
        # Basic geometry
        df['shot_distance'] = np.sqrt(
            (df['x_coord'] - df['goal_x'])**2 + 
            (df['y_coord'] - df['goal_y'])**2
        )
        
        df['shot_angle'] = np.abs(np.arctan2(
            df['y_coord'] - df['goal_y'],
            df['goal_x'] - df['x_coord']
        )) * 180 / np.pi
        
        # Game state
        df['is_powerplay'] = df['situation_code'].str.contains('1[4-5]51', na=False)
        df['is_shorthanded'] = df['situation_code'].str.contains('1[3-4]51', na=False)
        
        # Shot type encoding
        shot_types = ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'deflection']
        for shot_type in shot_types:
            df[f'shot_type_{shot_type}'] = (df['shot_type'] == shot_type).astype(int)
        
        # Time features
        df['period_seconds'] = df['time_in_period'].apply(
            lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) 
            if pd.notna(x) and ':' in str(x) else 0
        )
        
        return df

class XGModelTrainer:
    """Train XGBoost model with optimal settings"""
    
    def __init__(self, n_jobs=20):
        self.n_jobs = n_jobs
        self.model = None
        
    def train(self, X, y):
        """Train model with time series validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        params = {
            'objective': 'binary:logistic',
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_jobs': self.n_jobs,
            'tree_method': 'hist',
            'random_state': 42,
            'early_stopping_rounds': 50,
            'eval_metric': 'auc'
        }
        
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            scores.append(auc)
            logger.info(f"Fold AUC: {auc:.4f}")
        
        logger.info(f"Average AUC: {np.mean(scores):.4f}")
        
        # Train final model on all data
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X, y)
        
        return self.model

async def main():
    """Main training pipeline"""
    logger.info("Starting NHL xG model training...")
    
    # Step 1: Collect data
    collector = NHLDataCollector()
    df = await collector.collect_data(days_back=30)
    
    if df.empty:
        logger.error("No data collected!")
        return
    
    logger.info(f"Collected {len(df)} shots")
    
    # Save raw data
    df.to_csv('data/nhl/shots_raw.csv', index=False)
    
    # Step 2: Feature engineering
    engineer = FeatureEngineer()
    df = engineer.add_features(df)
    
    # Select features for model
    feature_cols = [
        'shot_distance', 'shot_angle', 'period', 'period_seconds',
        'is_powerplay', 'is_shorthanded'
    ] + [col for col in df.columns if col.startswith('shot_type_')]
    
    X = df[feature_cols]
    y = df['is_goal']
    
    # Step 3: Train model
    trainer = XGModelTrainer(n_jobs=20)
    model = trainer.train(X, y)
    
    # Save model
    joblib.dump(model, 'models/production/xg_model_nhl.pkl')
    logger.info("Model saved to models/production/xg_model_nhl.pkl")
    
    # Save feature names for API
    with open('models/production/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_cols))

if __name__ == "__main__":
    asyncio.run(main())
        </structure>
    </File>

    <File name="api/main.py">
        <changes_needed>
            <change>Update model path to models/production/xg_model_nhl.pkl</change>
            <change>Load feature names from models/production/feature_names.txt</change>
            <change>Add model versioning endpoint</change>
        </changes_needed>
    </File>

</CodeImplementation>

<!-- MODERN BEST PRACTICES -->
<ModernTechniques>
    <DataHandling>
        <technique>Async/await for all API calls</technique>
        <technique>Pandas 2.0 with PyArrow backend for speed</technique>
        <technique>Polars for even faster data processing (optional)</technique>
    </DataHandling>
    
    <ModelOptimizations>
        <technique>XGBoost hist method for speed</technique>
        <technique>Early stopping to prevent overfitting</technique>
        <technique>Time-based validation (not random)</technique>
        <technique>Feature importance analysis</technique>
    </ModelOptimizations>
    
    <ProductionReadiness>
        <technique>Comprehensive logging</technique>
        <technique>Error handling with retries</technique>
        <technique>Model versioning</technique>
        <technique>Performance metrics tracking</technique>
    </ProductionReadiness>
</ModernTechniques>

<!-- VALIDATION & TESTING -->
<ValidationStrategy>
    <Metrics>
        <metric>AUC-ROC (target > 0.75)</metric>
        <metric>Brier Score (calibration)</metric>
        <metric>Expected Calibration Error</metric>
    </Metrics>
    
    <Tests>
        <test>Known high-danger shots have xG > 0.3</test>
        <test>Long shots have xG < 0.05</test>
        <test>Power play shots have higher xG than even strength</test>
    </Tests>
</ValidationStrategy>

<!-- NEXT STEPS AFTER TRAINING -->
<PostTraining>
    <Step1>
        <action>Update API to use new model</action>
        <verify>Test endpoint with sample shots</verify>
    </Step1>
    
    <Step2>
        <action>Create model performance dashboard</action>
        <features>
            - Feature importance plot
            - Calibration curve
            - Shot location heat map
        </features>
    </Step2>
    
    <Step3>
        <action>Set up daily retraining pipeline</action>
        <automation>GitHub Actions or cron job</automation>
    </Step3>
</PostTraining>

<!-- TROUBLESHOOTING -->
<CommonIssues>
    <Issue>
        <problem>NHL API returns 429 (rate limited)</problem>
        <solution>Implement exponential backoff, reduce concurrent requests</solution>
    </Issue>
    
    <Issue>
        <problem>Not enough goal events (class imbalance)</problem>
        <solution>Use scale_pos_weight in XGBoost or SMOTE</solution>
    </Issue>
    
    <Issue>
        <problem>Memory issues with 64GB RAM</problem>
        <solution>Process data in chunks, use data types optimization</solution>
    </Issue>
</CommonIssues>
</NHL_AI_Implementation_Prompt>