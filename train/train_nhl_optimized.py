# File: penguins_ai/train/train_nhl_optimized.py
# Complete NHL AI Training Script with Real Data Collection

import sys
sys.path.append(".")
from config.optimal_settings import SYSTEM_OPTIMAL_CONFIG

import asyncio
import aiohttp
# import requests  # Unused import
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
# import json  # Unused import
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import time
import warnings
warnings.filterwarnings('ignore')


class OptimizedNHLTrainer:
    """NHL model training optimized for your P16 system"""

    def __init__(self):
        self.config = SYSTEM_OPTIMAL_CONFIG
        self.xgb_params = self.config["xgboost_config"].copy()
        self.base_url = "https://api-web.nhle.com/v1"

        # Configure GPU if available
        if self.config.get("use_gpu", False):
            self.xgb_params["tree_method"] = "gpu_hist"
            self.xgb_params["predictor"] = "gpu_predictor"
            self.xgb_params["gpu_id"] = 0
            print("🎮 GPU acceleration enabled for XGBoost")

        print("🖥️ System Configuration Loaded:")
        print(f"  - XGBoost threads: {self.xgb_params['n_jobs']}")
        print(f"  - Max memory: {self.config['max_memory_gb']}GB")
        print(f"  - Batch size: {self.config['batch_size']}")
        print(f"  - GPU enabled: {self.config['use_gpu']}")
        print(f"  - Tree method: {self.xgb_params.get('tree_method', 'auto')}")

    async def fetch_nhl_data(self, days_back=30):
        """Fetch real NHL shot data from the API"""
        print(f"\n📊 Fetching {days_back} days of NHL data...")
        all_shots = []
        games_processed = 0
        
        async with aiohttp.ClientSession() as session:
            end_date = datetime.now() - timedelta(days=1)
            
            for day in range(days_back):
                date = (end_date - timedelta(days=day)).strftime("%Y-%m-%d")
                
                # Get games for this date
                url = f"{self.base_url}/score/{date}"
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            games = data.get('games', [])
                            
                            if games:
                                print(f"  Date {date}: {len(games)} games found")
                            
                            # Process each completed game
                            for game in games:
                                if game.get('gameState') == 'OFF':  # Game finished
                                    game_id = game.get('id')
                                    home_team = game.get('homeTeam', {}).get('abbrev', 'UNK')
                                    away_team = game.get('awayTeam', {}).get('abbrev', 'UNK')
                                    print(f"    Processing: {away_team} @ {home_team}")
                                    
                                    shots = await self.extract_game_shots(session, game_id, date)
                                    all_shots.extend(shots)
                                    games_processed += 1
                                    
                except Exception as e:
                    print(f"  Error on {date}: {str(e)}")
                    
                await asyncio.sleep(0.5)  # Rate limiting
        
        df = pd.DataFrame(all_shots)
        print(f"\n✅ Data Collection Complete:")
        print(f"  - Games processed: {games_processed}")
        print(f"  - Total shots collected: {len(df)}")
        
        return df
    
    async def extract_game_shots(self, session, game_id, game_date):
        """Extract shot features from a single game"""
        shots = []
        
        try:
            url = f"{self.base_url}/gamecenter/{game_id}/play-by-play"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    plays = data.get('plays', [])
                    
                    # Process each play
                    for i, play in enumerate(plays):
                        if play.get('typeDescKey') in ['shot-on-goal', 'goal', 'missed-shot']:
                            shot_data = self.process_shot(play, plays[:i], game_id, game_date)
                            if shot_data:
                                shots.append(shot_data)
                                
        except Exception as e:
            print(f"      Error processing game {game_id}: {str(e)}")
            
        return shots
    
    def process_shot(self, play, previous_plays, game_id, game_date):
        """Extract features from a shot event"""
        details = play.get('details', {})
        
        # Get coordinates
        x_coord = details.get('xCoord', 0)
        y_coord = details.get('yCoord', 0)
        
        # Skip shots without coordinates
        if x_coord == 0 and y_coord == 0:
            return None
        
        # Calculate distance and angle from net
        # NHL rink: goal at x=89 (or -89), y=0
        goal_x = 89 if x_coord > 0 else -89
        goal_y = 0
        
        distance = np.sqrt((abs(goal_x) - abs(x_coord))**2 + (goal_y - y_coord)**2)
        
        # Calculate angle (in degrees)
        if distance > 0:
            angle = np.arctan2(abs(y_coord - goal_y), abs(goal_x - x_coord)) * 180 / np.pi
        else:
            angle = 0
        
        # Determine last event type and time since
        last_event_type = "None"
        time_since_last = 10.0  # Default
        
        if previous_plays:
            for prev_play in reversed(previous_plays[-5:]):  # Look at last 5 plays
                if prev_play.get('typeDescKey'):
                    last_event_type = self.simplify_event_type(prev_play.get('typeDescKey'))
                    # Could calculate actual time difference here
                    time_since_last = 5.0
                    break
        
        # Create shot record
        shot_data = {
            'game_id': game_id,
            'game_date': game_date,
            'period': play.get('periodDescriptor', {}).get('number', 1),
            'time_in_period': play.get('timeInPeriod', '00:00'),
            'shotDistance': round(distance, 2),
            'shotAngle': round(angle, 2),
            'shotType': details.get('shotType', 'Wrist'),
            'lastEventType': last_event_type,
            'timeSinceLast': time_since_last,
            'isRebound': 1 if self.is_rebound(previous_plays) else 0,
            'isRush': 1 if self.is_rush(previous_plays) else 0,
            'goal': 1 if play['typeDescKey'] == 'goal' else 0,
            'shooter_player_id': details.get('shootingPlayerId', 0),
            'goalie_player_id': details.get('goalieInNetId', 0),
            'x_coord': x_coord,
            'y_coord': y_coord,
            'situation_code': play.get('situationCode', '1551'),
            'home_score': play.get('homeScore', 0),
            'away_score': play.get('awayScore', 0)
        }
        
        return shot_data
    
    def simplify_event_type(self, event_type):
        """Simplify NHL event types to basic categories"""
        event_map = {
            'shot-on-goal': 'Shot',
            'missed-shot': 'Shot',
            'blocked-shot': 'Shot',
            'goal': 'Shot',
            'faceoff': 'Faceoff',
            'hit': 'Hit',
            'giveaway': 'Turnover',
            'takeaway': 'Turnover',
            'penalty': 'Penalty',
            'stoppage': 'Stoppage'
        }
        
        for key, value in event_map.items():
            if key in event_type.lower():
                return value
        
        return 'Other'
    
    def is_rebound(self, previous_plays, window=3):
        """Check if shot is a rebound"""
        if not previous_plays:
            return False
            
        for play in previous_plays[-window:]:
            event_type = play.get('typeDescKey', '')
            if any(shot_type in event_type for shot_type in ['shot', 'goal']):
                return True
        return False
    
    def is_rush(self, previous_plays, window=5):
        """Check if shot came from a rush"""
        if len(previous_plays) < 2:
            return False
            
        # Look for neutral zone events or quick zone changes
        for play in previous_plays[-window:]:
            details = play.get('details', {})
            zone = details.get('zoneCode', '')
            if zone == 'N':  # Neutral zone
                return True
                
        return False
    
    def calculate_traffic(self, shot, previous_plays):
        """Estimate traffic in front of net"""
        # Count hits and physical plays near the shot
        traffic_events = 0
        for play in previous_plays[-5:]:
            if play.get('typeDescKey') in ['hit', 'blocked-shot']:
                traffic_events += 1
        return min(traffic_events, 3)  # Cap at 3
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        print("\n🔧 Preparing features...")
        
        # Create additional features
        df['scoreState'] = df.apply(lambda x: 
            'tied' if x['home_score'] == x['away_score'] 
            else 'leading' if x['home_score'] > x['away_score'] 
            else 'trailing', axis=1)
        
        # Power play detection (simplified)
        df['isPowerPlay'] = df['situation_code'].apply(lambda x: 
            1 if x and (x[:2] in ['51', '41', '31']) else 0)
        
        # Shot danger based on location
        df['shotDanger'] = df.apply(lambda x: 
            'high' if x['shotDistance'] < 20 and x['shotAngle'] < 30 
            else 'medium' if x['shotDistance'] < 40 
            else 'low', axis=1)
        
        print(f"  Features created: {len(df.columns)}")
        print(f"  Goal rate: {df['goal'].mean():.1%}")
        
        return df
    
    def train_xgboost_model(self, df):
        """Train model with optimal settings"""
        print(f"\n🤖 Training XGBoost with {self.xgb_params['n_jobs']} threads...")
        
        # Show GPU status if available
        if self.xgb_params.get("tree_method") == "gpu_hist":
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"  🎮 GPU: {torch.cuda.get_device_name(0)}")
                    print(f"  📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            except:
                print("  ℹ️ GPU configured but PyTorch not available for detection")
        
        # Prepare features
        feature_cols = [
            'shotDistance', 'shotAngle', 'period', 'timeSinceLast',
            'isRebound', 'isRush', 'isPowerPlay'
        ]
        
        # One-hot encode categorical features
        categorical_cols = ['shotType', 'lastEventType', 'scoreState', 'shotDanger']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix_sep='_')
        
        # Get all feature columns
        all_features = feature_cols + [col for col in df_encoded.columns 
                                      if any(cat in col for cat in categorical_cols)]
        
        # Remove target and non-features
        features_to_use = [f for f in all_features if f in df_encoded.columns and f != 'goal']
        
        X = df_encoded[features_to_use].fillna(0)
        y = df_encoded['goal']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Test samples: {len(X_test):,}")
        print(f"  Features: {len(features_to_use)}")
        print(f"  Goal rate in train: {y_train.mean():.1%}")
        print(f"  Goal rate in test: {y_test.mean():.1%}")
        
        # Calculate scale_pos_weight for imbalanced classes
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"  Class weight ratio: {scale_pos_weight:.1f}")
        
        # Train model
        start_time = time.time()
        
        model = xgb.XGBClassifier(
            **self.xgb_params,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            early_stopping_rounds=50,
            verbosity=1
        )
        
        # Fit with evaluation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        train_time = time.time() - start_time
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✅ Training Complete!")
        print(f"  Training time: {train_time:.1f}s")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  AUC-ROC: {auc:.3f}")
        print(f"  Best iteration: {model.best_iteration}")
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Goal', 'Goal']))
        
        # Feature importance
        print("\n🎯 Top 10 Feature Importance:")
        importance_df = pd.DataFrame({
            'feature': features_to_use,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        for _, row in importance_df.iterrows():
            print(f"  {row['feature']:30s} {row['importance']:.3f}")
        
        # Save everything
        self.save_model(model, features_to_use, X_train, accuracy, auc, train_time)
        
        return model, auc
    
    def save_model(self, model, features, X_train, accuracy, auc, train_time):
        """Save model and metadata"""
        print("\n💾 Saving model...")
        
        # Create directories
        os.makedirs('models/production', exist_ok=True)
        
        # Save model
        joblib.dump(model, 'models/production/xg_model_nhl.pkl')
        joblib.dump(features, 'models/production/xg_features_nhl.pkl')
        
        # Save metadata
        metadata = {
            'accuracy': accuracy,
            'auc': auc,
            'features': features,
            'training_samples': len(X_train),
            'training_time': train_time,
            'best_iteration': model.best_iteration,
            'system_config': self.config,
            'trained_date': datetime.now().isoformat(),
            'model_params': self.xgb_params
        }
        joblib.dump(metadata, 'models/production/model_metadata.pkl')
        
        print(f"  ✓ Model saved to models/production/xg_model_nhl.pkl")
        print(f"  ✓ Metadata saved")
    
    async def run_training_pipeline(self, days_back=30):
        """Complete training pipeline"""
        print("\n🏒 NHL XG Model Training Pipeline - P16 Optimized")
        print("=" * 60)
        
        # Step 1: Fetch data
        df = await self.fetch_nhl_data(days_back)
        
        if len(df) < 1000:
            print("\n⚠️ Warning: Not enough data collected!")
            print(f"  Only {len(df)} shots found. Consider:")
            print(f"  - Increasing days_back (currently {days_back})")
            print(f"  - Checking if it's off-season")
            print(f"  - Verifying API connectivity")
            
            if len(df) < 100:
                print("\n❌ Too few samples to train. Exiting.")
                return
        
        # Save raw data
        os.makedirs('data/nhl', exist_ok=True)
        df.to_csv('data/nhl/shots_raw.csv', index=False)
        print(f"\n✓ Raw data saved to data/nhl/shots_raw.csv")
        
        # Step 2: Prepare features
        df = self.prepare_features(df)
        
        # Step 3: Train model
        model, auc = self.train_xgboost_model(df)
        
        print(f"\n🎯 Model Training Complete!")
        print(f"  Final AUC: {auc:.3f}")
        print(f"  Model location: models/production/xg_model_nhl.pkl")
        
        return model, auc


async def main():
    """Run the optimized training"""
    print("🏒 Starting NHL AI Training System...")
    print("=" * 60)
    
    trainer = OptimizedNHLTrainer()
    
    # You can adjust days_back based on how much data you want
    # More days = more data but longer training time
    days_back = 60  # Fetch 60 days of NHL games
    
    print(f"\n📅 Configuration:")
    print(f"  Days to fetch: {days_back}")
    print(f"  Estimated games: {days_back * 5}")  # ~5 games per day average
    print(f"  Estimated shots: {days_back * 5 * 60}")  # ~60 shots per game
    
    await trainer.run_training_pipeline(days_back=days_back)
    
    print("\n✨ Training Pipeline Complete!")
    print("\n💡 Next steps:")
    print("  1. Check model performance in models/production/model_metadata.pkl")
    print("  2. Update api/main.py to use the new model")
    print("  3. Test predictions with: python test_api.py")
    print("  4. View training data: data/nhl/shots_raw.csv")


if __name__ == "__main__":
    print("=" * 60)
    print("NHL AI TRAINING SCRIPT")
    print("=" * 60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)