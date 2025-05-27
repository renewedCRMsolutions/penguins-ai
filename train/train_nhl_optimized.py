# File: penguins_ai/train/train_nhl_optimized.py
# This is THE ONE to use - combines everything with your optimal settings

import sys
sys.path.append('.')
from config.optimal_settings import SYSTEM_OPTIMAL_CONFIG

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import time

class OptimizedNHLTrainer:
    """NHL model training optimized for your P16 system"""
    
    def __init__(self):
        self.config = SYSTEM_OPTIMAL_CONFIG
        self.xgb_params = self.config['xgboost_config']
        print(f"🖥️ System Configuration Loaded:")
        print(f"  - XGBoost threads: {self.xgb_params['n_jobs']}")
        print(f"  - Max memory: {self.config['max_memory_gb']}GB")
        print(f"  - Batch size: {self.config['batch_size']}")
        print(f"  - GPU enabled: {self.config['use_gpu']}")
        
    async def fetch_nhl_data(self, days_back=30):
        """Fetch real NHL shot data"""
        print(f"\n📊 Fetching {days_back} days of NHL data...")
        all_shots = []
        
        async with aiohttp.ClientSession() as session:
            end_date = datetime.now() - timedelta(days=1)
            
            for day in range(days_back):
                date = (end_date - timedelta(days=day)).strftime("%Y-%m-%d")
                
                # Get games for this date
                url = f"https://api-web.nhle.com/v1/score/{date}"
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            games = data.get('games', [])
                            
                            print(f"  Date {date}: {len(games)} games")
                            
                            # Process each completed game
                            for game in games:
                                if game.get('gameState') == 'OFF':
                                    shots = await self.extract_game_shots(session, game['id'])
                                    all_shots.extend(shots)
                                    
                except Exception as e:
                    print(f"  Error on {date}: {e}")
                    
                await asyncio.sleep(0.5)  # Rate limiting
                
        df = pd.DataFrame(all_shots)
        print(f"\n✓ Collected {len(df)} total shots")
        return df
    
    async def extract_game_shots(self, session, game_id):
        """Extract shot features from a game"""
        shots = []
        
        try:
            url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    plays = data.get('plays', [])
                    
                    for i, play in enumerate(plays):
                        if play.get('typeDescKey') in ['shot-on-goal', 'goal', 'missed-shot']:
                            shot = self.process_shot(play, plays[:i])
                            if shot:
                                shots.append(shot)
                                
        except Exception as e:
            print(f"    Error processing game {game_id}: {e}")
            
        return shots
    
    def process_shot(self, play, previous_plays):
        """Extract features from a shot"""
        details = play.get('details', {})
        
        # Basic features
        x = details.get('xCoord', 0)
        y = details.get('yCoord', 0)
        
        # Calculate distance and angle
        goal_x, goal_y = 89, 0  # NHL goal position
        distance = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
        angle = np.arctan2(abs(y - goal_y), abs(goal_x - x)) * 180 / np.pi
        
        # Advanced features
        last_event = previous_plays[-1] if previous_plays else {}
        time_since_last = 5.0  # Default, would calculate from timestamps
        
        return {
            'shotDistance': distance,
            'shotAngle': angle,
            'shotType': details.get('shotType', 'Wrist'),
            'period': play.get('period', 2),
            'isRebound': 1 if self.is_rebound(previous_plays) else 0,
            'isRush': 1 if self.is_rush(previous_plays) else 0,
            'timeSinceLast': time_since_last,
            'goal': 1 if play['typeDescKey'] == 'goal' else 0,
            # New features
            'shotVelocity': details.get('shotSpeed', 0),
            'traffic': self.calculate_traffic(play, previous_plays),
            'scoreState': play.get('situationCode', '1551')[:2],  # Even, PP, PK
        }
    
    def is_rebound(self, previous_plays, window=3):
        """Check if shot is a rebound"""
        if len(previous_plays) < 1:
            return False
        for play in previous_plays[-window:]:
            if 'shot' in play.get('typeDescKey', '').lower():
                return True
        return False
    
    def is_rush(self, previous_plays, window=5):
        """Check if shot is on a rush"""
        if len(previous_plays) < 2:
            return False
        # Look for zone entry in recent plays
        for play in previous_plays[-window:]:
            if play.get('details', {}).get('zoneCode') == 'N':  # Neutral zone
                return True
        return False
    
    def calculate_traffic(self, shot, previous_plays):
        """Estimate traffic in front of net"""
        # Simplified - would use player positions
        return len([p for p in previous_plays[-3:] if 'hit' in p.get('typeDescKey', '')])
    
    def train_xgboost_model(self, df):
        """Train model with optimal settings"""
        print(f"\n🤖 Training XGBoost with {self.xgb_params['n_jobs']} threads...")
        
        # Prepare features
        feature_cols = [
            'shotDistance', 'shotAngle', 'period', 'timeSinceLast',
            'isRebound', 'isRush', 'shotVelocity', 'traffic'
        ]
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=['shotType', 'scoreState'])
        
        # Get all feature columns
        all_features = feature_cols + [col for col in df_encoded.columns 
                                      if col.startswith(('shotType_', 'scoreState_'))]
        
        # Remove any missing features
        available_features = [f for f in all_features if f in df_encoded.columns]
        
        X = df_encoded[available_features].fillna(0)
        y = df_encoded['goal']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Features: {len(available_features)}")
        
        # Train with optimal parameters
        start_time = time.time()
        
        model = xgb.XGBClassifier(
            **self.xgb_params,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # Add this!
        )
        
        # Use evaluation set for early stopping
        model.set_params(early_stopping_rounds=50)
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
        
        # Save model
        os.makedirs('models/production', exist_ok=True)
        joblib.dump(model, 'models/production/xg_model_nhl.pkl')
        joblib.dump(available_features, 'models/production/xg_features_nhl.pkl')
        
        # Save metadata
        metadata = {
            'accuracy': accuracy,
            'auc': auc,
            'features': available_features,
            'training_samples': len(X_train),
            'training_time': train_time,
            'best_iteration': model.best_iteration,
            'system_config': self.config,
            'trained_date': datetime.now().isoformat()
        }
        joblib.dump(metadata, 'models/production/model_metadata.pkl')
        
        return model, auc
    
    async def run_training_pipeline(self, days_back=30):
        """Complete training pipeline"""
        print("🏒 NHL XG Model Training Pipeline - P16 Optimized")
        print("=" * 60)
        
        # Step 1: Fetch data
        df = await self.fetch_nhl_data(days_back)
        
        if len(df) < 1000:
            print("⚠️ Not enough data! Try increasing days_back.")
            return
        
        # Save raw data
        os.makedirs('data/nhl', exist_ok=True)
        df.to_csv('data/nhl/shots_raw.csv', index=False)
        print(f"✓ Raw data saved to data/nhl/shots_raw.csv")
        
        # Step 2: Train model
        model, auc = self.train_xgboost_model(df)
        
        # Step 3: Feature importance
        print("\n📊 Top Feature Importance:")
        importance = model.feature_importances_
        feature_names = model.get_booster().feature_names
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(10)
        
        for _, row in importance_df.iterrows():
            print(f"  {row['feature']:25s} {row['importance']:.3f}")
        
        print(f"\n🎯 Model ready for production!")
        print(f"  Location: models/production/xg_model_nhl.pkl")
        print(f"  Performance: {auc:.3f} AUC")

async def main():
    """Run the optimized training"""
    trainer = OptimizedNHLTrainer()
    
    # Train with last 30 days of data
    await trainer.run_training_pipeline(days_back=120)
    
    print("\n💡 Next steps:")
    print("1. Update api/main.py to use models/production/xg_model_nhl.pkl")
    print("2. Test predictions with new features")
    print("3. Deploy to production!")

if __name__ == "__main__":
    asyncio.run(main())