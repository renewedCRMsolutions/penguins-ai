# File: train/train_xg_model.py (UPDATED VERSION)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ExpectedGoalsTrainer:
    def __init__(self):
        self.model = None
        self.feature_columns = [
            'shotDistance', 'shotAngle', 'timeSinceLast',
            'isRebound', 'isRush', 'period'
        ]
        
    def load_data(self):
        """Load and combine all shot data"""
        print("Loading shot data...")
        
        all_data = []
        data_files = [f for f in os.listdir('data') if f.startswith('shots_') and f.endswith('.csv')]
        
        if not data_files:
            print("No data files found! Run setup/download_nhl_data.py first")
            return None
            
        for file in data_files:
            df = pd.read_csv(f'data/{file}')
            all_data.append(df)
            print(f"  - Loaded {file}: {len(df)} shots")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ Total shots loaded: {len(combined_df)}")
        print(f"✓ Goals: {combined_df['goal'].sum()} ({combined_df['goal'].mean()*100:.1f}%)")
        return combined_df
        
    def preprocess_data(self, df):
        """Prepare features for training"""
        print("\nPreprocessing data...")
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=['shotType', 'lastEventType'], prefix=['shot', 'event'])
        
        # Get all feature columns
        feature_cols = self.feature_columns.copy()
        
        # Add encoded columns
        for col in df_encoded.columns:
            if col.startswith('shot_') or col.startswith('event_'):
                feature_cols.append(col)
                
        print(f"✓ Total features: {len(feature_cols)}")
        return df_encoded, feature_cols
        
    def train_model(self):
        """Train the XGBoost model"""
        # Load data
        df = self.load_data()
        if df is None:
            return
            
        # Preprocess
        df_processed, features = self.preprocess_data(df)
        
        # Prepare training data
        X = df_processed[features]
        y = df_processed['goal']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} shots")
        print(f"Test set: {len(X_test)} shots")
        print("\nTraining XGBoost model...")
        
        # Train model with updated API
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            objective='binary:logistic',
            eval_metric='logloss',
            early_stopping_rounds=20,  # Move this parameter here
            random_state=42
        )
        
        # Fit the model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=True
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\n✓ Model Performance:")
        print(f"  - Accuracy: {accuracy:.3f}")
        print(f"  - AUC-ROC: {auc:.3f}")
        
        # Save model and features
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/xg_model.pkl')
        joblib.dump(features, 'models/xg_features.pkl')
        print(f"\n✓ Model saved to models/xg_model.pkl")
        print(f"✓ Features saved to models/xg_features.pkl")
        
        # Show feature importance
        self.show_feature_importance()
        
        # Save model info
        model_info = {
            'accuracy': accuracy,
            'auc': auc,
            'features': features,
            'n_estimators': self.model.n_estimators,
            'training_samples': len(X_train)
        }
        joblib.dump(model_info, 'models/model_info.pkl')
        
    def show_feature_importance(self):
        """Display feature importance"""
        importance = self.model.feature_importances_
        
        # Get feature names
        feature_names = self.model.get_booster().feature_names
        if not feature_names:
            # Fallback if feature names not available
            feature_names = [f'f{i}' for i in range(len(importance))]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print("-" * 40)
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:25s} {row['importance']:.4f}")

if __name__ == "__main__":
    print("Pittsburgh Penguins - Expected Goals Model Training")
    print("=" * 50)
    
    trainer = ExpectedGoalsTrainer()
    trainer.train_model()
    
    print("\n✓ Training complete!")
    print("\nNext steps:")
    print("1. Run: python -m uvicorn api.main:app --reload")
    print("2. Visit: http://localhost:8000/docs")