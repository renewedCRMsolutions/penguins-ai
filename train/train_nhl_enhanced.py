# File: train/train_nhl_enhanced.py
"""
Train XGBoost model with enhanced NHL features including shot speed and tracking data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import joblib
from datetime import datetime
import json
import logging
import asyncio
import os
from typing import Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

# Import our enhanced data fetcher
from fetch_enhanced_nhl_data import NHLEnhancedDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHLEnhancedTrainer:
    """Train XGBoost with enhanced NHL features"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.model = None
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_or_fetch_data(self, force_fetch: bool = False) -> pd.DataFrame:
        """Load existing data or fetch new data"""
        data_path = "data/nhl/shots_enhanced_latest.csv"

        if not force_fetch and os.path.exists(data_path):
            logger.info(f"Loading existing data from {data_path}")
            df = pd.read_csv(data_path)
        else:
            logger.info("Fetching new data from NHL API...")
            # Run async fetcher
            asyncio.run(self._fetch_new_data())

            # Load the newly fetched data
            # Find most recent file
            import glob

            files = glob.glob("data/nhl/shots_enhanced_*.csv")
            if files:
                latest_file = max(files)
                df = pd.read_csv(latest_file)
                # Save as latest for easy access
                df.to_csv(data_path, index=False)
            else:
                raise ValueError("No data files found after fetching")

        return df

    async def _fetch_new_data(self):
        """Fetch new data using enhanced fetcher"""
        from datetime import datetime, timedelta

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

        async with NHLEnhancedDataFetcher() as fetcher:
            await fetcher.fetch_all_data(start_date, end_date, max_games=200)
            fetcher.save_data()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features"""
        logger.info("Engineering enhanced features...")

        # Basic features that should always exist
        if "distance" not in df.columns:
            df["distance"] = np.sqrt((89 - df["x_coord"].abs()) ** 2 + df["y_coord"] ** 2)

        if "angle" not in df.columns:
            df["angle"] = np.abs(np.arctan2(df["y_coord"], 89 - df["x_coord"].abs()) * 180 / np.pi)

        # Shot speed features (if available)
        if "shot_speed" in df.columns and df["shot_speed"].notna().any():
            logger.info("✅ Using shot speed data!")
            # Fill missing speeds with median by shot type
            df["shot_speed_filled"] = df.groupby("shot_type_detail")["shot_speed"].transform(
                lambda x: x.fillna(x.median())
            )
            # Create speed categories
            df["speed_category"] = pd.cut(
                df["shot_speed_filled"], bins=[0, 70, 85, 95, 200], labels=["slow", "medium", "fast", "elite"]
            )
        else:
            logger.warning("❌ No shot speed data available - using distance as proxy")
            # Use distance as a proxy for shot difficulty
            df["shot_speed_filled"] = 0
            df["speed_category"] = "unknown"

        # Time-based features
        df["seconds_remaining"] = (20 * 60) - df["time_in_period"]
        df["period_progress"] = df["time_in_period"] / (20 * 60)

        # Situation features
        df["is_powerplay"] = (df["strength"] == "PP").astype(int)
        df["is_shorthanded"] = (df["strength"] == "SH").astype(int)
        df["is_evenstrength"] = (df["strength"] == "EV").astype(int)

        # Score state features
        df["score_close"] = (df["score_diff"].abs() <= 1).astype(int)
        df["trailing"] = (df["score_diff"] < 0).astype(int)
        df["leading"] = (df["score_diff"] > 0).astype(int)

        # Shot type encoding
        shot_type_quality = {
            "wrist": 0.7,
            "slap": 0.8,
            "snap": 0.75,
            "backhand": 0.6,
            "tip-in": 0.9,
            "deflected": 0.85,
            "wrap-around": 0.5,
        }
        df["shot_quality_score"] = df["shot_type_detail"].map(shot_type_quality).fillna(0.65)

        # Rush and rebound bonuses
        df["is_rush_shot"] = df.get("is_rush", 0)
        df["is_rebound_shot"] = df.get("is_rebound", 0)
        df["is_turnover_shot"] = df.get("is_off_turnover", 0)

        # Danger zone (high-danger area in front of net)
        df["is_high_danger"] = ((df["distance"] < 15) & (df["angle"].abs() < 40)).astype(int)

        # Shooting player quality (if available)
        if "shooter_career_shooting_pct" in df.columns:
            df["shooter_quality"] = df["shooter_career_shooting_pct"].fillna(df["shooter_career_shooting_pct"].median())
        else:
            df["shooter_quality"] = 0.09  # League average

        # Create composite danger score
        df["danger_score"] = (
            df["shot_quality_score"] * 0.3
            + df["is_high_danger"] * 0.3
            + (1 - df["distance"] / 100) * 0.2
            + (1 - df["angle"] / 90) * 0.1
            + df["is_rebound_shot"] * 0.05
            + df["is_rush_shot"] * 0.05
        )

        # Add shot speed to danger score if available
        if df["shot_speed_filled"].max() > 0:
            df["danger_score"] += (df["shot_speed_filled"] / 100) * 0.2

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training"""

        # Define feature columns
        numeric_features = [
            "distance",
            "angle",
            "time_in_period",
            "seconds_remaining",
            "period_progress",
            "shot_quality_score",
            "danger_score",
            "home_score",
            "away_score",
            "score_diff",
            "shooter_quality",
            "shot_speed_filled",
        ]

        categorical_features = ["period", "strength", "shot_type_detail", "zone", "speed_category", "shooter_position"]

        binary_features = [
            "is_home_team",
            "is_powerplay",
            "is_shorthanded",
            "is_evenstrength",
            "score_close",
            "trailing",
            "leading",
            "is_high_danger",
            "is_rush_shot",
            "is_rebound_shot",
            "is_turnover_shot",
        ]

        # Check which features exist
        numeric_features = [f for f in numeric_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        binary_features = [f for f in binary_features if f in df.columns]

        # Encode categorical features
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df[col].fillna("unknown"))
            else:
                df[f"{col}_encoded"] = self.label_encoders[col].transform(df[col].fillna("unknown"))

        # Combine all features
        encoded_categorical = [f"{col}_encoded" for col in categorical_features]
        self.feature_columns = numeric_features + encoded_categorical + binary_features

        # Create feature matrix
        X = df[self.feature_columns].fillna(0)
        y = df["is_goal"]

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train XGBoost model with optimal parameters"""
        logger.info("Training enhanced XGBoost model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # XGBoost parameters optimized for GPU
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1,
            "min_child_weight": 5,
            "scale_pos_weight": len(y_train) / y_train.sum() - 1,  # Handle class imbalance
            "random_state": 42,
            "n_jobs": 20,  # Use 20 threads as per your benchmark
        }

        if self.use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
            logger.info("Using GPU acceleration")
        else:
            params["tree_method"] = "hist"

        # Train model
        self.model = xgb.XGBClassifier(**params)

        self.model.fit(
            X_train_scaled, y_train, early_stopping_rounds=50, eval_set=[(X_test_scaled, y_test)], verbose=True
        )

        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)

        results = {
            "auc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "n_samples": len(X),
            "n_features": len(self.feature_columns),
            "n_goals": y.sum(),
            "goal_rate": y.mean(),
            "best_iteration": self.model.best_iteration,
        }

        # Feature importance
        feature_importance = pd.DataFrame(
            {"feature": self.feature_columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        results["top_features"] = feature_importance.head(10).to_dict("records")

        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL PERFORMANCE")
        logger.info(f"{'='*60}")
        logger.info(f"AUC: {results['auc']:.4f}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Samples: {results['n_samples']:,}")
        logger.info(f"Features: {results['n_features']}")
        logger.info(f"Goal rate: {results['goal_rate']:.1%}")

        logger.info(f"\nTop 10 Features:")
        for feat in results["top_features"]:
            logger.info(f"  {feat['feature']}: {feat['importance']:.4f}")

        # Check if shot speed was important
        shot_speed_importance = feature_importance[feature_importance["feature"].str.contains("speed", case=False)]
        if not shot_speed_importance.empty:
            logger.info(f"\n🎯 Shot speed feature importance:")
            for _, row in shot_speed_importance.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return results

    def save_model(self, results: Dict):
        """Save model and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model
        model_path = f"models/production/xg_model_enhanced_{timestamp}.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save scaler
        scaler_path = f"models/production/scaler_enhanced_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_path)

        # Save label encoders
        encoders_path = f"models/production/label_encoders_{timestamp}.pkl"
        joblib.dump(self.label_encoders, encoders_path)

        # Save feature columns
        features_path = f"models/production/features_enhanced_{timestamp}.json"
        with open(features_path, "w") as f:
            json.dump(self.feature_columns, f, indent=2)

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "results": results,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "encoders_path": encoders_path,
            "features_path": features_path,
            "feature_columns": self.feature_columns,
        }

        metadata_path = f"models/production/metadata_enhanced_{timestamp}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create symlinks to latest
        self._create_latest_symlinks(timestamp)

        logger.info(f"All artifacts saved with timestamp {timestamp}")

    def _create_latest_symlinks(self, timestamp: str):
        """Create symlinks to latest model files"""
        import shutil

        files = [
            ("xg_model_enhanced", "pkl"),
            ("scaler_enhanced", "pkl"),
            ("label_encoders", "pkl"),
            ("features_enhanced", "json"),
            ("metadata_enhanced", "json"),
        ]

        for base_name, ext in files:
            source = f"models/production/{base_name}_{timestamp}.{ext}"
            target = f"models/production/{base_name}_latest.{ext}"

            if os.path.exists(target):
                os.remove(target)
            shutil.copy2(source, target)


def main():
    """Run the enhanced training pipeline"""
    logger.info("🏒 NHL ENHANCED MODEL TRAINING")
    logger.info("=" * 60)

    # Create trainer
    trainer = NHLEnhancedTrainer(use_gpu=True)

    # Load or fetch data
    df = trainer.load_or_fetch_data(force_fetch=False)  # Set to True to fetch new data

    logger.info(f"Loaded {len(df)} shots")

    # Check if we have shot speed data
    if "shot_speed" in df.columns and df["shot_speed"].notna().any():
        logger.info(f"✅ Shot speed data available for {df['shot_speed'].notna().sum()} shots")
        logger.info(f"  Average speed: {df['shot_speed'].mean():.1f}")
        logger.info(f"  Max speed: {df['shot_speed'].max():.1f}")
    else:
        logger.warning("❌ No shot speed data found - will train without it")

    # Engineer features
    df = trainer.engineer_features(df)

    # Prepare for training
    X, y = trainer.prepare_features(df)

    # Train model
    results = trainer.train_model(X, y)

    # Save everything
    trainer.save_model(results)

    # Performance summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Final AUC: {results['auc']:.4f}")

    if results["auc"] >= 0.75:
        logger.info("✅ Model meets performance target!")
    else:
        logger.info("❌ Model below target - need more features or data")
        logger.info("\nNext steps:")
        logger.info("1. Fetch more historical data (200+ games)")
        logger.info("2. Add player tracking data from other sources")
        logger.info("3. Implement ensemble with multiple models")


if __name__ == "__main__":
    main()
