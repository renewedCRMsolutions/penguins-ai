# File: train/train_xg_shots_2024.py
"""
Train XGBoost model using your shots_2024.csv MoneyPuck data
This file has everything we need for xG modeling!
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import os
import logging
from typing import Dict
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShotDataXGTrainer:
    """Train XGBoost on MoneyPuck shot data"""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.shots_data = None

    def load_shot_data(self, filepath: str = "data/shots_2024.csv") -> pd.DataFrame:
        """Load the shot data"""
        logger.info(f"Loading shot data from {filepath}...")

        self.shots_data = pd.read_csv(filepath)

        logger.info(f"✅ Loaded {len(self.shots_data):,} shots")
        logger.info(f"  Goals: {self.shots_data['goal'].sum():,} ({self.shots_data['goal'].mean() * 100:.1f}%)")
        logger.info(f"  Columns: {len(self.shots_data.columns)}")

        # Show some key columns
        key_cols = ["xGoal", "shotDistance", "shotAngle", "shotType", "xCord", "yCord"]
        logger.info("\nKey columns found:")
        for col in key_cols:
            if col in self.shots_data.columns:
                logger.info(f"  ✓ {col}")

        return self.shots_data

    def explore_features(self):
        """Explore the shot data features"""
        if self.shots_data is None:
            logger.error("No shot data loaded!")
            return

        logger.info("\n" + "=" * 60)
        logger.info("SHOT DATA EXPLORATION")
        logger.info("=" * 60)

        # Check MoneyPuck's xGoal
        if "xGoal" in self.shots_data.columns:
            logger.info("\n🎯 MoneyPuck xGoal stats:")
            logger.info(f"  Mean xGoal: {self.shots_data['xGoal'].mean():.3f}")
            logger.info(f"  Actual goal rate: {self.shots_data['goal'].mean():.3f}")

        # Shot types
        logger.info("\n📊 Shot type distribution:")
        shot_types = self.shots_data["shotType"].value_counts()
        for stype, count in shot_types.head().items():
            pct = count / len(self.shots_data) * 100
            goal_rate = self.shots_data[self.shots_data["shotType"] == stype]["goal"].mean() * 100
            logger.info(f"  {stype}: {count:,} ({pct:.1f}%) - Goal rate: {goal_rate:.1f}%")

        # Distance distribution
        logger.info("\n📏 Shot distance stats:")
        logger.info(f"  Mean: {self.shots_data['shotDistance'].mean():.1f} ft")
        logger.info(f"  Median: {self.shots_data['shotDistance'].median():.1f} ft")
        logger.info(f"  Max: {self.shots_data['shotDistance'].max():.1f} ft")

        # Check for shot speed
        speed_cols = [col for col in self.shots_data.columns if "speed" in col.lower()]
        if speed_cols:
            logger.info(f"\n⚡ Speed-related columns found: {speed_cols}")
        else:
            logger.info("\n❌ No shot speed columns found (as expected)")

    def prepare_features(self) -> tuple:
        """Prepare features for training"""
        logger.info("\nPreparing features...")

        if self.shots_data is None:
            raise ValueError("No shot data loaded!")

        df = self.shots_data.copy()

        # Core features that should always be used
        feature_cols = [
            # Location features
            "arenaAdjustedShotDistance",
            "arenaAdjustedXCordABS",
            "arenaAdjustedYCordAbs",
            "shotAngleAdjusted",
            # Game state
            "period",
            "timeSinceFaceoff",
            "timeLeft",
            "homeTeamGoals",
            "awayTeamGoals",
            # Situation
            "homeSkatersOnIce",
            "awaySkatersOnIce",
            # Shot context
            "shotRebound",
            "shotRush",
            "shotOnEmptyNet",
            # Pre-shot movement
            "distanceFromLastEvent",
            "timeSinceLastEvent",
            "speedFromLastEvent",
        ]

        # Add MoneyPuck's features if we want to learn from them
        optional_features = [
            "xGoal",  # Their prediction
            "shotGoalProbability",  # Another model they have
            "xRebound",
            "xPlayContinuedInZone",
        ]

        # Check which features exist
        available_features = []
        for col in feature_cols:
            if col in df.columns:
                available_features.append(col)
            else:
                logger.warning(f"  Missing feature: {col}")

        # Encode shot type
        if "shotType" in df.columns:
            shot_type_dummies = pd.get_dummies(df["shotType"], prefix="shot")
            for col in shot_type_dummies.columns:
                df[col] = shot_type_dummies[col]
                available_features.append(col)

        # Encode last event
        if "lastEventCategory" in df.columns:
            # Only keep most common events
            top_events = df["lastEventCategory"].value_counts().head(10).index
            for event in top_events:
                col_name = f"lastEvent_{event}"
                df[col_name] = (df["lastEventCategory"] == event).astype(int)
                available_features.append(col_name)

        # Create some engineered features
        df["score_differential"] = df["homeTeamGoals"] - df["awayTeamGoals"]
        df["is_tied"] = (df["score_differential"] == 0).astype(int)
        df["is_powerplay"] = (df["homeSkatersOnIce"] != df["awaySkatersOnIce"]).astype(int)

        available_features.extend(["score_differential", "is_tied", "is_powerplay"])

        # Option to include MoneyPuck's xGoal for comparison
        include_xgoal = False  # Set to True to learn from their model
        if include_xgoal:
            for feat in optional_features:
                if feat in df.columns:
                    available_features.append(feat)

        self.feature_columns = available_features
        logger.info(f"Total features: {len(self.feature_columns)}")

        # Prepare X and y
        X = df[self.feature_columns].fillna(0)
        y = df["goal"]

        return X, y, df

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train XGBoost model"""
        logger.info("\nTraining XGBoost model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # XGBoost parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "max_depth": 8,
            "learning_rate": 0.03,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 5,
            "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
            "random_state": 42,
            "n_jobs": 20,  # Your optimal thread count
            "early_stopping_rounds": 50,  # Move this here
        }

        if self.use_gpu:
            params["tree_method"] = "hist"
            params["device"] = "cuda"
            logger.info("Using GPU acceleration")

        # Train with early stopping
        self.model = xgb.XGBClassifier(**params)

        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        self.model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=True)

        # Predictions
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)

        # Evaluate
        results = {
            "auc": roc_auc_score(y_test, y_pred_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "log_loss": log_loss(y_test, y_pred_proba),
            "n_samples": len(X),
            "n_features": len(self.feature_columns),
            "n_goals": y.sum(),
            "goal_rate": y.mean(),
            "best_iteration": self.model.best_iteration,
        }

        # Feature importance
        self._analyze_features()

        # Compare with MoneyPuck if available
        if self.shots_data is not None and "xGoal" in self.shots_data.columns:
            self._compare_with_moneypuck(X, y)

        self._print_results(results)

        return results

    def _analyze_features(self):
        """Analyze feature importance"""
        if self.model is None:
            return

        feature_importance = pd.DataFrame(
            {"feature": self.feature_columns, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(top_features["feature"], top_features["importance"])
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Most Important Features")
        plt.tight_layout()

        # Save plot
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/shot_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("\nTop 15 Features:")
        for idx, row in feature_importance.head(15).iterrows():
            logger.info(f"  {row['feature']:35s}: {row['importance']:.4f}")

    def _compare_with_moneypuck(self, X: pd.DataFrame, y: pd.Series):
        """Compare our model with MoneyPuck's xGoal"""
        if self.shots_data is None or "xGoal" not in self.shots_data.columns:
            logger.warning("Cannot compare - no xGoal data available")
            return

        if self.model is None:
            logger.warning("Cannot compare - model not trained yet")
            return

        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON WITH MONEYPUCK xGoal")
        logger.info("=" * 60)

        # Get our predictions for all data
        X_scaled = self.scaler.transform(X)
        our_predictions = self.model.predict_proba(X_scaled)[:, 1]

        # Get MoneyPuck's predictions
        mp_xgoal = self.shots_data["xGoal"].values[: len(our_predictions)]

        # Calculate metrics
        mp_auc = roc_auc_score(y, mp_xgoal)
        our_auc = roc_auc_score(y, our_predictions)

        correlation = np.corrcoef(our_predictions, mp_xgoal)[0, 1]

        logger.info(f"MoneyPuck AUC: {mp_auc:.4f}")
        logger.info(f"Our Model AUC: {our_auc:.4f}")
        logger.info(f"Correlation between models: {correlation:.3f}")
        logger.info(f"Our mean xG: {our_predictions.mean():.3f}")
        logger.info(f"MoneyPuck mean xG: {mp_xgoal.mean():.3f}")

    def _print_results(self, results: Dict):
        """Print training results"""
        logger.info("\n" + "=" * 60)
        logger.info("MODEL PERFORMANCE")
        logger.info("=" * 60)
        logger.info(f"Test AUC: {results['auc']:.4f}")
        logger.info(f"Log Loss: {results['log_loss']:.4f}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Samples: {results['n_samples']:,}")
        logger.info(f"Features: {results['n_features']}")
        logger.info(f"Goal rate: {results['goal_rate']:.1%}")

        if results["auc"] >= 0.75:
            logger.info("\n🎉 EXCELLENT! Model meets professional standard (0.75+ AUC)")
        elif results["auc"] >= 0.70:
            logger.info("\n✅ Good performance! Close to professional standard")
        else:
            logger.info("\n⚠️  More optimization needed")

    def save_model(self, results: Dict):
        """Save trained model and artifacts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs("models/moneypuck", exist_ok=True)

        # Save model
        model_path = f"models/moneypuck/xg_model_shots_{timestamp}.pkl"
        joblib.dump(self.model, model_path)

        # Save scaler
        joblib.dump(self.scaler, f"models/moneypuck/scaler_shots_{timestamp}.pkl")

        # Save feature list
        with open(f"models/moneypuck/features_shots_{timestamp}.txt", "w") as f:
            for feat in self.feature_columns:
                f.write(f"{feat}\n")

        # Save metadata
        import json

        metadata = {
            "timestamp": timestamp,
            "results": results,
            "feature_columns": self.feature_columns,
            "model_path": model_path,
            "data_source": "MoneyPuck shots_2024.csv",
        }

        with open(f"models/moneypuck/metadata_shots_{timestamp}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\nModel saved to {model_path}")


def main():
    """Run the training pipeline"""
    logger.info("🏒 NHL xG MODEL TRAINING - MoneyPuck Shot Data")
    logger.info("=" * 60)

    trainer = ShotDataXGTrainer(use_gpu=True)

    # Load shot data
    trainer.load_shot_data("data/shots_2024.csv")

    # Explore what we have
    trainer.explore_features()

    # Prepare features
    X, y, full_df = trainer.prepare_features()

    # Train model
    results = trainer.train_model(X, y)

    # Save model
    trainer.save_model(results)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    if results["auc"] >= 0.75:
        logger.info("🎉 Your model is competitive with professional xG models!")
        logger.info("\nNext steps:")
        logger.info("1. Deploy this model to your API")
        logger.info("2. Create real-time predictions for live games")
        logger.info("3. Test on 2025 season data")


if __name__ == "__main__":
    main()
