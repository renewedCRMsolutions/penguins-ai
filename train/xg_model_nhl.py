"""
Train XG Model using MoneyPuck Data
Target: 0.85+ AUC using 100k+ shots with player quality features
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoneyPuckXGTrainer:
    def __init__(self):
        self.data_dir = Path("data")
        self.model_dir = Path("models/production")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # XGBoost params optimized for your GPU
        self.xgb_params = {
            "n_estimators": 1000,
            "max_depth": 8,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "device": "cuda",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": 20,
        }

    def load_data(self):
        """Load and merge MoneyPuck datasets"""
        logger.info("Loading MoneyPuck data...")

        # Load shots data
        shots_df = pd.read_csv(self.data_dir / "shots_2024.csv")
        logger.info(f"Loaded {len(shots_df)} shots")

        # Load player quality data
        skaters_df = pd.read_csv(self.data_dir / "skaters.csv")
        goalies_df = pd.read_csv(self.data_dir / "goalies.csv")
        teams_df = pd.read_csv(self.data_dir / "teams.csv")

        return shots_df, skaters_df, goalies_df, teams_df

    def create_player_quality_features(self, skaters_df, goalies_df):
        """Create shooter and goalie quality lookup tables"""
        logger.info("Building player quality features...")

        # Shooter quality metrics
        shooter_quality = (
            skaters_df.groupby("playerId")
            .agg(
                {
                    "I_F_goals": "sum",
                    "I_F_xGoals": "sum",
                    "I_F_shotsOnGoal": "sum",
                    "I_F_highDangerShots": "sum",
                    "I_F_highDangerGoals": "sum",
                    "I_F_mediumDangerShots": "sum",
                    "I_F_mediumDangerGoals": "sum",
                }
            )
            .reset_index()
        )

        # Calculate shooting talent
        shooter_quality["shooting_talent"] = shooter_quality["I_F_goals"] / shooter_quality["I_F_xGoals"].clip(
            lower=0.1
        )
        shooter_quality["high_danger_conversion"] = shooter_quality["I_F_highDangerGoals"] / shooter_quality[
            "I_F_highDangerShots"
        ].clip(lower=1)
        shooter_quality["shot_quality_ratio"] = shooter_quality["I_F_highDangerShots"] / shooter_quality[
            "I_F_shotsOnGoal"
        ].clip(lower=1)

        # Goalie quality metrics
        goalie_quality = (
            goalies_df.groupby("playerId")
            .agg(
                {
                    "goals": "sum",
                    "xGoals": "sum",
                    "highDangerGoals": "sum",
                    "highDangerxGoals": "sum",
                    "mediumDangerGoals": "sum",
                    "mediumDangerxGoals": "sum",
                    "lowDangerGoals": "sum",
                    "lowDangerxGoals": "sum",
                }
            )
            .reset_index()
        )

        # Calculate save talent
        goalie_quality["save_talent"] = 1 - (goalie_quality["goals"] / goalie_quality["xGoals"].clip(lower=0.1))
        goalie_quality["high_danger_save_talent"] = 1 - (
            goalie_quality["highDangerGoals"] / goalie_quality["highDangerxGoals"].clip(lower=0.1)
        )

        return shooter_quality, goalie_quality

    def engineer_features(self, shots_df, shooter_quality, goalie_quality):
        """Create all features for model training"""
        logger.info("Engineering features...")

        # Merge player quality data
        df = shots_df.merge(
            shooter_quality, left_on="shooterPlayerId", right_on="playerId", how="left", suffixes=("", "_shooter")
        )

        df = df.merge(
            goalie_quality, left_on="goalieIdForShot", right_on="playerId", how="left", suffixes=("", "_goalie")
        )

        event_dummies = pd.get_dummies(df["lastEventCategory"], prefix="lastEvent")
        df = pd.concat([df, event_dummies], axis=1)

        # Fill missing player quality with averages
        shooter_cols = ["shooting_talent", "high_danger_conversion", "shot_quality_ratio"]
        goalie_cols = ["save_talent", "high_danger_save_talent"]

        for col in shooter_cols:
            df[col] = df[col].fillna(df[col].mean())
        for col in goalie_cols:
            df[col] = df[col].fillna(df[col].mean())

        # Create feature set
        features = [
            # Spatial features (most important)
            "arenaAdjustedShotDistance",
            "shotAngleAdjusted",
            "arenaAdjustedXCordABS",
            "arenaAdjustedYCordAbs",
            # Player quality (huge impact)
            "shooting_talent",
            "high_danger_conversion",
            "shot_quality_ratio",
            "save_talent",
            "high_danger_save_talent",
            # Shot context
            "shotRebound",
            "shotRush",
            "shotWasOnGoal",
            "speedFromLastEvent",
            "timeSinceLastEvent",
            # Game state
            "homeSkatersOnIce",
            "awaySkatersOnIce",
            "period",
            "timeLeft",
            "awayTeamGoals",
            "homeTeamGoals",
        ]

        # Add shot type dummies
        shot_type_dummies = pd.get_dummies(df["shotType"], prefix="shotType")
        df = pd.concat([df, shot_type_dummies], axis=1)
        features.extend(shot_type_dummies.columns.tolist())

        features.extend(event_dummies.columns.tolist())

        # Calculate additional features
        df["score_differential"] = df["homeTeamGoals"] - df["awayTeamGoals"]
        df["is_home_shooting"] = df["isHomeTeam"].astype(int)
        df["strength_differential"] = df["homeSkatersOnIce"] - df["awaySkatersOnIce"]

        features.extend(["score_differential", "is_home_shooting", "strength_differential"])

        return df, features

    def train_model(self, df, features):
        """Train XGBoost model"""
        logger.info(f"Training on {len(df)} shots with {len(features)} features...")

        # Prepare data
        X = df[features]
        y = df["goal"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Calculate scale for class imbalance
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        # Train model
        model = xgb.XGBClassifier(
            **self.xgb_params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=50,
            verbosity=1,
        )

        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"Test AUC: {auc:.4f}")

        # Feature importance
        importance_df = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )

        logger.info("\nTop 10 Features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

        return model, auc, importance_df

    def save_model(self, model, features):
        """Save model and metadata"""
        model_path = self.model_dir / "xg_model_nhl.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save feature list
        with open(self.model_dir / "features.txt", "w") as f:
            for feature in features:
                f.write(f"{feature}\n")

        # Save metadata
        metadata = {
            "model_type": "XGBoost",
            "n_features": len(features),
            "training_samples": model.n_features_in_,
            "params": self.xgb_params,
        }
        joblib.dump(metadata, self.model_dir / "model_metadata.pkl")

    def run(self):
        """Main training pipeline"""
        # Load data
        shots_df, skaters_df, goalies_df, teams_df = self.load_data()

        # Create player quality features
        shooter_quality, goalie_quality = self.create_player_quality_features(skaters_df, goalies_df)

        # Engineer features
        df, features = self.engineer_features(shots_df, shooter_quality, goalie_quality)

        # Train model
        model, auc, importance_df = self.train_model(df, features)

        # Save everything
        self.save_model(model, features)
        importance_df.to_csv(self.model_dir / "feature_importance.csv", index=False)

        logger.info(f"\nTraining complete! AUC: {auc:.4f}")
        logger.info(f"Model saved to {self.model_dir}")


if __name__ == "__main__":
    trainer = MoneyPuckXGTrainer()
    trainer.run()
