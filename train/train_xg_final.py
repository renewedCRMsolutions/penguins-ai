# File: train/train_xg_final.py
"""
Final fixed version of xG model training pipeline
All type errors resolved
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier for calibration"""

    def __init__(self, predictions: Optional[np.ndarray] = None):
        self.predictions = predictions

    def fit(self, X, y):
        return self

    def predict(self, X):
        if self.predictions is None:
            raise ValueError("No predictions set")
        return (self.predictions >= 0.5).astype(int)

    def predict_proba(self, X):
        if self.predictions is None:
            raise ValueError("No predictions set")
        return np.column_stack([1 - self.predictions, self.predictions])


class AdvancedXGModel:
    """Ensemble model for xG prediction with focus on goalie impact"""

    def __init__(self, model_dir: str = "models/production"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.scalers = {}
        self.feature_sets = {}
        self.calibrators = {}
        self.meta_model = None

        # Feature categories
        self.shot_features = [
            "shot_distance",
            "shot_angle",
            "x_coord",
            "y_coord",
            "danger_zone",
            "shot_type",
            "is_rush",
            "is_rebound",
            "is_one_timer",
            "traffic_score",
            "passes_before_shot",
            "time_since_zone_entry",
            "time_since_faceoff",
        ]

        self.goalie_features = [
            "goalie_savePctg",
            "goalie_quality_rating",
            "goalie_shots_faced_period",
            "goalie_shots_faced_last_5min",
            "goalie_save_pct_last_10_shots",
            "goalie_save_pct_period",
            "time_since_last_shot",
            "consecutive_saves",
            "goalie_high_danger_save_pct",
            "goalie_fatigue_score",
            "goalie_cold_start",
            "goalie_recent_form_trend",
            "shooter_vs_goalie_career_shooting_pct",
        ]

        self.game_state_features = [
            "period",
            "time_remaining",
            "score_differential",
            "is_home_team",
            "is_power_play",
            "momentum_score",
            "zone_time",
            "shot_attempts_sequence",
        ]

        self.shooter_features = [
            "shooter_goals_last_10",
            "shooter_shots_last_10",
            "shooter_shooting_pct_last_10",
            "shooter_hot_streak",
        ]

    def prepare_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare different feature sets for different models"""

        # Filter to only features that exist
        available_shot_features = [f for f in self.shot_features if f in df.columns]
        available_goalie_features = [f for f in self.goalie_features if f in df.columns]
        available_game_features = [f for f in self.game_state_features if f in df.columns]
        available_shooter_features = [f for f in self.shooter_features if f in df.columns]

        feature_sets = {
            "base": df[available_shot_features + available_game_features].copy(),
            "goalie": df[available_goalie_features].copy() if available_goalie_features else pd.DataFrame(),
            "shooter": df[available_shooter_features].copy() if available_shooter_features else pd.DataFrame(),
            "all": df[available_shot_features + available_goalie_features + available_game_features].copy(),
        }

        # Remove empty entries
        feature_sets = {k: v for k, v in feature_sets.items() if not v.empty}

        # Handle categorical features
        for name, features in feature_sets.items():
            # Convert categorical to numeric
            categorical_cols = features.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                if col in ["danger_zone", "shot_type"]:
                    features[f"{col}_encoded"] = pd.Categorical(features[col]).codes
                    features = features.drop(col, axis=1)

            feature_sets[name] = features

        return feature_sets

    def _get_proba_column(self, predictions: Union[np.ndarray, Any]) -> np.ndarray:
        """Safely extract probability column from predictions"""
        # Handle sparse matrix from sklearn
        if not isinstance(predictions, np.ndarray):
            if hasattr(predictions, "toarray") and callable(getattr(predictions, "toarray", None)):
                predictions = predictions.toarray()
            else:
                predictions = np.array(predictions)

        # If it's 2D, get the second column (positive class probability)
        if len(predictions.shape) == 2 and predictions.shape[1] == 2:
            return predictions[:, 1]
        elif len(predictions.shape) == 1:
            return predictions
        else:
            # Fallback - try to get column 1
            try:
                return predictions[:, 1]
            except Exception:
                return predictions.flatten()

    def train_base_models(
        self, X: pd.DataFrame, y: pd.Series, feature_sets: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict, Dict, Tuple]:
        """Train individual models on different feature sets"""

        models = {}
        predictions = {}

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 1. XGBoost on all features
        logger.info("Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=20,
            eval_metric="auc",
            early_stopping_rounds=50,  # Moved here for newer XGBoost versions
        )

        xgb_model.fit(
            feature_sets["all"].iloc[X_train.index],
            y_train,
            eval_set=[(feature_sets["all"].iloc[X_val.index], y_val)],
            verbose=False,
        )

        models["xgboost"] = xgb_model
        xgb_preds = xgb_model.predict_proba(feature_sets["all"])
        predictions["xgboost"] = self._get_proba_column(xgb_preds)

        val_preds = xgb_model.predict_proba(feature_sets["all"].iloc[X_val.index])
        val_auc = roc_auc_score(y_val, self._get_proba_column(val_preds))
        logger.info(f"XGBoost Validation AUC: {val_auc:.4f}")

        # 2. LightGBM focusing on goalie features
        logger.info("Training LightGBM model (goalie focus)...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=800,
            num_leaves=31,
            learning_rate=0.01,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            n_jobs=20,
            verbose=-1,
        )

        # Combine goalie features with basic shot features
        if "goalie" in feature_sets and not feature_sets["goalie"].empty:
            goalie_focused = pd.concat([feature_sets["base"], feature_sets["goalie"]], axis=1)
        else:
            goalie_focused = feature_sets["base"]

        lgb_model.fit(
            goalie_focused.iloc[X_train.index],
            y_train,
            eval_set=[(goalie_focused.iloc[X_val.index], y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        models["lightgbm"] = lgb_model
        lgb_preds = lgb_model.predict_proba(goalie_focused)
        predictions["lightgbm"] = self._get_proba_column(lgb_preds)

        val_preds = lgb_model.predict_proba(goalie_focused.iloc[X_val.index])
        val_auc = roc_auc_score(y_val, self._get_proba_column(val_preds))
        logger.info(f"LightGBM Validation AUC: {val_auc:.4f}")

        # 3. CatBoost for handling categorical features
        logger.info("Training CatBoost model...")

        # Prepare data with categorical features
        cat_features = ["danger_zone_encoded", "shot_type_encoded"]
        cat_feature_indices = [i for i, col in enumerate(feature_sets["all"].columns) if col in cat_features]

        cat_model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.01,
            random_seed=42,
            eval_metric="AUC",
            early_stopping_rounds=50,
            verbose=False,
            thread_count=20,
        )

        cat_model.fit(
            feature_sets["all"].iloc[X_train.index],
            y_train,
            eval_set=(feature_sets["all"].iloc[X_val.index], y_val),
            cat_features=cat_feature_indices if cat_feature_indices else None,
        )

        models["catboost"] = cat_model
        cat_preds = cat_model.predict_proba(feature_sets["all"])
        predictions["catboost"] = self._get_proba_column(cat_preds)

        val_preds = cat_model.predict_proba(feature_sets["all"].iloc[X_val.index])
        val_auc = roc_auc_score(y_val, self._get_proba_column(val_preds))
        logger.info(f"CatBoost Validation AUC: {val_auc:.4f}")

        return models, predictions, (X_train, X_val, y_train, y_val)

    def _safe_array_from_series(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Safely convert pandas Series or array to numpy array"""
        # If it's already a numpy array
        if isinstance(data, np.ndarray):
            return data.reshape(-1, 1)

        # If it's a pandas Series or has .values attribute
        if hasattr(data, "values"):
            values = data.values
            # Check if it's an ExtensionArray with special handling
            if not isinstance(values, np.ndarray) and hasattr(values, "to_numpy"):
                return values.to_numpy().reshape(-1, 1)
            else:
                return np.array(values).reshape(-1, 1)

        # Fallback - convert to numpy array
        return np.array(data).reshape(-1, 1)

    def train_meta_model(
        self,
        base_predictions: Dict[str, np.ndarray],
        original_features: pd.DataFrame,
        y: pd.Series,
        train_val_split: Tuple,
    ) -> Tuple[nn.Module, np.ndarray]:
        """Train neural network meta-learner"""

        logger.info("Training neural network meta-model...")

        X_train, X_val, y_train, y_val = train_val_split

        # Prepare meta features
        meta_features_list = []

        # Add base model predictions
        for model_name in ["xgboost", "lightgbm", "catboost"]:
            if model_name in base_predictions:
                pred_array = self._safe_array_from_series(base_predictions[model_name])
                meta_features_list.append(pred_array)

        # Add key original features
        key_features = [
            "shot_distance",
            "shot_angle",
            "goalie_save_pct_last_10_shots",
            "goalie_fatigue_score",
            "time_since_last_shot",
            "is_rush",
            "is_power_play",
        ]

        for feat in key_features:
            if feat in original_features.columns:
                feat_array = self._safe_array_from_series(original_features[feat])
                meta_features_list.append(feat_array)

        # Stack features
        meta_features = np.hstack(meta_features_list)

        # Scale features
        scaler = StandardScaler()
        meta_features_scaled = scaler.fit_transform(meta_features)
        self.scalers["meta"] = scaler

        # Neural network architecture
        input_dim = meta_features_scaled.shape[1]

        class MetaNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 1)
                self.dropout = nn.Dropout(0.3)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.sigmoid(self.fc4(x))
                return x

        # Train
        model = MetaNet(input_dim)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Prepare data
        X_train_t = torch.FloatTensor(meta_features_scaled[X_train.index]).to(device)
        y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
        X_val_t = torch.FloatTensor(meta_features_scaled[X_val.index]).to(device)

        # Training
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        best_val_auc = 0
        patience = 0
        max_patience = 20

        for epoch in range(200):
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_preds = val_outputs.cpu().numpy()
                val_auc = roc_auc_score(y_val, val_preds)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience = 0
                    # Save best model
                    torch.save(model.state_dict(), self.model_dir / "meta_model_best.pth")
                else:
                    patience += 1

                if patience >= max_patience:
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}")

        # Load best model
        model.load_state_dict(torch.load(self.model_dir / "meta_model_best.pth"))

        logger.info(f"Meta-model Best Validation AUC: {best_val_auc:.4f}")

        return model, meta_features_scaled

    def calibrate_predictions(self, model_predictions: np.ndarray, y_true: pd.Series) -> CalibratedClassifierCV:
        """Calibrate model predictions to match actual goal rates"""

        logger.info("Calibrating predictions...")

        # Create dummy classifier and fit it first
        dummy = DummyClassifier(model_predictions)
        dummy_X = np.arange(len(model_predictions)).reshape(-1, 1)
        dummy.fit(dummy_X, y_true)  # Fit the dummy classifier

        # Now calibrate
        calibrator = CalibratedClassifierCV(dummy, method="isotonic", cv="prefit")
        calibrator.fit(dummy_X, y_true)

        # Get calibrated predictions
        calibrated_probs = calibrator.predict_proba(dummy_X)
        calibrated_probs = self._get_proba_column(calibrated_probs)

        logger.info(f"Original mean probability: {model_predictions.mean():.4f}")
        logger.info(f"Calibrated mean probability: {calibrated_probs.mean():.4f}")
        logger.info(f"Actual goal rate: {y_true.mean():.4f}")

        return calibrator

    def train_full_pipeline(self, data_path: str):
        """Train the complete xG model pipeline"""

        logger.info("Starting advanced xG model training...")

        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} shots")

        # Prepare target
        y = df["is_goal"].astype(int)

        # Prepare feature sets
        feature_sets = self.prepare_features(df)
        self.feature_sets = feature_sets

        # Train base models
        models, predictions, train_val_split = self.train_base_models(df, y, feature_sets)
        self.models = models

        # Train meta model
        meta_model, meta_features = self.train_meta_model(predictions, df, y, train_val_split)
        self.meta_model = meta_model

        # Get final predictions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta_model.eval()
        with torch.no_grad():
            final_predictions = self.meta_model(torch.FloatTensor(meta_features).to(device))
            final_predictions = final_predictions.cpu().numpy().flatten()

        # Calibrate predictions
        calibrator = self.calibrate_predictions(final_predictions, y)
        self.calibrators["main"] = calibrator

        # Calculate final metrics
        X_train, X_val, y_train, y_val = train_val_split
        val_preds = final_predictions[X_val.index]

        final_auc = roc_auc_score(y_val, val_preds)
        logger.info(f"\nFINAL MODEL VALIDATION AUC: {final_auc:.4f}")

        # Save models
        self.save_models()

        # Generate evaluation report
        self.evaluate_model(df, y, final_predictions)

        return final_auc

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""

        # Prepare features
        feature_sets = self.prepare_features(df)

        # Get base model predictions
        predictions = {}
        for model_name, model in self.models.items():
            if model_name == "lightgbm":
                # LightGBM uses combined features
                if "goalie" in feature_sets and not feature_sets["goalie"].empty:
                    features = pd.concat([feature_sets["base"], feature_sets["goalie"]], axis=1)
                else:
                    features = feature_sets["base"]
            else:
                features = feature_sets["all"]

            preds = model.predict_proba(features)
            predictions[model_name] = self._get_proba_column(preds)

        # Prepare meta features
        meta_features_list = []
        for model_name in ["xgboost", "lightgbm", "catboost"]:
            if model_name in predictions:
                pred_array = self._safe_array_from_series(predictions[model_name])
                meta_features_list.append(pred_array)

        # Add key features
        key_features = [
            "shot_distance",
            "shot_angle",
            "goalie_save_pct_last_10_shots",
            "goalie_fatigue_score",
            "time_since_last_shot",
            "is_rush",
            "is_power_play",
        ]

        for feat in key_features:
            if feat in df.columns:
                feat_array = self._safe_array_from_series(df[feat])
                meta_features_list.append(feat_array)

        meta_features = np.hstack(meta_features_list)
        meta_features_scaled = self.scalers["meta"].transform(meta_features)

        # Get meta model predictions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.meta_model is not None:
            self.meta_model.eval()
            with torch.no_grad():
                final_predictions = self.meta_model(torch.FloatTensor(meta_features_scaled).to(device))
                final_predictions = final_predictions.cpu().numpy().flatten()
        else:
            # Fallback to average of base models
            final_predictions = np.mean(list(predictions.values()), axis=0)

        # Calibrate if available
        if "main" in self.calibrators:
            dummy = DummyClassifier(final_predictions)
            self.calibrators["main"].estimator = dummy
            dummy_indices = np.arange(len(final_predictions)).reshape(-1, 1)
            calibrated = self.calibrators["main"].predict_proba(dummy_indices)
            final_predictions = self._get_proba_column(calibrated)

        return final_predictions

    def evaluate_model(self, df: pd.DataFrame, y_true: pd.Series, predictions: np.ndarray):
        """Comprehensive model evaluation"""

        logger.info("\n=== MODEL EVALUATION ===")

        # Overall metrics
        auc = roc_auc_score(y_true, predictions)
        logger.info(f"Overall AUC: {auc:.4f}")
        logger.info(f"Mean predicted xG: {predictions.mean():.4f}")
        logger.info(f"Actual goal rate: {y_true.mean():.4f}")

        # Feature importance from XGBoost
        if "xgboost" in self.models:
            feature_importance = pd.DataFrame(
                {"feature": self.feature_sets["all"].columns, "importance": self.models["xgboost"].feature_importances_}
            ).sort_values("importance", ascending=False)

            logger.info("\nTop 15 Most Important Features:")
            for idx, row in feature_importance.head(15).iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.4f}")

        # Create visualizations
        self.create_evaluation_plots(y_true, predictions, df)

    def create_evaluation_plots(self, y_true: pd.Series, predictions: np.ndarray, df: pd.DataFrame):
        """Create evaluation visualizations"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, predictions)
        auc = roc_auc_score(y_true, predictions)

        axes[0, 0].plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
        axes[0, 0].plot([0, 1], [0, 1], "k--", label="Random")
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].legend()

        # 2. Calibration Plot
        from sklearn.calibration import calibration_curve

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, predictions, n_bins=10, strategy="uniform"
        )

        axes[0, 1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        axes[0, 1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        axes[0, 1].set_xlabel("Mean Predicted Probability")
        axes[0, 1].set_ylabel("Fraction of Goals")
        axes[0, 1].set_title("Calibration Plot")
        axes[0, 1].legend()

        # 3. Prediction Distribution
        axes[1, 0].hist(predictions[y_true == 0], bins=50, alpha=0.5, label="No Goal", density=True)
        axes[1, 0].hist(predictions[y_true == 1], bins=50, alpha=0.5, label="Goal", density=True)
        axes[1, 0].set_xlabel("Predicted Probability")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title("Prediction Distribution by Outcome")
        axes[1, 0].legend()

        # 4. Feature Importance (Top 10)
        if "xgboost" in self.models and hasattr(self.models["xgboost"], "feature_importances_"):
            feature_importance = (
                pd.DataFrame(
                    {
                        "feature": self.feature_sets["all"].columns,
                        "importance": self.models["xgboost"].feature_importances_,
                    }
                )
                .sort_values("importance", ascending=True)
                .tail(10)
            )

            axes[1, 1].barh(feature_importance["feature"], feature_importance["importance"])
            axes[1, 1].set_xlabel("Importance")
            axes[1, 1].set_title("Top 10 Feature Importances")

        plt.tight_layout()
        plt.savefig(self.model_dir / "model_evaluation.png", dpi=300)
        plt.close()

        logger.info(f"Saved evaluation plots to {self.model_dir / 'model_evaluation.png'}")

    def save_models(self):
        """Save all models and configurations"""

        # Save base models
        for name, model in self.models.items():
            if name == "xgboost":
                model.save_model(self.model_dir / f"{name}_model.json")
            else:
                joblib.dump(model, self.model_dir / f"{name}_model.pkl")

        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, self.model_dir / f"{name}_scaler.pkl")

        # Save calibrators
        for name, calibrator in self.calibrators.items():
            joblib.dump(calibrator, self.model_dir / f"{name}_calibrator.pkl")

        # Save configuration
        config = {
            "shot_features": self.shot_features,
            "goalie_features": self.goalie_features,
            "game_state_features": self.game_state_features,
            "shooter_features": self.shooter_features,
            "model_names": list(self.models.keys()),
            "training_date": datetime.now().isoformat(),
        }

        with open(self.model_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"All models saved to {self.model_dir}")


# Training script
if __name__ == "__main__":
    # Initialize model
    model = AdvancedXGModel()

    # Train on enhanced data
    data_path = "data/nhl/enhanced/october_2024_with_goalie_features.csv"

    final_auc = model.train_full_pipeline(data_path)

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Final Model AUC: {final_auc:.4f}")
    logger.info("Target to beat (MoneyPuck): 0.7800")
    logger.info(f"Did we beat it? {'YES! 🎉' if final_auc > 0.78 else 'Not yet, but close!'}")
    logger.info("=" * 50)
