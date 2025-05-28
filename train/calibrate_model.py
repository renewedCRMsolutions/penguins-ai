# File: train/calibrate_model.py
"""
Calibrate the XGBoost model to match actual goal rates
This fixes the issue where predictions are too high
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCalibrator:
    """Calibrate xG predictions to match actual goal rates"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.feature_columns = None

    def load_trained_model(self, model_path: str | None = None) -> None:
        """Load the most recent trained model"""
        if model_path is None:
            # Find most recent model
            model_files = [f for f in os.listdir("models/moneypuck") if f.startswith("xg_model_shots_")]
            if not model_files:
                raise ValueError("No trained models found!")
            model_path = os.path.join("models/moneypuck", sorted(model_files)[-1])

        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)

        # Load scaler
        scaler_path = model_path.replace("xg_model_", "scaler_")
        self.scaler = joblib.load(scaler_path)

        # Load features
        features_path = model_path.replace("xg_model_", "features_").replace(".pkl", ".txt")
        with open(features_path, "r") as f:
            self.feature_columns = [line.strip() for line in f.readlines()]

    def load_data_and_predict(self, data_path: str = "data/shots_2024.csv"):
        """Load data and get predictions"""
        logger.info("Loading shot data...")
        df = pd.read_csv(data_path)

        # Prepare features (same as training)
        X = self._prepare_features(df)
        y = df["goal"]

        # Get predictions
        if self.scaler is None or self.model is None:
            raise ValueError("Model or scaler not loaded!")
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict_proba(X_scaled)[:, 1]

        return X, y, predictions, df

    def _prepare_features(self, df):
        """Recreate the same features as training"""
        # This should match your training feature preparation
        # For now, using the feature columns we loaded

        # Handle shot type encoding
        if "shotType" in df.columns:
            shot_type_dummies = pd.get_dummies(df["shotType"], prefix="shot")
            for col in shot_type_dummies.columns:
                if self.feature_columns and col in self.feature_columns:
                    df[col] = shot_type_dummies[col]

        # Handle last event encoding
        if "lastEventCategory" in df.columns:
            top_events = df["lastEventCategory"].value_counts().head(10).index
            for event in top_events:
                col_name = f"lastEvent_{event}"
                if self.feature_columns and col_name in self.feature_columns:
                    df[col_name] = (df["lastEventCategory"] == event).astype(int)

        # Add calculated features
        df["score_differential"] = df["homeTeamGoals"] - df["awayTeamGoals"]
        df["is_tied"] = (df["score_differential"] == 0).astype(int)
        df["is_powerplay"] = (df["homeSkatersOnIce"] != df["awaySkatersOnIce"]).astype(int)

        # Select only the features used in training
        if not self.feature_columns:
            raise ValueError("Feature columns not loaded!")
        available_features = [col for col in self.feature_columns if col in df.columns]
        X = df[available_features].fillna(0)

        return X

    def calibrate_predictions(self, X, y, raw_predictions):
        """Apply different calibration methods"""
        logger.info("\nCalibrating predictions...")

        # Split for calibration
        X_train, X_cal, y_train, y_cal, pred_train, pred_cal = train_test_split(
            X, y, raw_predictions, test_size=0.3, random_state=42, stratify=y
        )

        results = {}

        # 1. No calibration (baseline)
        results["raw"] = {
            "predictions": pred_cal,
            "mean": pred_cal.mean(),
            "auc": roc_auc_score(y_cal, pred_cal),
            "log_loss": log_loss(y_cal, pred_cal),
            "brier": brier_score_loss(y_cal, pred_cal),
        }

        # 2. Platt Scaling (Sigmoid)
        logger.info("Applying Platt scaling...")
        platt = CalibratedClassifierCV(self.model, method="sigmoid", cv="prefit")
        if self.scaler is None:
            raise ValueError("Scaler not loaded!")
        platt.fit(self.scaler.transform(X_train), y_train)
        platt_pred = platt.predict_proba(self.scaler.transform(X_cal))[:, 1]

        results["platt"] = {
            "predictions": platt_pred,
            "mean": platt_pred.mean(),
            "auc": roc_auc_score(y_cal, platt_pred),
            "log_loss": log_loss(y_cal, platt_pred),
            "brier": brier_score_loss(y_cal, platt_pred),
        }

        # 3. Isotonic Regression
        logger.info("Applying isotonic regression...")
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(pred_train, y_train)
        iso_pred = iso.transform(pred_cal)

        results["isotonic"] = {
            "predictions": iso_pred,
            "mean": iso_pred.mean(),
            "auc": roc_auc_score(y_cal, iso_pred),
            "log_loss": log_loss(y_cal, iso_pred),
            "brier": brier_score_loss(y_cal, iso_pred),
            "calibrator": iso,
        }

        # 4. Beta Calibration (custom implementation)
        logger.info("Applying beta calibration...")
        beta_pred = self._beta_calibration(pred_train, y_train, pred_cal)

        results["beta"] = {
            "predictions": beta_pred,
            "mean": beta_pred.mean(),
            "auc": roc_auc_score(y_cal, beta_pred),
            "log_loss": log_loss(y_cal, beta_pred),
            "brier": brier_score_loss(y_cal, beta_pred),
        }

        # Print comparison
        self._print_calibration_results(results, y_cal.mean())

        # Plot calibration curves
        self._plot_calibration_curves(results, y_cal)

        return results

    def _beta_calibration(self, pred_train, y_train, pred_cal):
        """Simple beta calibration to match goal rate"""
        # Find scaling factor that makes mean prediction match actual goal rate
        actual_rate = y_train.mean()
        current_mean = pred_train.mean()

        # Apply power transformation to maintain ranking
        if current_mean > 0:
            power = np.log(actual_rate) / np.log(current_mean)
            calibrated = np.power(pred_cal, power)
        else:
            calibrated = pred_cal * (actual_rate / current_mean)

        return np.clip(calibrated, 0, 1)

    def _print_calibration_results(self, results, actual_goal_rate):
        """Print calibration comparison"""
        logger.info("\n" + "=" * 60)
        logger.info("CALIBRATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Actual goal rate: {actual_goal_rate:.3f}")
        logger.info("\nMethod      | Mean xG | AUC    | Log Loss | Brier Score")
        logger.info("-" * 55)

        for method, metrics in results.items():
            logger.info(
                f"{method:11s} | {metrics['mean']:.3f}   | {metrics['auc']:.4f} | {metrics['log_loss']:.4f}   | {metrics['brier']:.4f}"
            )

    def _plot_calibration_curves(self, results, y_true):
        """Plot reliability diagrams"""
        plt.figure(figsize=(12, 8))

        for i, (method, metrics) in enumerate(results.items()):
            plt.subplot(2, 2, i + 1)

            # Create bins
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            predictions = metrics["predictions"]

            # Calculate actual probability in each bin
            actual_probs = []
            predicted_probs = []

            for j in range(n_bins):
                mask = (predictions >= bin_edges[j]) & (predictions < bin_edges[j + 1])
                if mask.sum() > 0:
                    actual_probs.append(y_true[mask].mean())
                    predicted_probs.append(predictions[mask].mean())

            # Plot
            plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            plt.plot(predicted_probs, actual_probs, "o-", label=f"{method}")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Actual Probability")
            plt.title(f"{method.capitalize()} Calibration")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/calibration_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

    def save_best_calibrator(self, results):
        """Save the best calibration method"""
        # Choose based on lowest Brier score
        best_method = min(results.items(), key=lambda x: x[1]["brier"])[0]

        if best_method == "isotonic" and "calibrator" in results["isotonic"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            joblib.dump(results["isotonic"]["calibrator"], f"models/moneypuck/calibrator_isotonic_{timestamp}.pkl")
            logger.info("\nSaved isotonic calibrator (best method)")


def main():
    """Run model calibration"""
    calibrator = ModelCalibrator()

    # Load trained model
    calibrator.load_trained_model()

    # Load data and get predictions
    X, y, raw_predictions, df = calibrator.load_data_and_predict()

    logger.info(f"\nRaw predictions - Mean: {raw_predictions.mean():.3f}")
    logger.info(f"Actual goal rate: {y.mean():.3f}")
    logger.info(f"MoneyPuck xGoal mean: {df['xGoal'].mean():.3f}")

    # Calibrate
    results = calibrator.calibrate_predictions(X, y, raw_predictions)

    # Save best calibrator
    calibrator.save_best_calibrator(results)

    logger.info("\n✅ Calibration complete! Your model now outputs realistic probabilities.")


if __name__ == "__main__":
    main()
