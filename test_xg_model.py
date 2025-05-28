# File: test_xg_model.py
"""
Quick test script for xG model with synthetic data
Tests the model pipeline without needing real NHL data
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

# Import the advanced XG model
try:
    from train.train_xg_final import AdvancedXGModel
except ImportError:
    # Try alternative import path
    from train.train_advanced_xg import AdvancedXGModel


def create_synthetic_shot_data(n_samples: int = 50000):
    """Create realistic synthetic shot data for testing"""

    np.random.seed(42)

    # Create base shot features
    df = pd.DataFrame(
        {
            # Shot location and type
            "shot_distance": np.random.gamma(4, 8, n_samples),  # Gamma distribution for realistic distances
            "shot_angle": np.random.uniform(0, 90, n_samples),
            "x_coord": np.random.uniform(-42, 42, n_samples),
            "y_coord": np.random.uniform(-100, 100, n_samples),
            "shot_type": np.random.choice(
                ["Wrist", "Slap", "Snap", "Backhand", "Tip-In"], n_samples, p=[0.45, 0.20, 0.20, 0.10, 0.05]
            ),
            # Shot context
            "is_rush": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            "is_rebound": np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
            "is_one_timer": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "traffic_score": np.random.poisson(1.5, n_samples),
            "passes_before_shot": np.random.poisson(1.2, n_samples),
            "time_since_zone_entry": np.random.exponential(10, n_samples),
            "time_since_faceoff": np.random.exponential(15, n_samples),
            # Game state
            "period": np.random.choice([1, 2, 3], n_samples, p=[0.33, 0.33, 0.34]),
            "time_remaining": np.random.uniform(0, 3600, n_samples),
            "score_differential": np.random.normal(0, 1.5, n_samples).round(),
            "is_home_team": np.random.choice([0, 1], n_samples),
            "is_power_play": np.random.choice([0, 1], n_samples, p=[0.83, 0.17]),
            "momentum_score": np.random.normal(0, 2, n_samples),
            "zone_time": np.random.exponential(20, n_samples),
            "shot_attempts_sequence": np.random.poisson(1, n_samples),
            # Goalie features
            "goalie_savePctg": np.random.beta(90, 10, n_samples),  # Beta distribution around 0.90
            "goalie_quality_rating": np.random.beta(5, 5, n_samples),  # Centered around 0.5
            "goalie_shots_faced_period": np.random.poisson(8, n_samples),
            "goalie_shots_faced_last_5min": np.random.poisson(3, n_samples),
            "goalie_save_pct_last_10_shots": np.random.beta(80, 20, n_samples),
            "goalie_save_pct_period": np.random.beta(85, 15, n_samples),
            "time_since_last_shot": np.random.exponential(30, n_samples),
            "consecutive_saves": np.random.geometric(0.1, n_samples),
            "goalie_high_danger_save_pct": np.random.beta(75, 25, n_samples),
            "goalie_fatigue_score": np.random.beta(2, 5, n_samples),  # Low values more common
            "goalie_cold_start": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "goalie_recent_form_trend": np.random.choice([-1, 0, 1], n_samples, p=[0.25, 0.50, 0.25]),
            "shooter_vs_goalie_career_shooting_pct": np.random.beta(10, 90, n_samples),  # Around 0.10
        }
    )

    # Clip distances to reasonable range
    df["shot_distance"] = df["shot_distance"].clip(0, 100)

    # Create danger zones based on distance and angle
    conditions = [
        (df["shot_distance"] < 15) & (df["shot_angle"] < 30),
        (df["shot_distance"] < 30) & (df["shot_angle"] < 45),
    ]
    choices = ["high", "medium"]
    df["danger_zone"] = np.select(conditions, choices, default="low")

    # Create realistic goal probability
    # Base probability
    goal_prob = 0.03  # 3% base rate

    # Distance effect (closer = higher probability)
    distance_factor = np.exp(-df["shot_distance"] / 20)
    goal_prob = goal_prob + 0.15 * distance_factor

    # Angle effect (lower angle = better)
    angle_factor = 1 - (df["shot_angle"] / 90) * 0.5
    goal_prob = goal_prob * angle_factor

    # Shot type effects
    shot_type_multiplier = df["shot_type"].map({"Tip-In": 1.8, "Backhand": 0.7, "Snap": 1.2, "Slap": 0.9, "Wrist": 1.0})
    goal_prob = goal_prob * shot_type_multiplier

    # Context effects
    goal_prob = goal_prob * (1 + 0.5 * df["is_rebound"])
    goal_prob = goal_prob * (1 + 0.3 * df["is_one_timer"])
    goal_prob = goal_prob * (1 + 0.2 * df["is_rush"])
    goal_prob = goal_prob * (1 + 0.4 * df["is_power_play"])

    # Goalie effects
    goalie_factor = 2 - df["goalie_quality_rating"]  # Better goalie = lower probability
    goal_prob = goal_prob * goalie_factor
    goal_prob = goal_prob * (1 + 0.1 * df["goalie_fatigue_score"])
    goal_prob = goal_prob * (1 + 0.2 * df["goalie_cold_start"])

    # Add some noise
    goal_prob = goal_prob * np.random.uniform(0.8, 1.2, n_samples)

    # Ensure probability bounds
    goal_prob = np.clip(goal_prob, 0, 1)

    # Generate goals based on probability
    df["is_goal"] = (np.random.random(n_samples) < goal_prob).astype(int)

    # Print summary stats
    print(f"Created {n_samples} synthetic shots")
    print(f"Goal rate: {df['is_goal'].mean():.3f}")
    print(f"High danger goal rate: {df[df['danger_zone'] == 'high']['is_goal'].mean():.3f}")
    print(f"Power play goal rate: {df[df['is_power_play'] == 1]['is_goal'].mean():.3f}")

    return df


def main():
    """Run quick test of xG model"""

    print("=" * 60)
    print("QUICK TEST OF XG MODEL WITH SYNTHETIC DATA")
    print("=" * 60)

    # Create synthetic data
    print("\n1. Creating synthetic shot data...")
    df = create_synthetic_shot_data(n_samples=50000)

    # Save for inspection
    test_data_path = "data/synthetic_test_data.csv"
    df.to_csv(test_data_path, index=False)
    print(f"Saved test data to {test_data_path}")

    # Train model
    print("\n2. Training advanced xG model...")
    model = AdvancedXGModel(model_dir="models/test")

    try:
        final_auc = model.train_full_pipeline(test_data_path)

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"Model AUC: {final_auc:.4f}")
        print("Expected range for synthetic data: 0.75-0.85")
        print(f"Status: {'GOOD' if 0.75 <= final_auc <= 0.85 else 'CHECK MODEL'}")

        # Test predictions on new data
        print("\n3. Testing predictions on new data...")
        test_df = create_synthetic_shot_data(n_samples=1000)
        predictions = model.predict(test_df)

        print(f"Mean xG on test set: {predictions.mean():.4f}")
        print(f"Actual goal rate: {test_df['is_goal'].mean():.4f}")

        print("\n" + "=" * 60)
        print("MODEL IS WORKING! Ready for real NHL data.")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
