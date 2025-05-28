# File: setup/download_nhl_data.py
# Copy this entire content into the file

import os
# import requests  # Unused import
import pandas as pd
import numpy as np
# from datetime import datetime  # Unused import

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'logs']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    print("✓ Created project directories")

def create_sample_data():
    """Create sample NHL shot data for testing"""
    print("Creating sample NHL shot data...")
    
    # Create realistic sample data
    np.random.seed(42)
    
    teams = ['PIT', 'PHI', 'NYR', 'WSH', 'BOS', 'TOR', 'MTL', 'OTT']
    shot_types = ['Wrist', 'Slap', 'Snap', 'Backhand', 'Tip-In', 'Deflection']
    last_events = ['Shot', 'Pass', 'Carry', 'Faceoff', 'Rebound', 'Rush']
    
    data = []
    
    for season in ['2021', '2022', '2023']:
        n_shots = 5000
        
        for i in range(n_shots):
            # Create realistic shot data
            shot_distance = np.random.gamma(4, 5) + 5  # Most shots from 10-40 feet
            shot_angle = np.random.normal(0, 30)  # Most shots from center
            shot_angle = max(-89, min(89, shot_angle))  # Limit to valid angles
            
            # Base probability influenced by distance and angle
            base_prob = 0.15 * np.exp(-shot_distance/30) * np.exp(-abs(shot_angle)/40)
            
            # Adjust for shot type
            shot_type = np.random.choice(shot_types)
            if shot_type in ['Tip-In', 'Deflection']:
                base_prob *= 1.5
            elif shot_type == 'Slap':
                base_prob *= 0.8
                
            # Rebound shots have higher probability
            is_rebound = np.random.choice([0, 1], p=[0.85, 0.15])
            if is_rebound:
                base_prob *= 2
                
            # Rush shots slightly higher probability
            is_rush = np.random.choice([0, 1], p=[0.7, 0.3])
            if is_rush:
                base_prob *= 1.2
                
            # Determine if goal
            goal = 1 if np.random.random() < base_prob else 0
            
            data.append({
                'season': season,
                'game_id': f"{season}02{np.random.randint(1, 1000):04d}",
                'event_id': f"{season}_{i}",
                'team': np.random.choice(teams),
                'shooter': f"Player_{np.random.randint(1, 50)}",
                'shotDistance': round(shot_distance, 2),
                'shotAngle': round(shot_angle, 2),
                'shotType': shot_type,
                'lastEventType': np.random.choice(last_events),
                'timeSinceLast': round(np.random.exponential(5), 2),
                'isRebound': is_rebound,
                'isRush': is_rush,
                'period': np.random.choice([1, 2, 3], p=[0.33, 0.33, 0.34]),
                'goal': goal
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to files
    for season in ['2021', '2022', '2023']:
        season_df = df[df['season'] == season]
        filename = f'data/shots_{season}.csv'
        season_df.to_csv(filename, index=False)
        print(f"✓ Created {filename} with {len(season_df)} shots")
    
    # Show sample statistics
    print(f"\nOverall statistics:")
    print(f"Total shots: {len(df)}")
    print(f"Goals: {df['goal'].sum()}")
    print(f"Shooting percentage: {df['goal'].mean()*100:.1f}%")
    print(f"Average shot distance: {df['shotDistance'].mean():.1f} feet")

if __name__ == "__main__":
    create_directories()
    create_sample_data()
    print("\n✓ Data setup complete!")