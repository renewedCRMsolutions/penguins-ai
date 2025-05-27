# File: penguins_ai/setup/install_models.py

import subprocess
import sys

def setup_penguins_ai():
    """Install all required libraries for Penguins AI"""
    
    # Core ML libraries
    libraries = [
        'scikit-learn',
        'xgboost',
        'pandas',
        'numpy',
        'fastapi',
        'uvicorn',
        'plotly',
        'psycopg2-binary',  # PostgreSQL
        'google-cloud-bigquery',  # For BigQuery
        'google-cloud-storage',
    ]
    
    # Install each library
    for lib in libraries:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])
    
    print("✓ All libraries installed successfully")
    
    # Download sample NHL data for model training
    download_nhl_training_data()

def download_nhl_training_data():
    """Download historical NHL data for model training"""
    import requests
    import pandas as pd
    
    # MoneyPuck provides free historical data
    base_url = "http://peter-tanner.com/moneypuck/downloads/"
    seasons = ["2021", "2022", "2023"]
    
    for season in seasons:
        url = f"{base_url}shots_{season}.csv"
        df = pd.read_csv(url)
        df.to_csv(f"data/shots_{season}.csv")
        print(f"✓ Downloaded {season} shot data")