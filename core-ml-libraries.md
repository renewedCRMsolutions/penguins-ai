# File: penguins_ai/requirements.txt
# Machine Learning Core
scikit-learn==1.3.0      # For RandomForest, GradientBoosting
xgboost==2.0.0          # For Expected Goals model
lightgbm==4.1.0         # Alternative gradient boosting
tensorflow==2.15.0      # For LSTM player impact model
prophet==1.1.5          # For time series momentum analysis

# Data Processing
pandas==2.1.0
numpy==1.24.0
scipy==1.11.0

# Pre-trained embeddings for player names/teams
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 90MB