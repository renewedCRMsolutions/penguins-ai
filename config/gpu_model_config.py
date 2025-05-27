# File: config/gpu_model_config.py
GPU_XGBOOST_CONFIG = {
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'n_estimators': 2000,  # Double it
    'max_depth': 15,       # Deeper with GPU
    'learning_rate': 0.005,
    'gpu_id': 0,
    'max_bin': 256
}