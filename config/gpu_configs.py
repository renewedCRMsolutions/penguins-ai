# File: penguins_ai/config/gpu_configs.py
XGBOOST_GPU_CONFIG = {
    'n_estimators': 2000,
    'max_depth': 15,
    'learning_rate': 0.005,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'max_bin': 256,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': 20,
    'random_state': 42
}

LIGHTGBM_CONFIG = {
    'device': 'gpu',
    'gpu_device_id': 0,
    'num_leaves': 255,
    'num_iterations': 2000,
    'learning_rate': 0.005,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_threads': 20,
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42
}