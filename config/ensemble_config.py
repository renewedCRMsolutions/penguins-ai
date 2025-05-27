# File: config/ensemble_config.py
LIGHTGBM_CONFIG = {
    'device': 'gpu',
    'gpu_device_id': 0,
    'num_leaves': 255,
    'num_iterations': 2000,
    'num_threads': 20
}