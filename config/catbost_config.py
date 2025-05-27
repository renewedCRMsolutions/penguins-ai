# CatBoost configuration for GPU training
CATBOOST_CONFIG = {
    'iterations': 3000,
    'depth': 12,
    'learning_rate': 0.03,
    'task_type': 'GPU',
    'devices': '0',
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'od_wait': 50,
    'random_seed': 42,
    'verbose': 100
}