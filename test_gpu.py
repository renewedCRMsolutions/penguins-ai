# test_gpu.py
import xgboost as xgb
import numpy as np

# Force GPU usage
model = xgb.XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor", gpu_id=0)

print(f"Tree method: {model.get_params()['tree_method']}")

X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

try:
    model.fit(X, y)
    print("GPU training successful!")
except Exception as e:
    print(f"GPU error: {e}")
    print("Falling back to CPU")
