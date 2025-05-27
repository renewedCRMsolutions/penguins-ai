# check_model.py
import joblib

metadata = joblib.load('models/production/model_metadata.pkl')
print("\n🏒 Model Results:")
print(f"AUC Score: {metadata.get('auc', 'N/A')}")
print(f"Accuracy: {metadata.get('accuracy', 'N/A')}")
print(f"Training samples: {metadata.get('training_samples', 'N/A')}")