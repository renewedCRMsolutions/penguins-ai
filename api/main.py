# File: api/main.py
# Copy this entire content into the file

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Create FastAPI app
app = FastAPI(
    title="Pittsburgh Penguins AI - Expected Goals API",
    description="Hockey analytics API for shot quality prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
features = None
model_info = None

# Load model on startup
def load_model():
    global model, features, model_info
    try:
        model = joblib.load('models/production/xg_model_nhl.pkl')
        features = joblib.load('models/production/xg_features_nhl.pkl')
        model_info = joblib.load('models/model_info.pkl')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Model not loaded: {str(e)}")
        print("Run train/train_xg_model.py first!")

# Load model when server starts
load_model()

# Request/Response models
class ShotData(BaseModel):
    shotDistance: float
    shotAngle: float
    shotType: str = "Wrist"
    lastEventType: str = "Pass"
    timeSinceLast: float = 5.0
    isRebound: int = 0
    isRush: int = 0
    period: int = 2

class PredictionResponse(BaseModel):
    expected_goals: float
    shot_quality: str
    percentile: float
    recommendation: str

# API Routes
@app.get("/")
def root():
    return {
        "message": "Pittsburgh Penguins AI - Expected Goals API",
        "status": "active" if model is not None else "model not loaded",
        "endpoints": {
            "POST /predict/expected-goals": "Predict xG for a shot",
            "GET /model/info": "Get model information",
            "GET /health": "Check API health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict/expected-goals", response_model=PredictionResponse)
def predict_xg(shot: ShotData):
    """Predict expected goals (xG) for a shot"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please train the model first using train/train_xg_model.py"
        )
    
    try:
        # Create dataframe from input
        shot_dict = shot.dict()
        shot_df = pd.DataFrame([shot_dict])
        
        # One-hot encode to match training features
        shot_encoded = pd.get_dummies(
            shot_df, 
            columns=['shotType', 'lastEventType'], 
            prefix=['shot', 'event']
        )
        
        # Ensure all features are present (fill missing with 0)
        if features is not None:
            for feat in features:
                if feat not in shot_encoded.columns:
                    shot_encoded[feat] = 0
            
            # Reorder columns to match training
            shot_encoded = shot_encoded[features]
        else:
            # Handle case where model isn't loaded
            raise HTTPException(status_code=503, detail="Model features not loaded")
        
        # Make prediction
        xg_probability = float(model.predict_proba(shot_encoded)[0, 1])
        
        # Calculate percentile (simplified - in production, use historical data)
        percentile = min(99, max(1, xg_probability * 100 * 2.5))
        
        # Determine shot quality
        if xg_probability > 0.20:
            quality = "Excellent"
            recommendation = "High-danger chance! This shot location/type should be prioritized."
        elif xg_probability > 0.12:
            quality = "Good"
            recommendation = "Quality scoring chance. Continue creating these opportunities."
        elif xg_probability > 0.08:
            quality = "Average"
            recommendation = "Decent shot, but look for better positioning if possible."
        else:
            quality = "Poor"
            recommendation = "Low-percentage shot. Consider passing or improving position."
        
        return PredictionResponse(
            expected_goals=round(xg_probability, 4),
            shot_quality=quality,
            percentile=round(percentile, 1),
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
def get_model_info():
    """Get information about the loaded model"""
    if model_info is None:
        return {"status": "No model loaded"}
    
    return {
        "model_type": "XGBoost Classifier",
        "accuracy": round(model_info['accuracy'], 3),
        "auc_score": round(model_info['auc'], 3),
        "n_features": len(model_info['features']),
        "training_samples": model_info['training_samples'],
        "feature_categories": {
            "numeric": ["shotDistance", "shotAngle", "timeSinceLast", "period"],
            "binary": ["isRebound", "isRush"],
            "categorical": ["shotType", "lastEventType"]
        },
        "shot_types": ["Wrist", "Slap", "Snap", "Backhand", "Tip-In", "Deflection"],
        "event_types": ["Shot", "Pass", "Carry", "Faceoff", "Rebound", "Rush"]
    }

@app.post("/predict/batch")
def predict_batch(shots: list[ShotData]):
    """Predict xG for multiple shots"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for shot in shots:
        pred = predict_xg(shot)
        predictions.append(pred)
    
    return {
        "predictions": predictions,
        "average_xg": round(sum(p.expected_goals for p in predictions) / len(predictions), 4),
        "total_xg": round(sum(p.expected_goals for p in predictions), 2)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)