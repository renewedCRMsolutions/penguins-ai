# File: api/main.py
# Copy this entire content into the file

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Optional  # Create FastAPI app
from fastapi import WebSocket
import asyncio
import aiohttp

app = FastAPI(
    title="Pittsburgh Penguins AI - Expected Goals API",
    description="Hockey analytics API for shot quality prediction",
    version="1.0.0",
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
        model = joblib.load("models/production/xg_model_nhl.pkl")
        with open("models/production/features.txt", "r") as f:
            features = [line.strip() for line in f]
        model_info = joblib.load("models/production/model_metadata.pkl")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Model not loaded: {str(e)}")
        print("Run train/train_xg_model.py first!")


# Load model when server starts
load_model()


# Request/Response models
class ShotData(BaseModel):
    # Spatial features
    arenaAdjustedShotDistance: float
    shotAngleAdjusted: float
    arenaAdjustedXCordABS: float
    arenaAdjustedYCordAbs: float

    # Raw coordinates (for visualization)
    xCord: float
    yCord: float

    # Player quality (critical - these need lookups)
    shooting_talent: float = 1.0  # Default average
    high_danger_conversion: float = 0.15
    shot_quality_ratio: float = 0.3
    save_talent: float = 0.0
    high_danger_save_talent: float = 0.0

    # Shot characteristics
    shotRebound: int = 0
    shotRush: int = 0
    shotWasOnGoal: int = 1
    speedFromLastEvent: float = 10.0
    timeSinceLastEvent: float = 5.0
    distanceFromLastEvent: float = 0.0
    shotType: str = "WRIST"  # WRIST, SLAP, SNAP, BACK, TIP, DEFL, WRAP

    # Pre-shot context
    lastEventCategory: str = "CARRY"  # SHOT, PASS, CARRY, FACEOFF
    lastEventXCord: Optional[float] = None
    lastEventYCord: Optional[float] = None
    timeSinceFaceoff: float = 30.0

    # Game state
    homeSkatersOnIce: int = 5
    awaySkatersOnIce: int = 5
    period: int = 2
    timeLeft: int = 1200
    awayTeamGoals: int = 0
    homeTeamGoals: int = 0

    # Additional context
    shooterPosition: str = "C"  # C, L, R, D
    isPlayoffGame: int = 0
    shooterTimeOnIce: float = 30.0

    # Calculated fields
    score_differential: int = 0
    is_home_shooting: int = 1
    strength_differential: int = 0


class PredictionResponse(BaseModel):
    # Primary prediction
    expected_goals: float
    shot_quality: str  # Excellent, Good, Average, Poor
    danger_zone: str  # High, Medium, Low

    # Additional predictions (if you train for these)
    rebound_probability: Optional[float] = None
    on_goal_probability: Optional[float] = None

    # Analysis
    percentile: float
    recommendation: str
    key_factors: list[str]  # Top 3 factors influencing prediction

    # Spatial info for visualization
    shot_location: dict  # {"x": float, "y": float}
    distance: float
    angle: float
    
async def fetch_game_data(game_id: str):
url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        return await response.json()

def calculate_time_since_last_faceoff(play, pbp_data):
    # Find most recent faceoff before this play
    for i in range(pbp_data['plays'].index(play) - 1, -1, -1):
        if pbp_data['plays'][i]['typeDescKey'] == 'faceoff':
            return play['timeInPeriod'] - pbp_data['plays'][i]['timeInPeriod']
    return 30.0  # Default

player_stats = {} 


@app.websocket("/ws/game/{game_id}")
async def game_websocket(websocket: WebSocket, game_id: str):
    await websocket.accept()

    while True:
        # Fetch latest play-by-play
        pbp_data = await fetch_game_data(game_id)

        # Calculate real metrics
        for play in pbp_data["plays"]:
            if play["eventType"] == "shot":
                # Find last faceoff
                time_since_faceoff = calculate_time_since_last_faceoff(play, pbp_data)

                # Get shooter/goalie quality
                shooter_talent = player_stats[play["shooterId"]]["goals/xgoals"]

                # Send prediction
                await websocket.send_json(
                    {
                        "event": "shot",
                        "xG": predict_with_real_data(play, time_since_faceoff, shooter_talent),
                        "location": {"x": play["x"], "y": play["y"]},
                    }
                )

        await asyncio.sleep(5)  # Poll every 5 seconds


# API Routes
@app.get("/")
def root():
    return {
        "message": "Pittsburgh Penguins AI - Expected Goals API",
        "status": "active" if model is not None else "model not loaded",
        "endpoints": {
            "POST /predict/expected-goals": "Predict xG for a shot",
            "GET /model/info": "Get model information",
            "GET /health": "Check API health",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict/expected-goals", response_model=PredictionResponse)
def predict_xg(shot: ShotData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Log input data
        print(f"Distance: {shot.arenaAdjustedShotDistance:.1f} ft")
        print(f"Angle: {shot.shotAngleAdjusted:.1f}°")
        print(f"Shot Type: {shot.shotType}")
        print(f"Rebound: {shot.shotRebound}")
        print(f"Rush: {shot.shotRush}")

        # Create dataframe and make prediction
        shot_dict = shot.dict()
        shot_df = pd.DataFrame([shot_dict])
        shot_encoded = pd.get_dummies(
            shot_df, columns=["shotType", "lastEventCategory"], prefix=["shotType", "lastEvent"]
        )

        if features is not None:
            for feat in features:
                if feat not in shot_encoded.columns:
                    shot_encoded[feat] = 0
            shot_encoded = shot_encoded[features]
        else:
            raise HTTPException(status_code=503, detail="Model features not loaded")

        xg_probability = float(model.predict_proba(shot_encoded)[0, 1])

        # Print result AFTER calculating it
        print(f"xG Result: {xg_probability:.4f} ({xg_probability * 100:.1f}%)")
        print("========================\n")

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
            danger_zone="Low" if xg_probability < 0.08 else "Medium" if xg_probability < 0.15 else "High",
            percentile=round(percentile, 1),
            recommendation=recommendation,
            key_factors=["Distance", "Angle", "Shot Type"],  # Add feature importance later
            shot_location={"x": shot.xCord, "y": shot.yCord},
            distance=shot.arenaAdjustedShotDistance,
            angle=shot.shotAngleAdjusted,
        )

    except Exception as e:
        print(f"Error details: {str(e)}")
        print(f"Shot data received: {shot.dict()}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/model/info")
def get_model_info():
    """Get information about the loaded model"""
    if model_info is None:
        return {"status": "No model loaded"}

    return {
        "model_type": "XGBoost Classifier",
        "accuracy": round(model_info["accuracy"], 3),
        "auc_score": round(model_info["auc"], 3),
        "n_features": len(model_info["features"]),
        "training_samples": model_info["training_samples"],
        "feature_categories": {
            "numeric": ["shotDistance", "shotAngle", "timeSinceLast", "period"],
            "binary": ["isRebound", "isRush"],
            "categorical": ["shotType", "lastEventType"],
        },
        "shot_types": ["Wrist", "Slap", "Snap", "Backhand", "Tip-In", "Deflection"],
        "event_types": ["Shot", "Pass", "Carry", "Faceoff", "Rebound", "Rush"],
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
        "total_xg": round(sum(p.expected_goals for p in predictions), 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
