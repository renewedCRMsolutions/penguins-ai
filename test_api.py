# File: test_api.py

import requests
# import json  # Unused import

def test_api():
    """Test the Expected Goals API"""
    
    # API base URL
    base_url = "http://localhost:8000"
    
    print("Testing Penguins AI API...")
    print("=" * 50)
    
    # Test 1: Check health
    response = requests.get(f"{base_url}/health")
    print(f"Health Check: {response.json()}")
    
    # Test 2: Get model info
    response = requests.get(f"{base_url}/model/info")
    info = response.json()
    print(f"\nModel Info:")
    print(f"  - Model Type: {info.get('model_type', 'N/A')}")
    print(f"  - Accuracy: {info.get('accuracy', 'N/A')}")
    print(f"  - AUC Score: {info.get('auc_score', 'N/A')}")
    
    # Test 3: Make predictions
    test_shots = [
        {
            "shotDistance": 10,
            "shotAngle": 0,
            "shotType": "Wrist",
            "lastEventType": "Pass",
            "timeSinceLast": 2,
            "isRebound": 1,
            "isRush": 0,
            "period": 2
        },
        {
            "shotDistance": 40,
            "shotAngle": 45,
            "shotType": "Slap",
            "lastEventType": "Carry",
            "timeSinceLast": 10,
            "isRebound": 0,
            "isRush": 1,
            "period": 3
        }
    ]
    
    print("\n\nShot Predictions:")
    print("-" * 50)
    
    for i, shot in enumerate(test_shots, 1):
        response = requests.post(f"{base_url}/predict/expected-goals", json=shot)
        if response.status_code == 200:
            result = response.json()
            print(f"\nShot {i}:")
            print(f"  Distance: {shot['shotDistance']}ft, Angle: {shot['shotAngle']}°")
            print(f"  Type: {shot['shotType']}, Rebound: {'Yes' if shot['isRebound'] else 'No'}")
            print(f"  Expected Goals: {result['expected_goals']:.1%}")
            print(f"  Quality: {result['shot_quality']}")
            print(f"  Recommendation: {result['recommendation']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_api()