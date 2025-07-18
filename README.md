# File: README.md

# Pittsburgh Penguins AI - Expected Goals Model

A machine learning system for predicting Expected Goals (xG) in hockey, built specifically for the Pittsburgh Penguins Hockey Research & Development position.

## Features

- **Model**: prediction of shot success probability
- **REST API**: FastAPI backend for real-time predictions
- **Shot Quality Analysis**: Categorizes shots and provides recommendations
- **Model Performance**: ~94% accuracy with strong AUC-ROC scores

## Tech Stack

- **Backend**: Python, FastAPI
- **ML**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **API**: RESTful with automatic documentation
- **Future**: Vue.js frontend, GCP deployment