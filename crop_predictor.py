"""
crop_predictor.py
Prediction and recommendation utility for Crop Suitability.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Path to the trained model artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_suitability_model.pkl")

# Cached model instance
_model = None

FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def _get_model():
    """Lazy loader to cache and return the trained model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_crop_model.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def _format_input(N, P, K, temperature, humidity, ph, rainfall):
    """Encapsulates input variables into a single-row DataFrame with verified column names."""
    data = {
        'N': [float(N)],
        'P': [float(P)],
        'K': [float(K)],
        'temperature': [float(temperature)],
        'humidity': [float(humidity)],
        'ph': [float(ph)],
        'rainfall': [float(rainfall)]
    }
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)

def predict_crop(N, P, K, temperature, humidity, ph, rainfall) -> str:
    """
    Predicts the single most suitable crop for given soil and environmental conditions.
    
    Returns:
        crop (str): Recommended crop label.
    """
    model = _get_model()
    input_df = _format_input(N, P, K, temperature, humidity, ph, rainfall)
    prediction = model.predict(input_df)[0]
    return str(prediction)

def recommend_crops(N, P, K, temperature, humidity, ph, rainfall, top_n=3) -> list:
    """
    Returns top-N crop recommendations ranked by model suitability score.
    
    Returns:
        list of dict: [{"crop": crop_name, "suitability_score": score}, ...]
        Suitability score is the predicted class probability from the model.
    """
    model = _get_model()
    input_df = _format_input(N, P, K, temperature, humidity, ph, rainfall)

    if not hasattr(model, "predict_proba"):
        single_pred = model.predict(input_df)[0]
        return [{"crop": str(single_pred), "suitability_score": 1.0}]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    # Sort classes descending by probability
    ranked_indices = np.argsort(probabilities)[::-1][:top_n]

    recommendations = [
        {
            "crop": str(classes[idx]),
            "suitability_score": round(float(probabilities[idx]), 4)
        }
        for idx in ranked_indices
    ]
    return recommendations
