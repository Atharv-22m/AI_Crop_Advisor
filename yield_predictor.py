"""
yield_predictor.py
Reusable prediction module for Crop Yield Prediction (AI Crop Advisor).
"""

import os
import joblib
import pandas as pd

# Path to the serialized regression pipeline artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "crop_yield_model.pkl")

# Cached model instance
_model = None

FEATURE_COLUMNS = ['Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_Year']

def _get_model():
    """Lazy loader to cache and return the trained yield prediction model."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run train_yield_model.py first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_yield(crop: str, season: str, state: str, area: float,
                  annual_rainfall: float, fertilizer: float, pesticide: float,
                  crop_year: int = 2020) -> float:
    """
    Predicts crop yield per unit area (metric tonnes per hectare).

    Parameters:
        crop (str): Name of the crop (e.g., 'Rice', 'Wheat', 'Maize').
        season (str): Cropping season (e.g., 'Kharif', 'Rabi', 'Whole Year').
        state (str): State/location name (e.g., 'Punjab', 'Assam', 'Karnataka').
        area (float): Land area under cultivation in hectares.
        annual_rainfall (float): Annual rainfall in mm.
        fertilizer (float): Total fertilizer used in kg.
        pesticide (float): Total pesticide used in kg.
        crop_year (int): Cropping harvest year (default: 2020).

    Returns:
        float: Predicted crop yield (metric tonnes per hectare), rounded to 2 decimal places.
    """
    model = _get_model()

    # Clean string inputs
    crop_clean = str(crop).strip()
    season_clean = str(season).strip()
    state_clean = str(state).strip()

    # Prepare DataFrame matching pipeline input schema
    input_data = pd.DataFrame([{
        'Crop': crop_clean,
        'Season': season_clean,
        'State': state_clean,
        'Area': float(area),
        'Annual_Rainfall': float(annual_rainfall),
        'Fertilizer': float(fertilizer),
        'Pesticide': float(pesticide),
        'Crop_Year': int(crop_year)
    }], columns=FEATURE_COLUMNS)

    predicted_yield = model.predict(input_data)[0]

    # Non-negative constraint for physical validity
    final_yield = max(0.0, float(predicted_yield))
    return round(final_yield, 2)
