"""
advisor.py - Integrated Agricultural Advisory Engine (AI Crop Advisor)
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ECONOMIC_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "rice": {"price_per_tonne": 22030.0, "cost_per_hectare": 45000.0, "volatility": 0.15},
    "maize": {"price_per_tonne": 20900.0, "cost_per_hectare": 32000.0, "volatility": 0.22},
    "chickpea": {"price_per_tonne": 54400.0, "cost_per_hectare": 28000.0, "volatility": 0.25},
    "kidneybeans": {"price_per_tonne": 75000.0, "cost_per_hectare": 38000.0, "volatility": 0.35},
    "pigeonpeas": {"price_per_tonne": 70000.0, "cost_per_hectare": 30000.0, "volatility": 0.28},
    "mothbeans": {"price_per_tonne": 60000.0, "cost_per_hectare": 22000.0, "volatility": 0.40},
    "mungbean": {"price_per_tonne": 85580.0, "cost_per_hectare": 25000.0, "volatility": 0.30},
    "blackgram": {"price_per_tonne": 69500.0, "cost_per_hectare": 26000.0, "volatility": 0.32},
    "lentil": {"price_per_tonne": 64250.0, "cost_per_hectare": 27000.0, "volatility": 0.24},
    "pomegranate": {"price_per_tonne": 65000.0, "cost_per_hectare": 110000.0, "volatility": 0.45},
    "banana": {"price_per_tonne": 18000.0, "cost_per_hectare": 125000.0, "volatility": 0.38},
    "mango": {"price_per_tonne": 42000.0, "cost_per_hectare": 85000.0, "volatility": 0.42},
    "grapes": {"price_per_tonne": 52000.0, "cost_per_hectare": 140000.0, "volatility": 0.50},
    "watermelon": {"price_per_tonne": 12000.0, "cost_per_hectare": 48000.0, "volatility": 0.40},
    "muskmelon": {"price_per_tonne": 16000.0, "cost_per_hectare": 45000.0, "volatility": 0.42},
    "apple": {"price_per_tonne": 70000.0, "cost_per_hectare": 160000.0, "volatility": 0.35},
    "orange": {"price_per_tonne": 35000.0, "cost_per_hectare": 75000.0, "volatility": 0.38},
    "papaya": {"price_per_tonne": 15000.0, "cost_per_hectare": 65000.0, "volatility": 0.45},
    "coconut": {"price_per_tonne": 28000.0, "cost_per_hectare": 55000.0, "volatility": 0.20},
    "cotton": {"price_per_tonne": 70200.0, "cost_per_hectare": 52000.0, "volatility": 0.32},
    "jute": {"price_per_tonne": 50500.0, "cost_per_hectare": 36000.0, "volatility": 0.26},
    "coffee": {"price_per_tonne": 140000.0, "cost_per_hectare": 95000.0, "volatility": 0.30},
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATHS = {
    "suitability": os.path.join(CURRENT_DIR, "models", "crop_suitability_model.pkl"),
    "yield": os.path.join(CURRENT_DIR, "models", "crop_yield_model.pkl")
}

class CropAdvisorPipeline:
    def __init__(self, suitability_model_path: str, yield_model_path: str):
        self.suitability_model_path = suitability_model_path
        self.yield_model_path = yield_model_path
        self.suitability_model = None
        self.yield_model = None
        self._load_models()

    def _load_models(self) -> None:
        if not os.path.exists(self.suitability_model_path):
            raise FileNotFoundError(f"Crop Suitability model missing at: {self.suitability_model_path}")
        if not os.path.exists(self.yield_model_path):
            raise FileNotFoundError(f"Yield Prediction model missing at: {self.yield_model_path}")
        self.suitability_model = joblib.load(self.suitability_model_path)
        self.yield_model = joblib.load(self.yield_model_path)

    def get_candidate_crops(self, soil_climate_features: pd.DataFrame, top_k: int = 5, proba_thresh: float = 0.03) -> List[Dict[str, Any]]:
        if not hasattr(self.suitability_model, "predict_proba"):
            preds = self.suitability_model.predict(soil_climate_features)
            return [{"crop": str(preds[0]).lower(), "suitability_score": 1.0}]

        probas = self.suitability_model.predict_proba(soil_climate_features)[0]
        classes = [str(c).lower() for c in self.suitability_model.classes_]

        ranked_indices = np.argsort(probas)[::-1]
        candidates = []
        for idx in ranked_indices:
            score = float(probas[idx])
            if score >= proba_thresh and len(candidates) < top_k:
                candidates.append({
                    "crop": classes[idx],
                    "suitability_score": round(score, 4)
                })
        return candidates

    def predict_yield(self, crop: str, farmer_input: Dict[str, Any]) -> float:
        area_val = float(farmer_input.get("area", 1.0))
        yield_input_dict = {
            "Crop": [crop.capitalize()],
            "Season": [str(farmer_input.get("season", "Kharif")).strip()],
            "State": [str(farmer_input.get("state", "Assam")).strip()],
            "Area": [area_val],
            "Annual_Rainfall": [float(farmer_input.get("annual_rainfall", farmer_input.get("rainfall", 1000.0)))],
            "Fertilizer": [float(farmer_input.get("fertilizer", area_val * 137.0))],
            "Pesticide": [float(farmer_input.get("pesticide", area_val * 0.27))],
            "Crop_Year": [int(farmer_input.get("crop_year", 2020))]
        }
        df_yield = pd.DataFrame(yield_input_dict)
        try:
            pred_yield = self.yield_model.predict(df_yield)[0]
            return max(float(pred_yield), 0.05)
        except Exception as err:
            logger.warning(f"Yield model prediction failed for crop '{crop}': {err}")
            return 0.5

    @staticmethod
    def calculate_economics_and_risk(crop: str, yield_per_ha: float, area_ha: float, suitability_score: float) -> Dict[str, Any]:
        benchmarks = ECONOMIC_BENCHMARKS.get(
            crop.lower(), 
            {"price_per_tonne": 25000.0, "cost_per_hectare": 35000.0, "volatility": 0.35}
        )

        total_yield_tonnes = yield_per_ha * area_ha
        expected_revenue = total_yield_tonnes * benchmarks["price_per_tonne"]
        total_cost = area_ha * benchmarks["cost_per_hectare"]
        estimated_profit = expected_revenue - total_cost
        margin_pct = (estimated_profit / expected_revenue * 100.0) if expected_revenue > 0 else -100.0

        agronomic_risk = 1.0 - suitability_score
        market_risk = benchmarks["volatility"]
        financial_stress_penalty = 0.20 if estimated_profit < 0 else 0.0
        composite_risk = (0.50 * agronomic_risk) + (0.35 * market_risk) + financial_stress_penalty
        composite_risk = float(np.clip(composite_risk, 0.01, 0.99))

        if composite_risk <= 0.35:
            risk_level = "Low"
        elif composite_risk <= 0.60:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return {
            "yield_per_ha": round(yield_per_ha, 3),
            "total_yield": round(total_yield_tonnes, 2),
            "revenue": round(expected_revenue, 2),
            "cost": round(total_cost, 2),
            "profit": round(estimated_profit, 2),
            "margin_pct": round(margin_pct, 2),
            "composite_risk": round(composite_risk, 3),
            "risk_level": risk_level
        }

_PIPELINE_INSTANCE: Optional[CropAdvisorPipeline] = None

def get_crop_recommendations(
    farmer_input: Dict[str, Any],
    top_candidates_count: int = 5,
    suitability_model_path: str = DEFAULT_MODEL_PATHS["suitability"],
    yield_model_path: str = DEFAULT_MODEL_PATHS["yield"]
) -> List[Dict[str, Any]]:
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = CropAdvisorPipeline(suitability_model_path, yield_model_path)

    soil_features_df = pd.DataFrame([{
        "N": float(farmer_input["N"]),
        "P": float(farmer_input["P"]),
        "K": float(farmer_input["K"]),
        "temperature": float(farmer_input["temperature"]),
        "humidity": float(farmer_input["humidity"]),
        "ph": float(farmer_input["ph"]),
        "rainfall": float(farmer_input["rainfall"])
    }])

    farm_area = float(farmer_input.get("area", 1.0))
    candidates = _PIPELINE_INSTANCE.get_candidate_crops(soil_features_df, top_k=top_candidates_count)

    evaluated_candidates = []
    for cand in candidates:
        crop_name = cand["crop"]
        suit_score = cand["suitability_score"]
        pred_yield = _PIPELINE_INSTANCE.predict_yield(crop_name, farmer_input)
        econ = _PIPELINE_INSTANCE.calculate_economics_and_risk(crop_name, pred_yield, farm_area, suit_score)

        evaluated_candidates.append({
            "crop_name": crop_name,
            "suitability_score": suit_score,
            "predicted_yield_per_ha": econ["yield_per_ha"],
            "total_predicted_yield": econ["total_yield"],
            "expected_revenue": econ["revenue"],
            "estimated_cost": econ["cost"],
            "estimated_profit": econ["profit"],
            "profit_margin_pct": econ["margin_pct"],
            "composite_risk_score": econ["composite_risk"],
            "risk_level": econ["risk_level"]
        })

    profits = [c["estimated_profit"] for c in evaluated_candidates]
    min_p, max_p = min(profits), max(profits)
    spread = (max_p - min_p) if (max_p - min_p) > 0 else 1.0

    for item in evaluated_candidates:
        norm_profit = (item["estimated_profit"] - min_p) / spread
        utility = (0.40 * norm_profit) + (0.35 * item["suitability_score"]) - (0.25 * item["composite_risk_score"])
        item["_utility"] = utility

    evaluated_candidates.sort(key=lambda x: x["_utility"], reverse=True)

    recommendations = []
    for i, item in enumerate(evaluated_candidates, start=1):
        item.pop("_utility", None)
        item["rank"] = i
        recommendations.append(item)

    return recommendations
