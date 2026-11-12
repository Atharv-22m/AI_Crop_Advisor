"""
profit_risk.py
Profit and Risk Evaluation module for AI Crop Advisor.

This module provides deterministic, rule-based and statistical economic calculations.
It does NOT use or pretend to use machine learning.

Units & Currency Standards:
----------------------------
- Land Area: Hectares (ha)
- Predicted Yield: Metric tonnes per hectare (tonnes/ha)
- Expected Production: Metric tonnes (tonnes)
- Expected Market Price: Indian National Rupees per metric tonne (INR/tonne)
- Cultivation Cost: Indian National Rupees per hectare (INR/ha)
- Expected Revenue: Indian National Rupees (INR)
- Estimated Profit: Indian National Rupees (INR)
- Profit Margin: Percentage (%)
- Risk Level: Categorical ('Low', 'Moderate', 'High') accompanied by transparent component metrics.

Assumptions & Methodological Transparency:
------------------------------------------
1. Revenue is calculated as linear gross receipts before post-harvest deductions:
   Production (tonnes) = Predicted Yield (tonnes/ha) * Area (ha)
   Revenue (INR) = Production (tonnes) * Market Price (INR/tonne)
2. Total Cost scales with cultivated land area:
   Total Cost (INR) = Cultivation Cost per ha (INR/ha) * Area (ha)
3. Profit represents net farm operating margin:
   Profit (INR) = Expected Revenue - Total Cultivation Cost
4. Risk is evaluated via statistical and financial dispersion:
   - Yield Variability: Measured using historical Coefficient of Variation (CV = std / mean)
     derived directly from the historical crop yield dataset.
   - Profit Margin Buffer: Low or negative profit margins increase financial vulnerability.
   - Risk is an indicative heuristic, NOT a scientifically exact guarantee of loss or gain.
"""

import os
import json
from typing import Dict, Any, Optional

# File paths for economic configurations and historical statistics
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
ECONOMICS_FILE = os.path.join(PROJECT_ROOT, "data", "crop_economics.json")
STATS_FILE = os.path.join(PROJECT_ROOT, "data", "crop_yield_stats.json")

# Crop naming normalization map between recommendation dataset and yield dataset
CROP_NAME_MAPPINGS = {
    "chickpea": "Gram",
    "pigeonpeas": "Arhar/Tur",
    "mungbean": "Moong(Green Gram)",
    "blackgram": "Urad",
    "lentil": "Masoor",
    "cotton": "Cotton(lint)",
    "mothbeans": "Moth",
    "rice": "Rice",
    "maize": "Maize",
    "banana": "Banana",
    "coconut": "Coconut",
    "jute": "Jute",
    "watermelon": "Watermelon",
    "muskmelon": "Muskmelon",
    "apple": "Apple",
    "orange": "Orange",
    "papaya": "Papaya",
    "grapes": "Grapes",
    "mango": "Mango",
    "pomegranate": "Pomegranate",
    "coffee": "Coffee",
    "kidneybeans": "Kidneybeans"
}

def _load_json(file_path: str) -> Dict[str, Any]:
    """Helper to safely load a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

_ECONOMICS_DATA = _load_json(ECONOMICS_FILE)
_YIELD_STATS = _load_json(STATS_FILE)


def normalize_crop_name(crop_name: str) -> str:
    """Matches user or model crop input against standardized catalog keys."""
    raw = str(crop_name).strip()
    lowered = raw.lower()
    if lowered in CROP_NAME_MAPPINGS:
        return CROP_NAME_MAPPINGS[lowered]
    # Check direct case-insensitive match
    for k in _ECONOMICS_DATA:
        if k.lower() == lowered:
            return k
    for k in _YIELD_STATS:
        if k.lower() == lowered:
            return k
    return raw


def get_crop_economics(crop: str) -> Dict[str, Any]:
    """
    Retrieves benchmark market price and cultivation cost from configuration.
    Returns fallback defaults if specific benchmark data is unavailable.
    """
    canonical_crop = normalize_crop_name(crop)
    if canonical_crop in _ECONOMICS_DATA:
        entry = _ECONOMICS_DATA[canonical_crop]
        return {
            "crop": canonical_crop,
            "market_price_per_tonne": float(entry["market_price_per_tonne"]),
            "cultivation_cost_per_ha": float(entry["cultivation_cost_per_ha"]),
            "is_default": False,
            "description": entry.get("description", "")
        }
    
    # Sensible fallback averages across Indian agronomic baselines
    return {
        "crop": canonical_crop,
        "market_price_per_tonne": 25000.0,
        "cultivation_cost_per_ha": 40000.0,
        "is_default": True,
        "description": "General agronomic benchmark fallback (custom input recommended)"
    }


def calculate_production(predicted_yield: float, area: float) -> float:
    """
    Calculates expected total production in metric tonnes.
    Formula: Predicted Yield (tonnes/ha) * Area (ha)
    """
    if predicted_yield < 0 or area <= 0:
        return 0.0
    return round(float(predicted_yield) * float(area), 2)


def calculate_revenue(production: float, market_price_per_tonne: float) -> float:
    """
    Calculates expected gross revenue in INR.
    Formula: Total Production (tonnes) * Market Price (INR/tonne)
    """
    if production <= 0 or market_price_per_tonne <= 0:
        return 0.0
    return round(float(production) * float(market_price_per_tonne), 2)


def calculate_cultivation_cost(cultivation_cost_per_ha: float, area: float) -> float:
    """
    Calculates total operational cultivation expenditure in INR.
    Formula: Cultivation Cost (INR/ha) * Area (ha)
    """
    if cultivation_cost_per_ha <= 0 or area <= 0:
        return 0.0
    return round(float(cultivation_cost_per_ha) * float(area), 2)


def calculate_profit(revenue: float, total_cost: float) -> Dict[str, float]:
    """
    Calculates net estimated profit and operating profit margin.
    Formula:
        Net Profit (INR) = Revenue - Total Cost
        Profit Margin (%) = (Net Profit / Revenue) * 100
    """
    net_profit = float(revenue) - float(total_cost)
    margin = (net_profit / revenue * 100.0) if revenue > 0 else -100.0
    return {
        "estimated_profit": round(net_profit, 2),
        "profit_margin_percent": round(margin, 2)
    }


def calculate_risk(crop: str, predicted_yield: float, profit_margin_percent: float) -> Dict[str, Any]:
    """
    Calculates relative risk using historical statistical yield variability and financial buffer.
    
    Statistical components:
    - Yield Coefficient of Variation (CV = std / mean): Historical volatility of crop yield.
    - Financial Margin Buffer: Ability to absorb unfavorable weather/price shocks.
    
    Returns structured risk metrics and rule-based risk rating: Low, Moderate, or High.
    """
    canonical_crop = normalize_crop_name(crop)
    stats = _YIELD_STATS.get(canonical_crop, None)

    if stats and stats.get("count", 0) > 0:
        hist_mean = stats["mean_yield"]
        hist_std = stats["std_yield"]
        hist_cv = stats["cv_yield"]
        hist_min = stats["min_yield"]
        hist_max = stats["max_yield"]
        data_available = True
    else:
        # Fallback values if crop not directly recorded in historical yield dataset
        hist_mean = predicted_yield
        hist_std = 0.0
        hist_cv = 0.50
        hist_min = 0.0
        hist_max = predicted_yield * 2.0
        data_available = False

    # Risk Classification Rules:
    # 1. High Risk: Negative profit margin OR extreme historical volatility (CV > 1.2)
    # 2. Moderate Risk: Modest profit margin (0% to 25%) OR moderate volatility (0.6 <= CV <= 1.2)
    # 3. Low Risk: Healthy profit margin (> 25%) AND stable historical yield (CV < 0.6)
    if profit_margin_percent < 0:
        risk_level = "High"
        risk_reason = "Expected revenue is below estimated cultivation cost (negative profit margin)."
    elif hist_cv > 1.20:
        risk_level = "High"
        risk_reason = f"Historical yield variability is high (Coefficient of Variation: {hist_cv:.2f})."
    elif profit_margin_percent < 25.0 or hist_cv >= 0.60:
        risk_level = "Moderate"
        risk_reason = "Moderate profit margin or moderate historical yield variance observed."
    else:
        risk_level = "Low"
        risk_reason = f"Strong profit margin ({profit_margin_percent:.1f}%) and stable yield history (CV: {hist_cv:.2f})."

    return {
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "statistical_metrics": {
            "has_historical_stats": data_available,
            "historical_mean_yield": hist_mean,
            "historical_std_yield": hist_std,
            "coefficient_of_variation": hist_cv,
            "historical_min_yield": hist_min,
            "historical_max_yield": hist_max
        }
    }


def evaluate_crop(
    crop: str,
    predicted_yield: float,
    area: float,
    custom_market_price_per_tonne: Optional[float] = None,
    custom_cultivation_cost_per_ha: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates production, economics, and risk for a designated crop.

    Parameters:
        crop (str): Name of the crop.
        predicted_yield (float): Predicted yield in metric tonnes per hectare (tonnes/ha).
        area (float): Cultivated farm area in hectares (ha).
        custom_market_price_per_tonne (float, optional): Custom market price in INR per tonne.
        custom_cultivation_cost_per_ha (float, optional): Custom cost in INR per hectare.

    Returns:
        dict: Complete structured evaluation report ready for consumption by Streamlit or API.
    """
    econ = get_crop_economics(crop)
    market_price = float(custom_market_price_per_tonne) if custom_market_price_per_tonne is not None else econ["market_price_per_tonne"]
    cost_per_ha = float(custom_cultivation_cost_per_ha) if custom_cultivation_cost_per_ha is not None else econ["cultivation_cost_per_ha"]

    # Production & Financials
    production = calculate_production(predicted_yield=predicted_yield, area=area)
    revenue = calculate_revenue(production=production, market_price_per_tonne=market_price)
    total_cost = calculate_cultivation_cost(cultivation_cost_per_ha=cost_per_ha, area=area)
    profit_dict = calculate_profit(revenue=revenue, total_cost=total_cost)

    # Statistical Risk Evaluation
    risk_dict = calculate_risk(
        crop=crop,
        predicted_yield=predicted_yield,
        profit_margin_percent=profit_dict["profit_margin_percent"]
    )

    return {
        "crop": crop,
        "canonical_name": econ["crop"],
        "input_parameters": {
            "area_ha": round(float(area), 2),
            "predicted_yield_tonnes_per_ha": round(float(predicted_yield), 2),
            "market_price_per_tonne_inr": round(market_price, 2),
            "cultivation_cost_per_ha_inr": round(cost_per_ha, 2),
            "price_source_note": "User defined" if custom_market_price_per_tonne is not None else econ["description"]
        },
        "production_and_financials": {
            "expected_production_tonnes": production,
            "expected_revenue_inr": revenue,
            "estimated_cultivation_cost_inr": total_cost,
            "estimated_profit_inr": profit_dict["estimated_profit"],
            "profit_margin_percent": profit_dict["profit_margin_percent"],
            "currency": "INR"
        },
        "risk_evaluation": risk_dict
    }
