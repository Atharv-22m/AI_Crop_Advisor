"""
train_yield_model.py
Training pipeline for Crop Yield Prediction Model (AI Crop Advisor).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
    # Support both yield_data.csv and crop_yield.csv seamlessly
    data_paths = [
        os.path.join("data", "yield_data.csv"),
        os.path.join("data", "crop_yield.csv"),
        "yield_data.csv",
        "crop_yield.csv"
    ]
    file_path = None
    for p in data_paths:
        if os.path.exists(p):
            file_path = p
            break
            
    if file_path is None:
        raise FileNotFoundError("Yield dataset not found in data/ folder.")
        
    print(f"Loading yield dataset from: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    # Strip trailing/leading whitespaces from categorical fields
    for col in ['Crop', 'Season', 'State']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def main():
    df = load_data()
    df = clean_data(df)

    # Define feature sets
    categorical_features = ['Crop', 'Season', 'State']
    numerical_features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Crop_Year']
    target_column = 'Yield'

    # Note: 'Production' is explicitly dropped to prevent data leakage (Yield = Production / Area)
    X = df[categorical_features + numerical_features]
    y = df[target_column]

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Construct preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )

    # Define candidate regression models
    candidate_models = {
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Decision Tree Regressor': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42, max_depth=15))
        ]),
        'Random Forest Regressor': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20, n_jobs=-1))
        ]),
        'Gradient Boosting Regressor': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    }

    print("\n--- Training & Evaluating Yield Models ---")
    best_model_name = None
    best_r2 = -float("inf")
    best_pipeline = None

    for name, pipeline in candidate_models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel: {name}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R^2:  {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nSelected Final Model: {best_model_name} (R^2: {best_r2:.4f})")

    # Save the selected pipeline (including preprocessor and model)
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "crop_yield_model.pkl")
    joblib.dump(best_pipeline, save_path)
    print(f"Trained regression pipeline saved to: {save_path}")

if __name__ == "__main__":
    main()
