"""
train_crop_model.py
Training pipeline for Crop Suitability Model (AI Crop Advisor).
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    data_path = os.path.join("data", "Crop_recommendation.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load dataset
    df = pd.read_csv(data_path)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    X = df[feature_cols]
    y = df[target_col]

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Define candidate model pipelines
    candidate_models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Decision Tree': Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

    print("--- Training & Evaluating Models ---")
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    for name, pipe in candidate_models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        print(f"\nModel: {name}")
        print(f"  Accuracy:         {acc:.4f}")
        print(f"  Macro Precision:  {prec_macro:.4f}")
        print(f"  Macro Recall:     {rec_macro:.4f}")
        print(f"  Macro F1-Score:   {f1_macro:.4f}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name
            best_pipeline = pipe

    print(f"\nSelected Final Model: {best_model_name} (Macro F1: {best_f1:.4f})")

    # Confusion matrix for final model
    y_final_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_final_pred)
    print(f"Confusion Matrix Shape: {cm.shape}")

    # Save final pipeline
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "crop_suitability_model.pkl")
    joblib.dump(best_pipeline, save_path)
    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    main()
