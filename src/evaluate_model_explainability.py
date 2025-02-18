import pandas as pd
import numpy as np
import shap
import joblib
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Model and Explain Predictions with SHAP")
    parser.add_argument("--model_path", required=True, help="Path to trained model file (fraud_detection_model.pkl)")
    parser.add_argument("--data_path", required=True, help="Path to feature-engineered fraud dataset")
    return parser.parse_args()

def load_and_preprocess_data(data_path):
    """
    Load dataset and ensure all features match training format.
    """
    data = pd.read_csv(data_path)

    # Drop non-numeric columns
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"Dropping non-numeric columns: {non_numeric_cols}")
        data = data.drop(columns=non_numeric_cols)

    # Convert datetime columns to timestamps
    for col in ["signup_time", "purchase_time"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce").astype(int) // 10**9

    # Fill missing values
    if data.isnull().sum().sum() > 0:
        print("Filling missing values with column medians.")
        data = data.fillna(data.median())

    return data

def main():
    args = get_args()

    # Load model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model = joblib.load(args.model_path)

    # Load and preprocess dataset
    data = load_and_preprocess_data(args.data_path)

    # Separate features and target
    if "class" in data.columns:
        X = data.drop(columns=["class"])
    else:
        X = data

    # Ensure SHAP data matches training features
    model_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else X.columns.tolist()
    X = X[model_features]

    # Convert all features to float
    X = X.astype(np.float64)

    print(f"SHAP input shape: {X.shape}, Model input shape: {len(model_features)} features.")

    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X)

    # Compute SHAP values (disable additivity check)
    shap_values = explainer(X, check_additivity=False)

    # Save SHAP values
    shap_output_path = os.path.join(os.path.dirname(args.model_path), "shap_values.pkl")
    joblib.dump(shap_values, shap_output_path)
    print(f"SHAP values saved at {shap_output_path}")

    # Visualize SHAP summary plot
    shap.summary_plot(shap_values, X)

if __name__ == "__main__":
    main()
