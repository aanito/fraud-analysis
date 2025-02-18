import pandas as pd
import numpy as np
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_args():
    parser = argparse.ArgumentParser(description="Train Fraud Detection Model")
    parser.add_argument("--fraud_data", required=True, help="Path to engineered_Fraud_Data.csv")
    parser.add_argument("--creditcard_data", required=True, help="Path to cleaned_creditcard.csv")
    parser.add_argument("--output_dir", default="outputs/model", help="Directory to save trained model")
    return parser.parse_args()

def preprocess_data(data, target_column):
    """
    Splits data into train/test sets and ensures all features are numeric.
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data!")

    # Convert datetime columns to numerical timestamps
    for col in ["signup_time", "purchase_time"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce").astype(int) // 10**9  # Convert to Unix timestamp
    
    # Drop non-numeric columns (if any)
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print("Dropping non-numeric columns:", list(non_numeric_cols))
        data = data.drop(columns=non_numeric_cols)

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    args = get_args()

    # Load datasets
    fraud_data = pd.read_csv(args.fraud_data)
    creditcard_data = pd.read_csv(args.creditcard_data)

    # Verify target column exists
    if "class" not in fraud_data.columns:
        print("Error: 'class' column not found in fraud_data!")
        print("Available columns:", fraud_data.columns)
        return
    
    if "Class" not in creditcard_data.columns:
        print("Error: 'Class' column not found in creditcard_data!")
        print("Available columns:", creditcard_data.columns)
        return

    # Preprocess data
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = preprocess_data(fraud_data, "class")
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = preprocess_data(creditcard_data, "Class")

    # Combine both fraud and creditcard data
    X_train = pd.concat([X_train_fraud, X_train_credit], axis=0)
    y_train = pd.concat([y_train_fraud, y_train_credit], axis=0)
    X_test = pd.concat([X_test_fraud, X_test_credit], axis=0)
    y_test = pd.concat([y_test_fraud, y_test_credit], axis=0)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save trained model
    model_path = os.path.join(args.output_dir, "fraud_detection_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
