import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten

# Load datasets
def load_data(fraud_data_path, creditcard_data_path):
    fraud_data = pd.read_csv(fraud_data_path)
    creditcard_data = pd.read_csv(creditcard_data_path)
    return fraud_data, creditcard_data

# Preprocessing function
def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training function
def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

# MLflow experiment logging
def log_experiment(model_name, metrics):
    with mlflow.start_run():
        mlflow.log_params({"model": model_name})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model_name, "model")

# Main execution
def main(fraud_data_path, creditcard_data_path):
    fraud_data, creditcard_data = load_data(fraud_data_path, creditcard_data_path)
    
    # Preprocess datasets
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = preprocess_data(fraud_data, 'class')
    X_train_cc, X_test_cc, y_train_cc, y_test_cc = preprocess_data(creditcard_data, 'Class')
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "MLP": MLPClassifier()
    }
    
    for model_name, model in models.items():
        metrics = train_model(model, X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud)
        log_experiment(model_name, metrics)
        print(f"{model_name} Metrics: {metrics}")

if __name__ == "__main__":
    fraud_data_path = "outputs/cleaned_data/cleaned_Fraud_Data.csv"
    creditcard_data_path = "outputs/cleaned_data/cleaned_creditcard.csv"
    main(fraud_data_path, creditcard_data_path)
