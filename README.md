Fraud Analysis Step-by-step Analysis

project file structure:

fraud_detection_project/
│── data/  
│   ├── raw/                    # Raw datasets (Fraud_Data.csv, creditcard.csv, etc.)
│   ├── processed/               # Preprocessed datasets (cleaned data, transformed features)
│   ├── models/                  # Saved trained models
│   ├── logs/                    # Log files for API requests and predictions
│
│── src/
│   ├── data_preprocessing.py    # Data cleaning, merging, and preprocessing
│   ├── feature_engineering.py   # Feature extraction and transformation
│   ├── model_training.py        # Train ML models, save best model
│   ├── model_evaluation.py      # Evaluate models and generate reports
│   ├── fraud_detection_api.py   # Flask API to serve models
│   ├── fraud_dashboard.py       # Dash dashboard for fraud visualization
│   ├── utils.py                 # Helper functions for logging, model loading
│
│── notebooks/                   # Jupyter notebooks for EDA & testing
│── config/                      # Config files (e.g., hyperparameters, settings)
│── requirements.txt              # Required libraries
│── Dockerfile                    # Docker setup for deployment
│── .gitignore                    # Ignore unnecessary files
│── README.md                     # Project documentation



Preprocessing with the following steps:

Load the datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv) and check for missing values.
Handle missing values by imputing or dropping them as necessary.
Clean the data by removing duplicates and correcting data types.
Perform Exploratory Data Analysis (EDA):
Univariate analysis (distribution of variables).
Bivariate analysis (correlation between features and fraud label).
Merge Datasets for Geolocation Analysis:
Convert ip_address to integer format.
Merge Fraud_Data.csv with IpAddress_to_Country.csv.
Feature Engineering:
Compute transaction frequency and velocity.
Extract time-based features (hour_of_day, day_of_week).
Normalization and Scaling:
Standardize numerical features.
Encode categorical features.
Begun implementing the data loading, missing values handling, and cleaning in Python.

import pandas as pd
import numpy as np
import argparse
import os

# Argument parser to allow dynamic file input
def get_args():
    parser = argparse.ArgumentParser(description="Preprocess Fraud Detection Datasets")
    parser.add_argument("--fraud_data", required=True, help="Path to Fraud_Data.csv")
    parser.add_argument("--ip_data", required=True, help="Path to IpAddress_to_Country.csv")
    parser.add_argument("--credit_data", required=True, help="Path to creditcard.csv")
    parser.add_argument("--output_dir", default="outputs/cleaned_data", help="Directory to save cleaned datasets")
    return parser.parse_args()

def main():
    args = get_args()

    # Load datasets
    fraud_data = pd.read_csv(args.fraud_data)
    ip_data = pd.read_csv(args.ip_data)
    credit_data = pd.read_csv(args.credit_data)

    # Check for missing values
    missing_values = {
        "Fraud_Data": fraud_data.isnull().sum(),
        "IpAddress_to_Country": ip_data.isnull().sum(),
        "Credit_Card": credit_data.isnull().sum()
    }
    print("Missing Values:", missing_values)

    # Handle missing values (drop rows with missing values)
    fraud_data.dropna(inplace=True)
    ip_data.dropna(inplace=True)
    credit_data.dropna(inplace=True)

    # Remove duplicates
    fraud_data.drop_duplicates(inplace=True)
    ip_data.drop_duplicates(inplace=True)
    credit_data.drop_duplicates(inplace=True)

    # Convert data types where necessary
    fraud_data["signup_time"] = pd.to_datetime(fraud_data["signup_time"], errors='coerce')
    fraud_data["purchase_time"] = pd.to_datetime(fraud_data["purchase_time"], errors='coerce')
    fraud_data["ip_address"] = fraud_data["ip_address"].astype(float).astype(int)  # Convert IP to integer

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save cleaned data
    fraud_data.to_csv(os.path.join(args.output_dir, "cleaned_Fraud_Data.csv"), index=False)
    ip_data.to_csv(os.path.join(args.output_dir, "cleaned_IpAddress_to_Country.csv"), index=False)
    credit_data.to_csv(os.path.join(args.output_dir, "cleaned_creditcard.csv"), index=False)

    print("Data preprocessing completed. Cleaned files saved in:", args.output_dir)

if __name__ == "__main__":
    main()


EDA Tasks
✅ Univariate Analysis

Distribution of numeric variables (histograms, boxplots)
Count plots for categorical variables
✅ Bivariate Analysis

Correlation heatmaps for numerical features
Fraud vs. non-fraud feature comparisons
Geolocation-based fraud analysis
✅ Save Outputs

Visualizations and summary statistics will be saved in outputs/eda/


    import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def load_data(file_path):
    """Load dataset from a given file path."""
    return pd.read_csv(file_path)

def save_plot(fig, filename, output_dir="outputs/eda/"):
    """Save a figure to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def univariate_analysis(df, output_dir):
    """Perform univariate analysis on numeric and categorical variables."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    
    # Histogram for numeric variables
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        save_plot(fig, f"hist_{col}.png", output_dir)
    
    # Count plot for categorical variables
    for col in categorical_cols:
        fig, ax = plt.subplots()
        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
        ax.set_title(f'Count of {col}')
        save_plot(fig, f"count_{col}.png", output_dir)

def bivariate_analysis(df, output_dir):
    """Perform bivariate analysis for fraud vs. non-fraud comparisons."""
    if 'class' in df.columns:
        fraud_df = df[df['class'] == 1]
        non_fraud_df = df[df['class'] == 0]
        
        # Compare numeric feature distributions
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.kdeplot(non_fraud_df[col], label='Non-Fraud', fill=True, alpha=0.5)
            sns.kdeplot(fraud_df[col], label='Fraud', fill=True, alpha=0.5)
            ax.set_title(f'Fraud vs. Non-Fraud: {col}')
            ax.legend()
            save_plot(fig, f"fraud_vs_nonfraud_{col}.png", output_dir)
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Perform EDA on fraud detection dataset.')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--output', type=str, default='outputs/eda/', help='Output directory for plots')
    args = parser.parse_args()
    
    df = load_data(args.data)
    
    univariate_analysis(df, args.output)
    bivariate_analysis(df, args.output)
    
    print(f"EDA completed. Plots saved in {args.output}")

if __name__ == "__main__":
    main()



![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)





Fraud Detection Modelling:

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
    fraud_data_path = "path/to/Fraud_Data.csv"
    creditcard_data_path = "path/to/creditcard.csv"
    main(fraud_data_path, creditcard_data_path)



Python script implementing SHAP and LIME for model explainability in the fraud detection project. 
It loads a trained model, computes feature importance, and visualizes explanations using SHAP and LIME


import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset (assuming preprocessed dataset is available)
data = pd.read_csv('processed_fraud_data.csv')  # Adjust path as needed

# Separate features and target
X = data.drop(columns=['class'])
y = data['class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load('trained_fraud_model.pkl')  # Adjust model path

# SHAP Explainability
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)
plt.savefig('outputs/shap_summary_plot.png')

# SHAP Force Plot (Example for first instance)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0].values, X_test.iloc[0])
plt.savefig('outputs/shap_force_plot.png')

# SHAP Dependence Plot
shap.dependence_plot('purchase_value', shap_values, X_test)
plt.savefig('outputs/shap_dependence_plot.png')

# LIME Explainability
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Not Fraud', 'Fraud'],
    mode='classification'
)

# Explain a single prediction
idx = 0  # Adjust index as needed
exp = explainer_lime.explain_instance(X_test.iloc[idx].values, model.predict_proba)
exp.save_to_file('outputs/lime_explanation.html')

print("SHAP and LIME explainability reports generated successfully.")


Flask API implementation for deploying the fraud detection model. The script includes API endpoints for predicting fraud, loading the trained model, and logging API requests. It also integrates logging for tracking requests and errors.


import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='fraud_api.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Load the trained model
MODEL_PATH = "models/fraud_detection_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    logging.info("Model loaded successfully.")
else:
    logging.error("Model file not found.")
    model = None

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("Empty input data")
        
        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_data)
        fraud_probability = model.predict_proba(input_data)[:, 1]
        
        response = {
            'prediction': int(prediction[0]),
            'fraud_probability': float(fraud_probability[0])
        }
        
        logging.info(f"Prediction made: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



Dashboard Creation

Planned Features for the Dashboard
Summary Statistics
Total transactions, fraud cases, and fraud percentage.
Fraud Cases Over Time
A line chart showing fraud trends over time.
Geographical Fraud Analysis
Map visualization of fraud occurrences based on geolocation.
Device & Browser-Based Fraud
A bar chart comparing fraud cases across different devices and browsers.

from flask import Flask, jsonify
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Initialize Flask app
app = Flask(__name__)

def load_fraud_data():
    """Load fraud data from CSV"""
    fraud_df = pd.read_csv("data/Fraud_Data.csv", parse_dates=["signup_time", "purchase_time"])
    return fraud_df

@app.route("/fraud-summary", methods=["GET"])
def fraud_summary():
    """API endpoint to return fraud statistics"""
    fraud_df = load_fraud_data()
    total_transactions = len(fraud_df)
    total_fraud = fraud_df[fraud_df["class"] == 1].shape[0]
    fraud_percentage = round((total_fraud / total_transactions) * 100, 2)
    
    summary = {
        "total_transactions": total_transactions,
        "total_fraud": total_fraud,
        "fraud_percentage": fraud_percentage
    }
    return jsonify(summary)

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, routes_pathname_prefix="/dashboard/")
fraud_df = load_fraud_data()

dash_app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    
    html.Div(id="summary-box"),
    
    dcc.Graph(id="fraud-trend-chart"),
    dcc.Graph(id="device-fraud-bar-chart"),
])

@dash_app.callback(
    Output("summary-box", "children"),
    Input("fraud-trend-chart", "figure")
)
def update_summary(_):
    summary = fraud_summary().json
    return html.Div([
        html.P(f"Total Transactions: {summary['total_transactions']}", style={'fontSize': 20}),
        html.P(f"Total Fraud Cases: {summary['total_fraud']}", style={'fontSize': 20}),
        html.P(f"Fraud Percentage: {summary['fraud_percentage']}%", style={'fontSize': 20})
    ])

@dash_app.callback(
    Output("fraud-trend-chart", "figure"),
    Input("summary-box", "children")
)
def update_trend(_):
    fraud_df["purchase_date"] = fraud_df["purchase_time"].dt.date
    fraud_trend = fraud_df.groupby("purchase_date")["class"].sum().reset_index()
    fig = px.line(fraud_trend, x="purchase_date", y="class", title="Fraud Cases Over Time")
    return fig

@dash_app.callback(
    Output("device-fraud-bar-chart", "figure"),
    Input("summary-box", "children")
)
def update_device_fraud(_):
    fraud_counts = fraud_df[fraud_df["class"] == 1]["device_id"].value_counts().nlargest(10).reset_index()
    fraud_counts.columns = ["device_id", "fraud_cases"]
    fig = px.bar(fraud_counts, x="device_id", y="fraud_cases", title="Top 10 Devices with Fraudulent Transactions")
    return fig

if __name__ == "__main__":
    app.run(debug=True)


The initial dashboard implementation includes:

A Flask API endpoint (/fraud-summary) to provide fraud statistics.
A Dash-based frontend with:
Fraud trend over time
Fraud cases by device
Summary statistics (total transactions, fraud cases, fraud percentage)


Steps to Implement Flask Functionality from Scratch
To integrate Flask from scratch for the fraud detection project, follow these steps:

1. Install Required Libraries
Ensure you have the necessary dependencies installed:

bash
Copy
Edit
pip install flask flask-cors pandas joblib scikit-learn dash plotly
2. Set Up the Flask API
Create a file called fraud_detection_api.py inside src/:

python
Copy
Edit
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
from utils import load_model, preprocess_input

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(filename="data/logs/api.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load trained fraud detection model
model = load_model("data/models/fraud_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to predict fraud probability.
    Expects JSON with transaction details.
    """
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        processed_df = preprocess_input(df)
        prediction = model.predict(processed_df)[0]
        result = {"fraud_prediction": int(prediction)}

        # Log request & response
        logging.info(f"Received: {data} | Prediction: {result}")
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({"error": "Invalid input"}), 400

@app.route('/fraud-summary', methods=['GET'])
def fraud_summary():
    """
    API endpoint to provide fraud statistics.
    """
    df = pd.read_csv("data/processed/fraud_summary.csv")
    summary = df.describe().to_dict()
    return jsonify(summary)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
3. Create a Utility File (utils.py)
To handle loading models, preprocessing data, and logging, create utils.py:

python
Copy
Edit
import joblib
import pandas as pd

def load_model(model_path):
    """Load the trained ML model."""
    return joblib.load(model_path)

def preprocess_input(df):
    """Process input data before making predictions."""
    df.fillna(0, inplace=True)  # Example preprocessing
    return df
4. Develop the Interactive Dashboard (Dash)
Create fraud_dashboard.py:

python
Copy
Edit
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load fraud data
df = pd.read_csv("data/processed/fraud_summary.csv")

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard"),
    html.Div([
        dcc.Graph(id='fraud-trend'),
        dcc.Graph(id='device-fraud'),
    ])
])

@app.callback(
    Output('fraud-trend', 'figure'),
    Input('fraud-trend', 'id')
)
def update_trend(_):
    """Update fraud trend graph."""
    fraud_trend = df.groupby("purchase_time")["class"].sum().reset_index()
    fig = px.line(fraud_trend, x="purchase_time", y="class", title="Fraud Cases Over Time")
    return fig

@app.callback(
    Output('device-fraud', 'figure'),
    Input('device-fraud', 'id')
)
def update_device_chart(_):
    """Update fraud by device chart."""
    fraud_by_device = df.groupby("device_id")["class"].sum().reset_index()
    fig = px.bar(fraud_by_device, x="device_id", y="class", title="Fraud Cases by Device")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
5. Dockerize the Flask App
Create a Dockerfile to containerize the API:

Dockerfile
Copy
Edit
# Use an official Python image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose API port
EXPOSE 5000

# Run the Flask API
CMD ["python", "src/fraud_detection_api.py"]
6. Run and Test Everything
Run Flask API


python src/fraud_detection_api.py
Run the Dashboard

python src/fraud_dashboard.py
Test API with Curl or Postman

curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"purchase_value": 200, "device_id": "XYZ", "browser": "Chrome"}'
Build & Run Docker

docker build -t fraud-detection .
docker run -p 5000:5000 fraud-detection