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
