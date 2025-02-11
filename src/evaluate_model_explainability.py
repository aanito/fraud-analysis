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
