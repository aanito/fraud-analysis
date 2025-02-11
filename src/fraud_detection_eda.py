import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load datasets
def load_data(fraud_path, ip_path, credit_path):
    fraud_data = pd.read_csv(fraud_path)
    ip_data = pd.read_csv(ip_path)
    credit_data = pd.read_csv(credit_path)
    return fraud_data, ip_data, credit_data

# Exploratory Data Analysis (EDA)
def perform_eda(data, title):
    print(f"Summary Statistics for {title}")
    print(data.describe())
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data['class'], bins=2, kde=True)
    plt.title(f'{title} - Fraud Class Distribution')
    plt.show()

# Merge geolocation data
def merge_geolocation(fraud_data, ip_data):
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)
    
    fraud_data = fraud_data.merge(ip_data, how='left', left_on='ip_address', right_on='lower_bound_ip_address')
    fraud_data.drop(columns=['lower_bound_ip_address', 'upper_bound_ip_address'], inplace=True)
    return fraud_data

# Feature Engineering
def create_features(fraud_data):
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_data['transaction_time_diff'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds()
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    return fraud_data

# Normalize & Encode Categorical Features
def normalize_and_encode(data):
    scaler = StandardScaler()
    data[['purchase_value', 'transaction_time_diff']] = scaler.fit_transform(data[['purchase_value', 'transaction_time_diff']])
    
    label_enc = LabelEncoder()
    data['browser'] = label_enc.fit_transform(data['browser'])
    data['source'] = label_enc.fit_transform(data['source'])
    data['sex'] = label_enc.fit_transform(data['sex'])
    return data

# Main execution
fraud_path = "outputs/cleaned_data/cleaned_Fraud_Data.csv"
ip_path = "outputs/cleaned_data/cleaned_IpAddress_to_Country.csv"
credit_path = "outputs/cleaned_data/cleaned_creditcard.csv"
fraud_data, ip_data, credit_data = load_data(fraud_path, ip_path, credit_path)

perform_eda(fraud_data, "Fraud Data")
fraud_data = merge_geolocation(fraud_data, ip_data)
fraud_data = create_features(fraud_data)
fraud_data = normalize_and_encode(fraud_data)

print("Data preprocessing and feature engineering complete.")
