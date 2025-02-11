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
