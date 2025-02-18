import pandas as pd
import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Feature Engineering for Fraud Detection")
    parser.add_argument("--fraud_data", required=True, help="Path to cleaned_Fraud_Data.csv")
    parser.add_argument("--ip_data", required=True, help="Path to cleaned_IpAddress_to_Country.csv")
    parser.add_argument("--output_dir", default="outputs/engineered_data", help="Directory to save processed dataset")
    return parser.parse_args()

def map_ip_to_country(ip, ip_df):
    """ Map IP address to country by checking range """
    matched_row = ip_df[(ip_df["lower_bound_ip_address"] <= ip) & (ip_df["upper_bound_ip_address"] >= ip)]
    return matched_row["country"].values[0] if not matched_row.empty else "Unknown"

def main():
    args = get_args()

    # Load datasets
    fraud_data = pd.read_csv(args.fraud_data)
    ip_data = pd.read_csv(args.ip_data)

    # Convert timestamps to datetime
    fraud_data["signup_time"] = pd.to_datetime(fraud_data["signup_time"], errors="coerce")
    fraud_data["purchase_time"] = pd.to_datetime(fraud_data["purchase_time"], errors="coerce")

    # Feature: Time difference between signup and purchase
    fraud_data["time_to_purchase"] = (fraud_data["purchase_time"] - fraud_data["signup_time"]).dt.total_seconds()

    # Convert IP addresses to integer format
    fraud_data["ip_address"] = fraud_data["ip_address"].astype(float).astype(int)
    ip_data["lower_bound_ip_address"] = ip_data["lower_bound_ip_address"].astype(int)
    ip_data["upper_bound_ip_address"] = ip_data["upper_bound_ip_address"].astype(int)

    # Apply IP mapping
    fraud_data["country"] = fraud_data["ip_address"].apply(lambda x: map_ip_to_country(x, ip_data))

    # Encode categorical variables
    fraud_data = pd.get_dummies(fraud_data, columns=["source", "browser", "sex"], drop_first=True)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save processed dataset
    fraud_data.to_csv(os.path.join(args.output_dir, "engineered_Fraud_Data.csv"), index=False)

    print("Feature Engineering completed. Processed file saved in:", args.output_dir)

if __name__ == "__main__":
    main()
