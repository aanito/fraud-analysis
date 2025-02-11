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
