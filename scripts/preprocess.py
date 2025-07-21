import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from datetime import datetime

def load_data(fraud_path, ip_path):
    fraud_data = pd.read_csv(fraud_path)
    ip_data = pd.read_csv(ip_path)
    return fraud_data, ip_data

def handle_missing_values(df):
    # Check for missing values
    missing = df.isnull().sum()
    print("Missing values:\n", missing)
    # For simplicity, drop rows with missing values (if any)
    df = df.dropna()
    return df

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    # Correct data types
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def ip_to_int(ip):
    if isinstance(ip, str) and ip.count('.') == 3:
        parts = ip.split('.')
        try:
            return int(parts[0]) * 256**3 + int(parts[1]) * 256**2 + int(parts[2]) * 256 + int(parts[3])
        except ValueError:
            return None
    return None  # fallback for floats, malformed strings, or missing data


def merge_datasets(fraud_df, ip_df):
    # Convert IP addresses to integer
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    # Function to map IP to country
    def get_country(ip_int, ip_data):
        if ip_int is None:
            return 'Unknown'
        for _, row in ip_data.iterrows():
            if row['lower_bound_ip_address'] <= ip_int <= row['upper_bound_ip_address']:
                return row['country']
        return 'Unknown'

    # Apply mapping
    fraud_df['country'] = fraud_df['ip_int'].apply(lambda x: get_country(x, ip_df))
    return fraud_df.drop(columns=['ip_int'])

def feature_engineering(df):
    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600  # in hours
    # Transaction frequency and velocity
    df['transaction_count'] = df.groupby('user_id')['purchase_time'].transform('count')
    df['transaction_velocity'] = df['transaction_count'] / (df['time_since_signup'] + 1e-6)  # Avoid division by zero
    return df

def transform_data(df):
    # Encode categorical features
    categorical_cols = ['source', 'browser', 'sex', 'country']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]), 
                                columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_cols], axis=1)
    # Normalize/scale numerical features
    numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'transaction_count', 'transaction_velocity']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, encoder, scaler

def handle_class_imbalance(X, y):
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def preprocess_pipeline(fraud_path, ip_path):
    # Load data
    fraud_df, ip_df = load_data(fraud_path, ip_path)
    # Handle missing values
    fraud_df = handle_missing_values(fraud_df)
    # Clean data
    fraud_df = clean_data(fraud_df)
    # Merge datasets
    fraud_df = merge_datasets(fraud_df, ip_df)
    # Feature engineering
    fraud_df = feature_engineering(fraud_df)
    # Separate features and target
    X = fraud_df.drop(columns=['class', 'user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address'])
    y = fraud_df['class']
    # Transform data
    X_transformed, encoder, scaler = transform_data(X)
    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(X_transformed, y)
    return X_resampled, y_resampled, encoder, scaler, fraud_df

if __name__ == "__main__":
    fraud_path = "data/Fraud_Data.csv"
    ip_path = "data/IpAddress_to_Country.csv"
    X, y, encoder, scaler, original_df = preprocess_pipeline(fraud_path, ip_path)
    print("Preprocessing complete. Shape of resampled data:", X.shape, y.shape)