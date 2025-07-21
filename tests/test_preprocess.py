import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../scripts'))

from scripts.preprocess import load_data, handle_missing_values, clean_data, merge_datasets, feature_engineering
import pytest
import os

# Mock data for testing
@pytest.fixture
def mock_fraud_data():
    return pd.DataFrame({
        'user_id': [1, 2, 3],
        'signup_time': ['2023-01-01 10:00:00', '2023-01-02 12:00:00', '2023-01-03 14:00:00'],
        'purchase_time': ['2023-01-01 12:00:00', '2023-01-02 13:00:00', '2023-01-03 15:00:00'],
        'purchase_value': [50, 100, 75],
        'device_id': ['device1', 'device2', 'device3'],
        'source': ['SEO', 'Ads', 'Direct'],
        'browser': ['Chrome', 'Safari', 'Firefox'],
        'sex': ['M', 'F', 'M'],
        'age': [25, 30, 35],
        'ip_address': ['192.168.1.1', '192.168.1.2', '192.168.1.3'],
        'class': [0, 1, 0]
    })

@pytest.fixture
def mock_ip_data():
    return pd.DataFrame({
        'lower_bound_ip_address': [3232235776, 3232235777],
        'upper_bound_ip_address': [3232235776, 3232235778],
        'country': ['USA', 'Canada']
    })

def test_load_data(mock_fraud_data, mock_ip_data, tmp_path):
    # Save mock data to temporary files
    fraud_path = tmp_path / "Fraud_Data.csv"
    ip_path = tmp_path / "IpAddress_to_Country.csv"
    mock_fraud_data.to_csv(fraud_path, index=False)
    mock_ip_data.to_csv(ip_path, index=False)
    
    fraud_df, ip_df = load_data(str(fraud_path), str(ip_path))
    assert not fraud_df.empty
    assert not ip_df.empty
    assert list(fraud_df.columns) == list(mock_fraud_data.columns)
    assert list(ip_df.columns) == list(mock_ip_data.columns)

def test_handle_missing_values(mock_fraud_data):
    # Test with no missing values
    df = handle_missing_values(mock_fraud_data.copy())
    assert df.shape == mock_fraud_data.shape
    # Test with missing values
    df_with_missing = mock_fraud_data.copy()
    df_with_missing.loc[0, 'purchase_value'] = np.nan
    df_cleaned = handle_missing_values(df_with_missing)
    assert df_cleaned.shape[0] == mock_fraud_data.shape[0] - 1

def test_clean_data(mock_fraud_data):
    # Test duplicate removal
    df_with_duplicates = pd.concat([mock_fraud_data, mock_fraud_data.iloc[[0]]], ignore_index=True)
    df_cleaned = clean_data(df_with_duplicates)
    assert df_cleaned.shape[0] == mock_fraud_data.shape[0]
    # Test data type conversion
    assert pd.api.types.is_datetime64_any_dtype(df_cleaned['signup_time'])
    assert pd.api.types.is_datetime64_any_dtype(df_cleaned['purchase_time'])

def test_ip_to_int():
    assert ip_to_int('192.168.1.1') == 3232235777
    assert ip_to_int('10.0.0.1') == 167772161

def test_merge_datasets(mock_fraud_data, mock_ip_data):
    df = mock_fraud_data.copy()
    df = merge_datasets(df, mock_ip_data)
    assert 'country' in df.columns
    assert df.loc[0, 'country'] == 'USA'
    assert df.loc[1, 'country'] == 'Canada'
    assert df.loc[2, 'country'] == 'Unknown'

def test_feature_engineering(mock_fraud_data):
    df = feature_engineering(mock_fraud_data.copy())
    assert 'hour_of_day' in df.columns
    assert 'day_of_week' in df.columns
    assert 'time_since_signup' in df.columns
    assert 'transaction_count' in df.columns
    assert 'transaction_velocity' in df.columns
    assert df['hour_of_day'].iloc[0] == 12
    assert df['time_since_signup'].iloc[0] == 2.0  
    assert df['transaction_count'].iloc[0] == 1