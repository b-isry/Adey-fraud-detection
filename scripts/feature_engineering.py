import pandas as pd

def create_time_features(df, time_col='purchase_time'):
    """Creates time-based features from a datetime column."""
    df['hour_of_day'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek # Monday=0, Sunday=6
    return df

def time_since_signup(df):
    """Calculates the time difference between signup and purchase."""
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()
    return df

def transaction_frequency(df, window='1D'):
    """Calculates transaction frequency for each user."""
    # This requires setting user_id and purchase_time as index
    df_sorted = df.set_index('purchase_time').sort_index()
    # Count transactions per user in a given time window
    freq = df_sorted.groupby('user_id').rolling(window).count()['purchase_value'].rename('transaction_frequency')
    # Merge it back into the original dataframe
    df = df.merge(freq, on=['user_id', 'purchase_time'], how='left').fillna(0)
    return df