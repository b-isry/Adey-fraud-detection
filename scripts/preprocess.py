import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Loads data from a CSV file."""
    return pd.read_csv(file_path)

def clean_fraud_data(df):
    """Cleans the Fraud_Data.csv dataset."""
    # Drop missing values if any (though this dataset is clean)
    df.dropna(inplace=True)
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    # Convert time columns to datetime objects
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def clean_credit_card_data(df):
    """Cleans the creditcard.csv dataset."""
    # Drop missing values if any
    df.dropna(inplace=True)
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    return df

def convert_ip_to_int(df, ip_col='ip_address'):
    """Converts IP addresses to integer format for merging."""
    def safe_convert(ip):
        if isinstance(ip, str) and '.' in ip:
            return int(''.join([f'{int(part):03d}' for part in ip.split('.')]))
        else:
            return None  # or np.nan if you'd rather have a NaN

    df['ip_address_int'] = df[ip_col].apply(safe_convert).astype('Int64')  # 'Int64' allows nulls
    return df


def merge_with_country(fraud_df, ip_country_df):
    """Merges the fraud data with IP-to-country mapping."""
    # The IP mapping file can be quite large, so a simple merge might be slow.
    # A more optimized approach like a binary search or a database lookup would be better for production.
    # For this project, we can iterate and find the country for each IP.
    country_list = []
    for _, row in fraud_df.iterrows():
        ip_int = row['ip_address_int']
        country_match = ip_country_df[
            (ip_country_df['lower_bound_ip_address'] <= ip_int) &
            (ip_country_df['upper_bound_ip_address'] >= ip_int)
        ]
        if not country_match.empty:
            country_list.append(country_match.iloc[0]['country'])
        else:
            country_list.append('Unknown')
    fraud_df['country'] = country_list
    return fraud_df

def handle_imbalance(X_train, y_train):
    """Handles class imbalance using SMOTE."""
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def scale_and_encode(df, categorical_cols, numerical_cols):
    """Scales numerical features and one-hot encodes categorical features."""
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Scale numerical features
    scaled_features = scaler.fit_transform(df[numerical_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=numerical_cols, index=df.index)

    # Encode categorical features
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_cols, index=df.index)

    # Combine processed features
    processed_df = pd.concat([df_scaled, df_encoded, df.drop(columns=numerical_cols + categorical_cols)], axis=1)
    return processed_df