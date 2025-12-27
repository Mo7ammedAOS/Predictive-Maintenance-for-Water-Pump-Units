"""
Data Processing Utilities for Predictive Maintenance
Handles complete data pipeline from loading to feature engineering
FIXED VERSION - Corrects train/test split empty dataset bug
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple, List, Dict
import streamlit as st


@st.cache_data
def load_sensor_data(file_path: str = None, uploaded_file=None) -> pd.DataFrame:
    """
    Load sensor data from CSV file or uploaded file
    
    Args:
        file_path: Path to CSV file
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        DataFrame with sensor readings
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif file_path is not None:
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Either file_path or uploaded_file must be provided")
    
    # Drop unnamed index column if exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Drop any other unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df


def preprocess_data(df: pd.DataFrame, 
                    drop_sensors: List[str] = ['sensor_15']) -> pd.DataFrame:
    """
    Preprocess sensor data: drop problematic sensors and handle missing values
    
    Args:
        df: Raw sensor DataFrame
        drop_sensors: List of sensor columns to drop
    
    Returns:
        Preprocessed DataFrame
    """
    df_clean = df.copy()
    
    # Drop status columns if present (will be recreated)
    if 'machine_status_updated' in df_clean.columns:
        df_clean = df_clean.drop(columns=['machine_status_updated'])
    
    # Drop specified problematic sensors
    for sensor in drop_sensors:
        if sensor in df_clean.columns:
            df_clean = df_clean.drop(columns=[sensor])
    
    # Fill missing values with -1 (indicator value for missing)
    sensor_cols = [col for col in df_clean.columns 
                   if col.startswith('sensor_')]
    
    for col in sensor_cols:
        df_clean[col].fillna(-1, inplace=True)
    
    return df_clean


def create_labels(df: pd.DataFrame, 
                  status_col: str = 'machine_status') -> pd.DataFrame:
    """
    Create binary labels from machine status
    
    Args:
        df: DataFrame with machine_status column
        status_col: Name of the status column
    
    Returns:
        DataFrame with added 'labels' column (1=NORMAL, 0=BROKEN)
    """
    df = df.copy()
    df['labels'] = df[status_col].map(lambda x: 1 if x == 'NORMAL' else 0)
    
    return df


def shift_labels(df: pd.DataFrame, 
                 sensor_cols: List[str], 
                 shift_steps: int = 10) -> pd.DataFrame:
    """
    Shift labels forward by 10 minutes for predictive modeling
    
    FIXED VERSION: Properly handles sensor columns and labels
    
    Args:
        df: DataFrame with sensor data and labels
        sensor_cols: List of sensor column names (should NOT include 'labels')
        shift_steps: Number of time steps to shift (default: 10 minutes)
    
    Returns:
        DataFrame with shifted labels and sensor data
    """
    # Create new dataframe with only sensor columns
    new_df = df[sensor_cols].copy()
    
    # Add shifted labels (shift forward = negative shift to predict future)
    new_df['labels'] = df['labels'].shift(-shift_steps)
    
    # Drop rows with NaN (last 'shift_steps' rows will have NaN labels)
    new_df = new_df.dropna()
    
    return new_df


def generate_deviation_features(df: pd.DataFrame, 
                                sensor_cols: List[str]) -> pd.DataFrame:
    """
    Generate deviation features: difference from normal state mean
    
    Args:
        df: DataFrame with sensor data and labels
        sensor_cols: List of sensor column names
    
    Returns:
        DataFrame with deviation features
    """
    new_features = {}
    
    # Calculate mean values for normal state (label = 1)
    normal_df = df[df['labels'] == 1]
    
    for sensor in sensor_cols:
        if sensor != 'labels' and sensor in df.columns:
            normal_mean = normal_df[sensor].mean()
            new_features[f'{sensor}_deviation'] = df[sensor] - normal_mean
    
    new_features['labels'] = df['labels']
    
    return pd.DataFrame(new_features)


def generate_window_features(df: pd.DataFrame, 
                             sensor_cols: List[str],
                             window_size: int = 10) -> pd.DataFrame:
    """
    Generate time window features using shifted values (10-min window aggregation)
    
    Args:
        df: DataFrame with sensor data and labels
        sensor_cols: List of sensor column names
        window_size: Size of window
    
    Returns:
        DataFrame with window features
    """
    new_df = df[sensor_cols].copy()
    
    for col in sensor_cols:
        if col != 'labels' and col in df.columns:
            new_df[col] = df[col].shift(window_size)
    
    new_df['labels'] = df['labels']
    
    return new_df.iloc[window_size:].reset_index(drop=True)


def normalize_features(X_train: pd.DataFrame, 
                      X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Normalize features using MinMaxScaler (0-1 range)
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        Tuple of (normalized_train, normalized_test, scaler)
    """
    scaler = MinMaxScaler()
    
    train_scaled = scaler.fit_transform(X_train.values)
    test_scaled = scaler.transform(X_test.values)
    
    train_df = pd.DataFrame(
        train_scaled, 
        columns=X_train.columns, 
        index=X_train.index
    )
    
    test_df = pd.DataFrame(
        test_scaled, 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    return train_df, test_df, scaler


def split_train_test(df: pd.DataFrame, 
                     train_size: int = 50000) -> Tuple:
    """
    Split data into train and test sets using time-based split
    
    FIXED VERSION: Handles small datasets gracefully
    
    Args:
        df: DataFrame with features and labels
        train_size: Number of samples for training
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Ensure train_size doesn't exceed dataset size
    total_samples = len(df)
    
    if total_samples < 100:
        raise ValueError(f"Dataset too small ({total_samples} samples). Need at least 100 samples.")
    
    # Adjust train_size if needed (use 80% for training if specified size is too large)
    if train_size >= total_samples:
        train_size = int(total_samples * 0.8)
    
    # Ensure we have at least some test data
    if total_samples - train_size < 10:
        train_size = total_samples - max(10, int(total_samples * 0.2))
    
    y = df['labels']
    X = df.drop(columns=['labels'])
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    return X_train, X_test, y_train, y_test


def get_class_distribution(df: pd.DataFrame, 
                          col: str = 'labels') -> Dict:
    """
    Calculate class distribution statistics
    
    Args:
        df: DataFrame or Series
        col: Column name to analyze (if df is DataFrame)
    
    Returns:
        Dictionary with class counts and percentages
    """
    if isinstance(df, pd.Series):
        value_counts = df.value_counts()
        total = len(df)
    else:
        value_counts = df[col].value_counts()
        total = len(df)
    
    distribution = {}
    for label, count in value_counts.items():
        distribution[label] = {
            'count': int(count),
            'percentage': round(count / total * 100, 2)
        }
    
    return distribution


def get_missing_value_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing value statistics for all columns
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        DataFrame with missing value counts and percentages
    """
    null_count = df.isna().sum()
    null_pct = (null_count / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Column': null_count.index,
        'Missing_Count': null_count.values,
        'Missing_Percentage': null_pct.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    return missing_df


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive dataset statistics
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with various statistics
    """
    stats = {
        'shape': df.shape,
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return stats


@st.cache_data
def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate sample sensor data for demonstration
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with synthetic sensor readings
    """
    np.random.seed(42)
    
    data = {}
    
    # Generate 58 sensor readings (after removing sensor_15)
    for i in range(59):
        if i == 15:
            continue  # Skip sensor_15
        sensor_name = f'sensor_{i:02d}'
        # Normal readings: mean around 0.5, std 0.1
        data[sensor_name] = np.random.normal(0.5, 0.1, n_samples)
    
    # Add machine status (mostly normal to match real distribution)
    statuses = np.random.choice(
        ['NORMAL', 'BROKEN'], 
        size=n_samples, 
        p=[0.97, 0.03]
    )
    data['machine_status'] = statuses
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values realistically
    for col in df.columns[:10]:
        missing_idx = np.random.choice(
            df.index, 
            size=int(len(df) * 0.02), 
            replace=False
        )
        df.loc[missing_idx, col] = np.nan
    
    return df


def prepare_complete_pipeline(uploaded_file=None, 
                             file_path: str = None,
                             feature_type: str = 'deviation') -> Dict:
    """
    Complete data preparation pipeline from raw data to normalized features
    
    FIXED VERSION: Properly handles the entire pipeline sequence
    """
    # Step 1: Load data
    if uploaded_file is not None or file_path is not None:
        df_raw = load_sensor_data(file_path=file_path, uploaded_file=uploaded_file)
    else:
        df_raw = generate_sample_data(n_samples=5000)
    
    # Step 2: Preprocess (drop sensor_15, fill missing values)
    df_clean = preprocess_data(df_raw)
    
    # Step 3: Create labels (1=NORMAL, 0=BROKEN)
    df_labels = create_labels(df_clean)
    
    # Step 4: Get sensor columns (EXCLUDING labels and machine_status)
    sensor_cols = [col for col in df_labels.columns 
                   if col.startswith('sensor_')]
    
    # CRITICAL FIX: sensor_cols should NOT include 'labels'
    # The shift_labels function will add labels separately
    
    # Step 5: Shift labels for 10-minute advance warning
    df_shifted = shift_labels(df_labels, sensor_cols, shift_steps=10)
    
    # CHECK: Ensure we have data after shifting
    if len(df_shifted) == 0:
        raise ValueError("No data remaining after label shifting. Your dataset may be too small (need >10 rows)")
    
    # Step 6: Generate features based on type
    if feature_type == 'deviation':
        df_features = generate_deviation_features(df_shifted, sensor_cols)
    else:  # window
        df_features = generate_window_features(df_shifted, sensor_cols, window_size=10)
    
    # CHECK: Ensure features were created
    if len(df_features) == 0:
        raise ValueError("No features generated. Check your data quality.")
    
    # Step 7: Split train/test (use adaptive split size)
    train_size = min(50000, int(len(df_features) * 0.8))
    X_train, X_test, y_train, y_test = split_train_test(df_features, train_size=train_size)
    
    # CHECK: Ensure train/test splits have data
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"Train or test set is empty. Dataset size: {len(df_features)}, Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Step 8: Normalize features
    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)
    
    # Step 9: Get statistics
    class_dist_train = get_class_distribution(y_train)
    class_dist_test = get_class_distribution(y_test)
    
    return {
        'raw': df_raw,
        'clean': df_clean,
        'labels': df_labels,
        'shifted': df_shifted,
        'features': df_features,
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'sensor_cols': sensor_cols,
        'feature_type': feature_type,
        'class_dist_train': class_dist_train,
        'class_dist_test': class_dist_test,
        'stats': {
            'total_samples': len(df_features),
            'total_features': len(sensor_cols),
            'train_samples': len(X_train_norm),
            'test_samples': len(X_test_norm),
        }
    }