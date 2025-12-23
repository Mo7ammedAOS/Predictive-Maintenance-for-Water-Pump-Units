"""
Data Processing Utilities for Predictive Maintenance
Handles data loading, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
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
    
    # Drop unnamed columns if they exist
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df


def preprocess_data(df: pd.DataFrame, 
                    drop_sensors: List[str] = ['sensor_15']) -> pd.DataFrame:
    """
    Preprocess sensor data: handle missing values, drop problematic sensors
    
    Args:
        df: Raw sensor DataFrame
        drop_sensors: List of sensor columns to drop
    
    Returns:
        Preprocessed DataFrame
    """
    df_clean = df.copy()
    
    # Drop specified sensors
    for sensor in drop_sensors:
        if sensor in df_clean.columns:
            df_clean.drop(columns=[sensor], inplace=True)
    
    # Fill missing values with -1 (indicator value)
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
    df['machine_status_updated'] = df[status_col].map(
        lambda x: 'NORMAL' if x == 'NORMAL' else 'BROKEN'
    )
    return df


def shift_labels(df: pd.DataFrame, 
                 sensor_cols: List[str], 
                 shift_steps: int = 10) -> pd.DataFrame:
    """
    Shift labels forward for predictive modeling
    
    Args:
        df: DataFrame with sensor data and labels
        sensor_cols: List of sensor column names
        shift_steps: Number of time steps to shift (default: 10 minutes)
    
    Returns:
        DataFrame with shifted labels
    """
    new_df = df[sensor_cols].copy()
    new_df['labels'] = df['labels'].shift(-shift_steps)
    return new_df.dropna()


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
        if sensor != 'labels':
            normal_mean = normal_df[sensor].mean()
            new_features[f'{sensor}_deviation'] = df[sensor] - normal_mean
    
    new_features['labels'] = df['labels']
    
    return pd.DataFrame(new_features)


def generate_window_features(df: pd.DataFrame, 
                             sensor_cols: List[str],
                             window_size: int = 10) -> pd.DataFrame:
    """
    Generate time window features using rolling mean
    
    Args:
        df: DataFrame with sensor data and labels
        sensor_cols: List of sensor column names
        window_size: Size of rolling window
    
    Returns:
        DataFrame with window features
    """
    new_df = df[sensor_cols].copy()
    
    for col in sensor_cols:
        if col != 'labels':
            new_df[col] = df[col].shift(window_size)
    
    new_df['labels'] = df['labels']
    
    return new_df.iloc[window_size:].reset_index(drop=True)


def normalize_features(X_train: pd.DataFrame, 
                      X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize features using MinMaxScaler
    
    Args:
        X_train: Training features
        X_test: Test features
    
    Returns:
        Tuple of (normalized_train, normalized_test)
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
                     train_size: int = 131000) -> Tuple:
    """
    Split data into train and test sets
    
    Args:
        df: DataFrame with features and labels
        train_size: Number of samples for training
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    y = df['labels']
    X = df.drop(columns=['labels'])
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    return X_train, X_test, y_train, y_test


def get_class_distribution(df: pd.DataFrame, 
                          col: str = 'labels') -> dict:
    """
    Calculate class distribution statistics
    
    Args:
        df: DataFrame
        col: Column name to analyze
    
    Returns:
        Dictionary with class counts and percentages
    """
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


@st.cache_data
def generate_sample_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate sample sensor data for demonstration
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with synthetic sensor readings
    """
    np.random.seed(42)
    
    data = {}
    
    # Generate 58 sensor readings
    for i in range(58):
        # Normal readings: mean around 0.5, std 0.1
        # Broken readings: mean around 0.3 or 0.7, std 0.15
        sensor_name = f'sensor_{i:02d}'
        data[sensor_name] = np.random.normal(0.5, 0.1, n_samples)
    
    # Add machine status (mostly normal)
    statuses = np.random.choice(
        ['NORMAL', 'BROKEN'], 
        size=n_samples, 
        p=[0.97, 0.03]
    )
    data['machine_status'] = statuses
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    for col in df.columns[:10]:
        missing_idx = np.random.choice(
            df.index, 
            size=int(len(df) * 0.05), 
            replace=False
        )
        df.loc[missing_idx, col] = np.nan
    
    return df


def prepare_for_prediction(sensor_readings: dict, 
                          scaler: MinMaxScaler = None) -> pd.DataFrame:
    """
    Prepare sensor readings for model prediction
    
    Args:
        sensor_readings: Dictionary of sensor values
        scaler: Fitted MinMaxScaler object
    
    Returns:
        DataFrame ready for prediction
    """
    df = pd.DataFrame([sensor_readings])
    
    if scaler is not None:
        df_scaled = scaler.transform(df)
        df = pd.DataFrame(df_scaled, columns=df.columns)
    
    return df


def calculate_statistics(df: pd.DataFrame) -> dict:
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
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'missing_values': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return stats