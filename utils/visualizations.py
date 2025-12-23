"""
Visualization Utilities for Predictive Maintenance Dashboard
Provides interactive and static plotting functions
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
import streamlit as st


def plot_machine_status_timeline(labels: pd.Series) -> go.Figure:
    """
    Create interactive timeline plot of machine status
    
    Args:
        labels: Series with binary labels (0=BROKEN, 1=NORMAL)
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=labels.index,
        y=labels.values,
        mode='lines',
        name='Machine Status',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title='Machine Status Over Time',
        xaxis_title='Time Step',
        yaxis_title='Status',
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['BROKEN', 'NORMAL']
        ),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_class_distribution(labels: pd.Series, col: str = None) -> go.Figure:
    """
    Create bar chart for class distribution
    
    Args:
        labels: Series with labels
        col: Column name (optional)
    
    Returns:
        Plotly figure object
    """
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    
    value_counts = labels.value_counts().sort_index()
    
    colors = ['#d62728' if label == 0 else '#2ca02c' 
              for label in value_counts.index]
    
    labels_text = ['BROKEN' if x == 0 else 'NORMAL' for x in value_counts.index]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels_text,
            y=value_counts.values,
            marker_color=colors,
            text=value_counts.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title='Class Distribution',
        xaxis_title='Machine Status',
        yaxis_title='Count',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_missing_values(df: pd.DataFrame, 
                       top_n: int = 20) -> go.Figure:
    """
    Create horizontal bar chart of missing values
    
    Args:
        df: DataFrame to analyze
        top_n: Number of top columns to show
    
    Returns:
        Plotly figure object
    """
    null_count = df.isna().sum()
    null_count = null_count[null_count > 0].sort_values(ascending=True).tail(top_n)
    
    if len(null_count) == 0:
        # Return empty figure if no missing values
        fig = go.Figure()
        fig.add_annotation(text="No missing values detected", showarrow=False)
        return fig
    
    fig = go.Figure(data=[
        go.Bar(
            y=null_count.index,
            x=null_count.values,
            orientation='h',
            marker_color='#ff7f0e',
            text=null_count.values,
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Columns with Missing Values',
        xaxis_title='Missing Value Count',
        yaxis_title='Column',
        template='plotly_white',
        height=600
    )
    
    return fig


def plot_sensor_distribution(df: pd.DataFrame, 
                            sensor: str) -> go.Figure:
    """
    Create distribution plot comparing normal vs broken states
    
    Args:
        df: DataFrame with sensor data and labels
        sensor: Sensor column name
    
    Returns:
        Plotly figure object
    """
    normal = df[df['labels'] == 1][sensor].dropna()
    broken = df[df['labels'] == 0][sensor].dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=normal,
        name='Normal',
        opacity=0.7,
        marker_color='#2ca02c',
        nbinsx=50
    ))
    
    fig.add_trace(go.Histogram(
        x=broken,
        name='Broken',
        opacity=0.7,
        marker_color='#d62728',
        nbinsx=50
    ))
    
    fig.update_layout(
        title=f'{sensor} Distribution: Normal vs Broken',
        xaxis_title=f'{sensor} Reading',
        yaxis_title='Frequency',
        barmode='overlay',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray) -> go.Figure:
    """
    Create confusion matrix heatmap
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Plotly figure object
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Broken', 'Normal']
    
    # Calculate percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            annotations.append(
                f"{cm[i, j]}<br>({cm_pct[i, j]:.1f}%)"
            )
    
    annotations = np.array(annotations).reshape(2, 2)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Greens',
        text=annotations,
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        template='plotly_white',
        height=400,
        width=450
    )
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, 
                           top_n: int = 15,
                           title: str = 'Feature Importance') -> go.Figure:
    """
    Create horizontal bar chart for feature importance
    
    Args:
        importance_df: DataFrame with 'Feature' and 'Importance' columns
        top_n: Number of top features to show
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    # Ensure we have the right columns
    if 'Importance' in importance_df.columns:
        importance_col = 'Importance'
    elif 'SHAP_Importance' in importance_df.columns:
        importance_col = 'SHAP_Importance'
    else:
        importance_col = importance_df.columns[1]
    
    top_features = importance_df.head(top_n).sort_values(importance_col)
    
    fig = go.Figure(data=[
        go.Bar(
            y=top_features['Feature'],
            x=top_features[importance_col],
            orientation='h',
            marker_color='#1f77b4',
            text=top_features[importance_col].round(4),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_model_comparison(models_df: pd.DataFrame) -> go.Figure:
    """
    Create grouped bar chart comparing model performance
    
    Args:
        models_df: DataFrame with model names and metrics
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Accuracy',
        x=models_df['Model'],
        y=models_df['Accuracy'],
        text=models_df['Accuracy'].round(4),
        textposition='outside',
        marker_color='#2ca02c'
    ))
    
    fig.add_trace(go.Bar(
        name='F1 Macro',
        x=models_df['Model'],
        y=models_df['F1 Macro'],
        text=models_df['F1 Macro'].round(4),
        textposition='outside',
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis_range=[0.95, 1.0],
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_probability_gauge(probability: float, 
                          threshold: float = 0.5) -> go.Figure:
    """
    Create gauge chart for failure probability
    
    Args:
        probability: Predicted probability of failure
        threshold: Decision threshold
    
    Returns:
        Plotly figure object
    """
    # Determine status
    if probability < 0.3:
        color = "#2ca02c"  # Green
        status = "Normal"
    elif probability < 0.7:
        color = "#ff7f0e"  # Orange
        status = "Warning"
    else:
        color = "#d62728"  # Red
        status = "Critical"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Failure Risk: {status}"},
        delta={'reference': threshold * 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},
                {'range': [30, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def plot_sensor_time_series(df: pd.DataFrame, 
                           sensors: List[str],
                           max_points: int = 5000) -> go.Figure:
    """
    Create multi-line time series plot for sensors
    
    Args:
        df: DataFrame with sensor data
        sensors: List of sensor names to plot
        max_points: Maximum number of points to plot
    
    Returns:
        Plotly figure object
    """
    # Subsample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        df_plot = df.iloc[::step]
    else:
        df_plot = df
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for i, sensor in enumerate(sensors):
        if sensor in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot[sensor],
                mode='lines',
                name=sensor,
                line=dict(color=colors[i % len(colors)])
            ))
    
    fig.update_layout(
        title='Sensor Readings Over Time',
        xaxis_title='Time Step',
        yaxis_title='Sensor Reading',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig