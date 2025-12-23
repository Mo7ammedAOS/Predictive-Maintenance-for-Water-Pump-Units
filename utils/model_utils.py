"""
Model Training and Evaluation Utilities
Handles ML model training with Random Forest only
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support
)
from typing import Dict, Tuple, Any
import joblib
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


class ModelTrainer:
    """Class to handle Random Forest model training and evaluation"""
    
    def __init__(self, random_state: int = 21):
        self.random_state = random_state
        self.model = None
        self.results = {}
        self.training_history = []
    
    def train_random_forest(self, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series) -> Dict:
        """Train Random Forest model with optimized parameters"""
        
        # Use exact parameters from notebook
        self.model = RandomForestClassifier(
            criterion='gini',
            random_state=self.random_state,
            n_jobs=-1,
            n_estimators=150,
            max_depth=5,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        self.training_history.append(('Random Forest', 'Training Complete'))
        
        return {
            'model': self.model,
            'status': 'Successfully trained'
        }
    
    def evaluate_model(self, 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict:
        """Evaluate Random Forest model on test data"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate comprehensive metrics
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Get precision and recall for each class
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
        
        # Count misclassifications for broken state (class 0)
        misclassifications_total = (y_test != y_pred).sum()
        broken_false_negatives = cm[0, 1]  # Broken predicted as Normal
        broken_false_positives = cm[1, 0]  # Normal predicted as Broken
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        self.results = {
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'misclassifications': int(misclassifications_total),
            'broken_false_negatives': int(broken_false_negatives),
            'broken_false_positives': int(broken_false_positives),
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1
        }
        
        return self.results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {}
        
        return {
            'name': 'Random Forest',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'criterion': self.model.criterion,
            'status': 'Trained'
        }
    
    def get_feature_importance(self, 
                              feature_names: list) -> pd.DataFrame:
        """Get feature importance from Random Forest"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        
        self.model = joblib.load(filepath)


def predict_with_confidence(model: Any, 
                           X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores
    
    Args:
        model: Trained Random Forest model
        X: Features
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    # Validate input
    if len(X) == 0:
        raise ValueError("Input data is empty. Please check your CSV file.")
    
    if X.isnull().any().any():
        X = X.fillna(0.5)
    
    try:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")
    
    return predictions, probabilities


def get_maintenance_recommendation(probability: float,
                                  threshold_low: float = 0.3,
                                  threshold_high: float = 0.7) -> Dict:
    """
    Generate maintenance recommendation based on failure probability
    
    Args:
        probability: Predicted failure probability
        threshold_low: Low risk threshold
        threshold_high: High risk threshold
    
    Returns:
        Dictionary with recommendation details
    """
    
    if probability < threshold_low:
        return {
            'status': 'NORMAL',
            'color': '#2ca02c',
            'icon': '✅',
            'message': 'Normal Operation - No Action Required',
            'description': 'Machine is operating within normal parameters. Continue regular monitoring.',
            'priority': 'Low'
        }
    elif probability < threshold_high:
        return {
            'status': 'WARNING',
            'color': '#ff7f0e',
            'icon': '⚠️',
            'message': 'Schedule Preventive Maintenance',
            'description': 'Elevated risk detected. Schedule maintenance within the next 24-48 hours.',
            'priority': 'Medium'
        }
    else:
        return {
            'status': 'CRITICAL',
            'color': '#d62728',
            'icon': '🚨',
            'message': 'Immediate Maintenance Required',
            'description': 'High failure risk! Stop operations and perform immediate inspection.',
            'priority': 'High'
        }


def calculate_cost_savings(failures_prevented: int,
                          cost_per_failure: float,
                          implementation_cost: float = 50000) -> Dict:
    """
    Calculate ROI and cost savings from predictive maintenance
    
    Args:
        failures_prevented: Number of failures prevented
        cost_per_failure: Average cost per failure
        implementation_cost: Cost to implement the system
    
    Returns:
        Dictionary with financial metrics
    """
    
    total_savings = failures_prevented * cost_per_failure
    net_savings = total_savings - implementation_cost
    roi = (net_savings / implementation_cost) * 100 if implementation_cost > 0 else 0
    payback_months = (implementation_cost / (total_savings / 12)) if total_savings > 0 else float('inf')
    
    return {
        'total_savings': total_savings,
        'implementation_cost': implementation_cost,
        'net_savings': net_savings,
        'roi_percentage': roi,
        'payback_period_months': payback_months
    }