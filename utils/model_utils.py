"""
Model Training and Evaluation Utilities
Handles patient ML model training, optimization, and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
    """Class to handle model training and evaluation"""
    
    def __init__(self, random_state: int = 21):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.results = {}
        self.training_history = []
    
    def train_logistic_regression(self, 
                                 X_train: pd.DataFrame, 
                                 y_train: pd.Series,
                                 cv_folds: int = 5) -> Dict:
        """Train and optimize Logistic Regression model"""
        
        model = LogisticRegression(n_jobs=-1, random_state=self.random_state, max_iter=1000)
        
        param_grid = {
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        
        folds = TimeSeriesSplit(n_splits=cv_folds)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=folds,
            scoring='f1_macro',
            n_jobs=-1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['Logistic Regression'] = grid_search.best_estimator_
        self.best_params['Logistic Regression'] = grid_search.best_params_
        self.training_history.append(('Logistic Regression', 'Grid Search Complete'))
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_,
            'best_score': grid_search.best_score_
        }
    
    def train_svm(self, 
                  X_train: pd.DataFrame, 
                  y_train: pd.Series,
                  cv_folds: int = 5) -> Dict:
        """Train and optimize SVM model"""
        
        model = SGDClassifier(
            loss="hinge", 
            penalty='l2', 
            n_jobs=-1, 
            random_state=self.random_state,
            max_iter=1000
        )
        
        param_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        
        folds = TimeSeriesSplit(n_splits=cv_folds)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=folds,
            scoring='f1_macro',
            n_jobs=-1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['SVM'] = grid_search.best_estimator_
        self.best_params['SVM'] = grid_search.best_params_
        self.training_history.append(('SVM', 'Grid Search Complete'))
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_,
            'best_score': grid_search.best_score_
        }
    
    def train_random_forest(self, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           cv_folds: int = 5) -> Dict:
        """Train and optimize Random Forest model - PATIENCE REQUIRED"""
        
        model = RandomForestClassifier(
            criterion='gini',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        # Extended parameter grid for thorough search
        param_grid = {
            'n_estimators': [10, 25, 50, 100, 150, 200],
            'max_depth': [1, 3, 5, 10, 20, 30, 50]
        }
        
        folds = TimeSeriesSplit(n_splits=cv_folds)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=folds,
            scoring='f1_macro',
            n_jobs=-1,
            return_train_score=True,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['Random Forest'] = grid_search.best_estimator_
        self.best_params['Random Forest'] = grid_search.best_params_
        self.training_history.append(('Random Forest', f"Best params: {grid_search.best_params_}"))
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_,
            'best_score': grid_search.best_score_
        }
    
    def train_xgboost(self, 
                     X_train: pd.DataFrame, 
                     y_train: pd.Series,
                     cv_folds: int = 5) -> Dict:
        """Train and optimize XGBoost model - PATIENCE REQUIRED"""
        
        model = XGBClassifier(
            random_state=self.random_state,
            verbosity=1
        )
        
        # Extended parameter grid
        param_grid = {
            'n_estimators': [5, 10, 20, 30, 40, 50],
            'max_depth': [1, 3, 5, 7, 10, 20, 30]
        }
        
        folds = TimeSeriesSplit(n_splits=cv_folds)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=folds,
            scoring='f1_macro',
            n_jobs=-1,
            return_train_score=True,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['XGBoost'] = grid_search.best_estimator_
        self.best_params['XGBoost'] = grid_search.best_params_
        self.training_history.append(('XGBoost', f"Best params: {grid_search.best_params_}"))
        
        return {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_results': grid_search.cv_results_,
            'best_score': grid_search.best_score_
        }
    
    def evaluate_model(self, 
                      model_name: str,
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict:
        """Evaluate a trained model on test data with comprehensive metrics"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
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
        
        self.results[model_name] = {
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
        
        return self.results[model_name]
    
    def get_feature_importance(self, 
                              model_name: str,
                              feature_names: list) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError(f"Model {model_name} does not have feature_importances_")
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to disk"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        joblib.dump(self.models[model_name], filepath)
    
    def load_model(self, model_name: str, filepath: str):
        """Load trained model from disk"""
        
        self.models[model_name] = joblib.load(filepath)
    
    def compare_models(self) -> pd.DataFrame:
        """Create comparison table of all trained models"""
        
        comparison = []
        
        for model_name in self.results:
            result = self.results[model_name]
            comparison.append({
                'Model': model_name,
                'Accuracy': round(result['accuracy'], 4),
                'F1 Macro': round(result['f1_macro'], 4),
                'F1 Weighted': round(result['f1_weighted'], 4),
                'Broken Recall': round(result['recall'][0], 4),  # Recall for class 0 (BROKEN)
                'Normal Recall': round(result['recall'][1], 4),  # Recall for class 1 (NORMAL)
                'False Negatives': result['broken_false_negatives'],
                'False Positives': result['broken_false_positives'],
                'Total Misclassifications': result['misclassifications'],
                'Best Parameters': str(self.best_params.get(model_name, {}))
            })
        
        comparison_df = pd.DataFrame(comparison)
        
        # Sort by: Lower False Negatives (most important), then higher Accuracy
        comparison_df = comparison_df.sort_values(
            by=['False Negatives', 'Accuracy'],
            ascending=[True, False]
        )
        
        return comparison_df


def predict_with_confidence(model: Any, 
                           X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores
    
    Args:
        model: Trained model
        X: Features
    
    Returns:
        Tuple of (predictions, probabilities)
    """
    predictions = model.predict(X)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
    else:
        # For models without predict_proba, use decision_function
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(X)
            # Convert to probabilities using sigmoid
            probabilities = 1 / (1 + np.exp(-decision))
            probabilities = np.column_stack([1 - probabilities, probabilities])
        else:
            # Default to binary predictions
            probabilities = np.column_stack([1 - predictions, predictions])
    
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