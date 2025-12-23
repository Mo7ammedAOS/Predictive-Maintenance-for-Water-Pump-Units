"""
Model Training and Evaluation Utilities
Handles ML model training, optimization, and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from typing import Dict, Tuple, Any
import joblib
import streamlit as st


class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self, random_state: int = 21):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.results = {}
    
    def train_logistic_regression(self, 
                                 X_train: pd.DataFrame, 
                                 y_train: pd.Series,
                                 cv_folds: int = 5) -> Dict:
        """Train and optimize Logistic Regression model"""
        
        model = LogisticRegression(n_jobs=-1, random_state=self.random_state)
        
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
            random_state=self.random_state
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
        """Train and optimize Random Forest model"""
        
        model = RandomForestClassifier(
            criterion='gini',
            random_state=self.random_state,
            n_jobs=-1
        )
        
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
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['Random Forest'] = grid_search.best_estimator_
        self.best_params['Random Forest'] = grid_search.best_params_
        
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
        """Train and optimize XGBoost model"""
        
        model = XGBClassifier(random_state=self.random_state)
        
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
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models['XGBoost'] = grid_search.best_estimator_
        self.best_params['XGBoost'] = grid_search.best_params_
        
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
        """Evaluate a trained model on test data"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        f1_macro = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        misclassifications = (y_test != y_pred).sum()
        
        self.results[model_name] = {
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'classification_report': report,
            'misclassifications': misclassifications,
            'accuracy': (y_test == y_pred).sum() / len(y_test)
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
            comparison.append({
                'Model': model_name,
                'Macro F1 Score': round(self.results[model_name]['f1_macro'], 4),
                'Accuracy': round(self.results[model_name]['accuracy'], 4),
                'Misclassifications': self.results[model_name]['misclassifications'],
                'Best Parameters': str(self.best_params.get(model_name, {}))
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Macro F1 Score', ascending=False)
        
        return comparison_df


def calculate_shap_importance(model: Any, 
                             X_sample: pd.DataFrame,
                             sample_size: int = 1000) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate SHAP values for feature importance
    
    Args:
        model: Trained model
        X_sample: Sample of features to explain
        sample_size: Number of samples to use
    
    Returns:
        Tuple of (shap_values, importance_df)
    """
    import shap
    
    # Sample data if too large
    if len(X_sample) > sample_size:
        X_sample = X_sample.sample(n=sample_size, random_state=42)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle different output formats
    if isinstance(shap_values, list):
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values
    
    # Calculate mean absolute SHAP values
    shap_importance = np.abs(shap_values_plot).mean(axis=0)
    
    if len(shap_importance.shape) > 1:
        shap_importance = shap_importance.mean(axis=1)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': X_sample.columns,
        'SHAP_Importance': shap_importance
    }).sort_values('SHAP_Importance', ascending=False)
    
    return shap_values_plot, importance_df


@st.cache_resource
def train_all_models(X_train: pd.DataFrame, 
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series) -> ModelTrainer:
    """
    Train all models and return trainer object
    Cached to avoid retraining
    """
    
    trainer = ModelTrainer()
    
    with st.spinner('Training Logistic Regression...'):
        trainer.train_logistic_regression(X_train, y_train)
        trainer.evaluate_model('Logistic Regression', X_test, y_test)
    
    with st.spinner('Training SVM...'):
        trainer.train_svm(X_train, y_train)
        trainer.evaluate_model('SVM', X_test, y_test)
    
    with st.spinner('Training Random Forest...'):
        trainer.train_random_forest(X_train, y_train)
        trainer.evaluate_model('Random Forest', X_test, y_test)
    
    with st.spinner('Training XGBoost...'):
        trainer.train_xgboost(X_train, y_train)
        trainer.evaluate_model('XGBoost', X_test, y_test)
    
    return trainer


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
            'color': 'green',
            'icon': '✅',
            'message': 'Normal Operation - No Action Required',
            'description': 'Machine is operating within normal parameters. Continue regular monitoring.',
            'priority': 'Low'
        }
    elif probability < threshold_high:
        return {
            'status': 'WARNING',
            'color': 'orange',
            'icon': '⚠️',
            'message': 'Schedule Preventive Maintenance',
            'description': 'Elevated risk detected. Schedule maintenance within the next 24-48 hours.',
            'priority': 'Medium'
        }
    else:
        return {
            'status': 'CRITICAL',
            'color': 'red',
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
    
    return {
        'total_savings': total_savings,
        'implementation_cost': implementation_cost,
        'net_savings': net_savings,
        'roi_percentage': roi,
        'payback_period_months': (implementation_cost / (total_savings / 12)) if total_savings > 0 else 0
    }