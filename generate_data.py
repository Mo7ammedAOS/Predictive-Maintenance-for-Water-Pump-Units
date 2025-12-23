import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
import os

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Create sample test data
np.random.seed(42)
n_samples = 1000
n_features = 51

# Generate sample features
X_test = np.random.randn(n_samples, n_features)

# Generate labels (with some pattern)
y_test = np.random.choice([0, 1], size=n_samples, p=[0.15, 0.85])

# Create test data DataFrame
feature_names = [f"sensor_{i:02d}_deviation" for i in range(52) if i != 15]
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['labels'] = y_test
test_df.to_csv('data_source/test_data.csv', index=False)

# Create and save sample models
print("Creating sample models...")

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    random_state=42
)
rf_model.fit(X_test, y_test)
joblib.dump(rf_model, 'models/random_forest.pkl')

# Logistic Regression
lr_model = LogisticRegression(C=0.01, random_state=42)
lr_model.fit(X_test, y_test)
joblib.dump(lr_model, 'models/logistic_regression.pkl')

# XGBoost
xgb_model = XGBClassifier(n_estimators=40, max_depth=5, random_state=42)
xgb_model.fit(X_test, y_test)
joblib.dump(xgb_model, 'models/xgboost.pkl')

# SVM
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_test, y_test)
joblib.dump(svm_model, 'models/svm.pkl')

print("✅ Sample models and data created successfully!")