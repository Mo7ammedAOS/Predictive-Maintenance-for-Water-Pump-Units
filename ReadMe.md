# 🔧 Predictive Maintenance for Water Pump Units

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Predicting water pump failures 10 minutes in advance using machine learning**

An end-to-end machine learning solution that predicts water pump failures before they occur, enabling proactive maintenance interventions and reducing operational downtime.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **predictive maintenance system** for water pump units using time-series sensor data. By analyzing patterns in sensor readings, the system can predict pump failures **10 minutes before they occur**, allowing maintenance teams to take preventive action.

### Why This Matters

- 🚫 **Prevents unexpected downtime** and service interruptions
- 💰 **Reduces maintenance costs** through proactive interventions
- ⚡ **Improves operational efficiency** by scheduling maintenance optimally
- 🛡️ **Enhances safety** by preventing catastrophic failures

---

## 🔍 Problem Statement

### Business Challenge

A remote field of water pumps experiences frequent unexpected failures, causing:
- Operational disruptions
- Increased repair costs
- Safety concerns
- Customer dissatisfaction

### Machine Learning Solution

**Problem Type:** Binary Classification
- **Class 0 (BROKEN):** Pump is failing or in critical state
- **Class 1 (NORMAL):** Pump is operating normally

**Prediction Window:** 10 minutes ahead of actual failure

**Evaluation Metrics:**
- Macro F1-Score
- Confusion Matrix
- Misclassification Count

---

## 📊 Dataset

### Data Characteristics

- **Collection Frequency:** Every 1 minute
- **Duration:** 5 months of continuous monitoring
- **Total Samples:** ~220,000 data points
- **Sensors:** 52 sensor channels (sensor_00 to sensor_51)
- **Target Variable:** Machine status (NORMAL/BROKEN/RECOVERING)

### Class Distribution
```
NORMAL:    ~85% (187,000 samples)
BROKEN:    ~15% (33,000 samples)
```

### Data Quality Issues Addressed

- ✅ **Sensor_15:** Completely dropped (100% missing values)
- ✅ **Intermittent missing values:** Filled with -1 (out-of-distribution marker)
- ✅ **Pattern discovered:** Missing values correlate with BROKEN states
- ✅ **Label consolidation:** RECOVERING merged with BROKEN (pattern analysis showed they're equivalent)

---

## ✨ Key Features

### 1. **Intelligent Feature Engineering**

#### Distance from Normal Mean
```python
feature = sensor_reading - mean(NORMAL_state_readings)
```
Captures deviation from healthy operating conditions

#### Rolling Window Statistics
```python
feature = rolling_mean(sensor_readings, window=10_minutes)
```
Smooths noise and captures trends leading to failure

### 2. **Time-Series Cross-Validation**

- Uses `TimeSeriesSplit` to prevent data leakage
- Maintains temporal ordering during model training
- 5-fold cross-validation for robust performance estimates

### 3. **Multiple Algorithm Comparison**

Evaluated 4 different algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

## 🚀 Methodology

### Pipeline Overview
```
Raw Sensor Data
      ↓
Data Preprocessing
      ↓
Feature Engineering (2 strategies)
      ↓
Label Shifting (-10 minutes)
      ↓
Train/Test Split (temporal)
      ↓
Normalization (MinMaxScaler)
      ↓
Model Training (4 algorithms)
      ↓
Hyperparameter Tuning (GridSearchCV)
      ↓
Model Evaluation
      ↓
Best Model Selection
```

### Feature Engineering Strategies

#### Strategy 1: Deviation Features
- Compute difference from normal state mean
- Highlights abnormal sensor behavior
- 51 features (one per sensor)

#### Strategy 2: Rolling Mean Features
- 10-minute rolling average
- Captures temporal trends
- Reduces noise impact
- **Best performing approach**

### Model Training

- **Split:** 131,000 training samples, ~39,000 test samples
- **Validation:** TimeSeriesSplit (5 folds)
- **Scoring:** Macro F1-score (handles class imbalance)
- **Hyperparameters tuned:**
  - Logistic Regression: C (regularization)
  - SVM: alpha (penalty)
  - Random Forest: n_estimators, max_depth
  - XGBoost: n_estimators, max_depth

---

## 🏆 Results

### Final Performance Comparison

| Model | Feature Strategy | Macro F1-Score | Misclassifications | Rank |
|-------|-----------------|----------------|-------------------|------|
| **Random Forest** 🥇 | Rolling Mean (10-min) | **0.9949** | **53** | 1st |
| XGBoost | Rolling Mean (10-min) | 0.9900 | 203 | 2nd |
| Logistic Regression | Rolling Mean (10-min) | 0.9794 | 379 | 3rd |
| SVM | Rolling Mean (10-min) | 0.9608 | 834 | 4th |
| Random Forest | Deviation from Mean | 0.9938 | 65 | - |
| XGBoost | Deviation from Mean | 0.9892 | 170 | - |
| Logistic Regression | Deviation from Mean | 0.9786 | 394 | - |
| SVM | Deviation from Mean | 0.9597 | 895 | - |

### 🎯 Champion Model: Random Forest

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    criterion='gini',
    random_state=21
)
```

**Performance Highlights:**
- ✅ **99.49% Macro F1-Score**
- ✅ **99.86% Accuracy** (53 errors out of 39,000 samples)
- ✅ **Excellent recall** for BROKEN class (critical for safety)
- ✅ **Stable performance** across all cross-validation folds

### Confusion Matrix (Test Set)
```
                Predicted
              BROKEN  NORMAL
Actual BROKEN   5,845     35
       NORMAL      18  33,102
```

**Key Insights:**
- Only **35 false negatives** (missed failures) → 99.4% failure detection rate
- Only **18 false positives** (false alarms) → Minimal unnecessary interventions

---

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8 or higher
pip or conda package manager
```

### Clone Repository
```bash
git clone https://github.com/yourusername/water-pump-predictive-maintenance.git
cd water-pump-predictive-maintenance
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
prettytable>=2.5.0
```

---

## 💻 Usage

### 1. Data Preparation
```python
# Load sensor data
import pandas as pd
sensor_data = pd.read_csv('data_source/sensor.csv')

# Run preprocessing
from preprocessing import preprocess_data
processed_data = preprocess_data(sensor_data)
```

### 2. Feature Engineering
```python
from features import generate_rolling_features

# Create 10-minute rolling mean features
featured_data = generate_rolling_features(processed_data, window=10)
```

### 3. Train Model
```python
from model import train_random_forest

# Train the champion model
model = train_random_forest(X_train, y_train)
```

### 4. Make Predictions
```python
# Predict pump status 10 minutes ahead
predictions = model.predict(X_test)

# Get failure probabilities
probabilities = model.predict_proba(X_test)
```

### 5. Real-Time Inference
```python
# For live sensor data stream
import joblib

# Load trained model
model = joblib.load('models/random_forest_model.pkl')

# Predict on new data
current_status = model.predict(latest_sensor_readings)

if current_status == 0:
    print("⚠️ ALERT: Pump failure predicted in 10 minutes!")
```

---

## 🔧 Technologies Used

### Core Technologies

- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework

### Visualization

- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualizations
- **PrettyTable** - Formatted result tables

### Development Tools

- **Jupyter Notebook** - Interactive development
- **Git** - Version control

---

App Demo : [Water Pump Units PDM Dashboard](https://predictive-maintenance-for-water-pump-units.streamlit.app/)


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Your Name**
- GitHub: [@Mo7ammedAOS](https://github.com/Mo7ammedAOS)
- LinkedIn: [Mohammed Abdelmoneim](https://www.linkedin.com/in/mohammed-abelmoneim-5415991b6/)
- Email: [mohammedossidahmed@gmail.com](mohammedossidahmed@gmail.com)

---


## 🙏 Acknowledgments

- Dataset provided by [[Source/Organization](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)]
- Inspired by industrial predictive maintenance best practices
- Thanks to the open-source community for amazing libraries

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

## 📚 References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. XGBoost Documentation: https://xgboost.readthedocs.io/

---

<div align="center">

**Built with ❤️ for smarter maintenance operations**

[⬆ Back to Top](#-predictive-maintenance-for-water-pump-units)

</div>