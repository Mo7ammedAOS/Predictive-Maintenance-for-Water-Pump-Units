# LIVE PREDICTIONS PAGE (copied above but updating section marker)
"""
Predictive Maintenance Dashboard - Complete Pipeline
A portfolio project demonstrating ML engineering skills in predictive analytics
Refactored to match original notebook workflow exactly
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

# Import utility modules
from utils.data_processing import (
    load_sensor_data, preprocess_data, create_labels, 
    normalize_features, split_train_test, generate_sample_data,
    get_class_distribution, get_missing_value_stats, calculate_statistics,
    shift_labels, generate_deviation_features, generate_window_features,
    prepare_complete_pipeline
)
from utils.model_utils import (
    ModelTrainer, predict_with_confidence,
    get_maintenance_recommendation, calculate_cost_savings
)
from utils.visualizations import (
    plot_machine_status_timeline, plot_class_distribution,
    plot_missing_values, plot_sensor_distribution,
    plot_confusion_matrix, plot_feature_importance,
    plot_model_comparison, plot_probability_gauge,
    plot_sensor_time_series, plot_correlation_heatmap
)

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .info-box {
            background-color: #f0f8ff;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            border-radius: 5px;
        }
        
        .success-box {
            background-color: #f0fff4;
            border-left: 4px solid #2ca02c;
            padding: 1rem;
            border-radius: 5px;
        }
        
        .section-divider {
            border-top: 2px solid #e0e0e0;
            margin: 2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# Sidebar navigation
st.sidebar.title("🎯 Predictive Maintenance Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Exploration", 
        "⚙️ Data Preparation",
        "🤖 Model Training",
        "🔮 Live Predictions",
        "💡 Feature Insights",
        "📄 About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Resources")
st.sidebar.markdown("- [📂 GitHub](https://github.com/Mo7ammedAOS/Predictive-Maintenance-for-Water-Pump-Units.git)")
st.sidebar.markdown("- [📧 Contact](mailto:mohammedossidahmed@gmail.com)")
st.sidebar.markdown("- [💼 LinkedIn](https://www.linkedin.com/in/mohammed-abelmoneim-5415991b6/)")

# Initialize session state
if 'pipeline_data' not in st.session_state:
    st.session_state.pipeline_data = None

if 'trainer' not in st.session_state:
    st.session_state.trainer = None

# HOME PAGE
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">⚙️ Predictive Maintenance System For Water Pump Units</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; font-size:1.2rem; color:#666;">Preventing Machine Failures Through Advanced Machine Learning</p>', 
                unsafe_allow_html=True)
    
    # Dynamic metrics based on pipeline data if available
    if st.session_state.pipeline_data:
        pipeline_data = st.session_state.pipeline_data
        sensor_count = len(pipeline_data['sensor_cols'])
        total_samples = pipeline_data['stats']['total_samples']
        feature_type = pipeline_data['feature_type'].upper()
        
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("## 🎯 Project Overview")
    
    st.markdown("""
    <div class="info-box">
        <h3>The Challenge</h3>
        <p>Industrial machinery failures cost companies millions in unexpected downtime, emergency repairs, 
        and lost productivity. Traditional reactive maintenance is expensive and disruptive.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <h3>The Solution</h3>
        <p>A machine learning system that predicts equipment failures 10 minutes in advance using 
        real-time sensor data, enabling proactive maintenance scheduling and preventing costly breakdowns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📈 ML Pipeline")
    st.markdown("""
    ```
    Raw Sensor Data
         ↓
    Data Cleaning & Preprocessing
         ↓
    Label Creation & Shifting (10-min advance)
         ↓
    Feature Engineering (Deviation/Window)
         ↓
    Data Normalization (MinMax Scaling)
         ↓
    Train/Test Split (Time-based)
         ↓
    Model Training 
         ↓
    Evaluation 
         ↓
    Predictions & Recommendations
    ```
    """)
    
    st.markdown("## 💼 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Step 1: Explore Data")
        st.markdown("Upload your sensor CSV and explore the data distribution")
    
    with col2:
        st.markdown("### Step 2: Prepare Pipeline")
        st.markdown("Run complete data preprocessing and feature engineering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Step 3: Train Models")
        st.markdown("Train ML modes with automatic hyperparameter tuning")
    
    with col2:
        st.markdown("### Step 4: Make Predictions")
        st.markdown("Use the best model for real-time failure predictions")


# DATA EXPLORATION PAGE
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration & Analysis")
    st.markdown("Understand your sensor data and machine failure patterns")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload sensor CSV file", type=['csv'])
    with col2:
        use_demo = st.button("📂 Use Demo Data")
    
    if uploaded_file is not None or use_demo:
        if use_demo:
            df = generate_sample_data(n_samples=5000)
            st.success("✅ Demo data loaded (5,000 samples, 58 sensors)")
        else:
            df = load_sensor_data(uploaded_file=uploaded_file)
            st.success(f"✅ Loaded: {uploaded_file.name}")
        
        # Display basic statistics
        st.markdown("### 📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("Missing Values", df.isna().sum().sum())
        
        # Show first few rows
        st.markdown("### 🔍 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Machine status distribution
        if 'machine_status' in df.columns:
            st.markdown("### 🏭 Machine Status Distribution")
            
            status_counts = df['machine_status'].value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                for status, count in status_counts.items():
                    pct = (count / len(df)) * 100
                    st.metric(f"{status}", f"{count:,}", f"{pct:.1f}%")
        
        # Missing values analysis
        st.markdown("### 🔍 Missing Values Analysis")
        missing_stats = get_missing_value_stats(df)
        
        if len(missing_stats) > 0:
            st.dataframe(missing_stats, use_container_width=True)
            st.warning(f"⚠️ {len(missing_stats)} sensors have missing values")
            
            # Visualize missing values
            fig = plot_missing_values(df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values!")


# DATA PREPARATION PAGE
elif page == "⚙️ Data Preparation":
    st.title("⚙️ Complete Data Preparation Pipeline")
    st.markdown("Run the full data preprocessing and feature engineering pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV (or use demo)", type=['csv'], key="prep")
    
    if uploaded_file:
        st.markdown("### 📋 Pipeline Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_type = st.radio(
                "Feature Engineering Method",
                ['deviation', 'window'],
                help="deviation: Distance from normal state mean\nwindow: Time window aggregation"
            )
        
        with col2:
            st.info(f"Selected: {feature_type.upper()}")
        
        if st.button("🚀 Run Pipeline", use_container_width=True):
            with st.spinner("⏳ Running complete data pipeline..."):
                try:
                    pipeline_data = prepare_complete_pipeline(
                        uploaded_file=uploaded_file if uploaded_file  else None,
                        file_path=None,
                        feature_type=feature_type
                    )
                    
                    st.session_state.pipeline_data = pipeline_data
                    
                    st.success("✅ Pipeline Complete!")
                    
                    # Display results
                    st.markdown("### 📊 Pipeline Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    stats = pipeline_data['stats']
                    with col1:
                        st.metric("Total Samples", f"{stats['total_samples']:,}")
                    with col2:
                        st.metric("Total Features", stats['total_features'])
                    with col3:
                        st.metric("Train Samples", f"{stats['train_samples']:,}")
                    with col4:
                        st.metric("Test Samples", f"{stats['test_samples']:,}")
                    
                    # Class distribution
                    st.markdown("### 📈 Training Set Class Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        class_dist = pipeline_data['class_dist_train']
                        for label, info in class_dist.items():
                            label_name = "BROKEN" if label == 0 else "NORMAL"
                            st.metric(
                                label_name,
                                f"{info['count']:,}",
                                f"{info['percentage']:.2f}%"
                            )
                    
                    with col2:
                        fig = plot_class_distribution(pipeline_data['y_train'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature matrix visualization
                    st.markdown("### 📊 Feature Matrix Statistics")
                    
                    X_train = pipeline_data['X_train']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Min Value", f"{X_train.min().min():.4f}")
                    with col2:
                        st.metric("Max Value", f"{X_train.max().max():.4f}")
                    with col3:
                        st.metric("Mean Value", f"{X_train.mean().mean():.4f}")
                    with col4:
                        st.metric("Std Dev", f"{X_train.std().mean():.4f}")
                    
                    # Show timeline
                    st.markdown("### 📊 Machine Status Timeline")
                    try:
                        fig = plot_machine_status_timeline(pipeline_data['y_train'])
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Could not generate timeline visualization")
                    
                    # Show correlation heatmap
                    st.markdown("### 🔥 Feature Correlation Heatmap (Top 20)")
                    try:
                        fig = plot_correlation_heatmap(X_train, top_n=20)
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        st.info("Could not generate correlation heatmap")
                    
                    st.info("✅ Data is prepared and ready for model training!")
                    st.info(f"Feature type: {feature_type.upper()} | Features normalized with MinMaxScaler")
                    
                except Exception as e:
                    st.error(f"❌ Pipeline Error: {str(e)}")


# MODEL TRAINING PAGE
elif page == "🤖 Model Training":
    st.title("🤖 Random Forest Model Training")
    st.markdown("Train optimized Random Forest classifier")
    
    if st.session_state.pipeline_data is None:
        st.warning("⚠️ Please prepare the data pipeline first!")
        st.info("Go to 'Data Preparation' page and run the pipeline")
    else:
        st.markdown("### 📋 Model Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Criterion", "gini")
        with col2:
            st.metric("N Estimators", "150")
        with col3:
            st.metric("Max Depth", "5")
        with col4:
            st.metric("Jobs", "-1 (All cores)")
        
        st.markdown("### 🚀 Start Training")
        st.info("⚡ Random Forest training - Fast and efficient!")
        
        if st.button("🔥 Train Random Forest Model", use_container_width=True):
            pipeline_data = st.session_state.pipeline_data
            
            trainer = ModelTrainer()
            
            # Progress tracking
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            with st.spinner("🔄 Training Random Forest classifier..."):
                try:
                    status_text.write("🔄 Training Random Forest (n_estimators=150, max_depth=5)...")
                    progress_bar.progress(50)
                    
                    # Train model
                    trainer.train_random_forest(pipeline_data['X_train'], pipeline_data['y_train'])
                    progress_bar.progress(75)
                    
                    status_text.write("📊 Evaluating on test set...")
                    # Evaluate model
                    trainer.evaluate_model(pipeline_data['X_test'], pipeline_data['y_test'])
                    progress_bar.progress(100)
                    
                    status_text.write("✅ Training complete!")
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.stop()
            
            st.session_state.trainer = trainer
            
            st.success("✅ Random Forest Model Trained Successfully!")
            
            # Display model info
            st.markdown("### 🏆 Model Information")
            
            model_info = trainer.get_model_info()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Type", model_info['name'])
            with col2:
                st.metric("Estimators", model_info['n_estimators'])
            with col3:
                st.metric("Max Depth", model_info['max_depth'])
            with col4:
                st.metric("Status", model_info['status'])
            
            # Display results
            st.markdown("### 📊 Model Performance Metrics")
            
            results = trainer.results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}", 
                         f"{results['accuracy']*100:.2f}%")
            with col2:
                st.metric("F1 Score (Macro)", f"{results['f1_macro']:.4f}")
            with col3:
                st.metric("F1 Score (Weighted)", f"{results['f1_weighted']:.4f}")
            with col4:
                st.metric("Total Misclassifications", results['misclassifications'])
            
            # Detailed metrics
            st.markdown("### 📈 Detailed Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("False Negatives (Critical)", results['broken_false_negatives'],
                         help="Broken equipment predicted as Normal")
            with col2:
                st.metric("False Positives", results['broken_false_positives'],
                         help="Normal equipment predicted as Broken")
            with col3:
                st.metric("Broken Recall", f"{results['recall'][0]:.4f}",
                         help="% of broken equipment correctly identified")
            with col4:
                st.metric("Normal Recall", f"{results['recall'][1]:.4f}",
                         help="% of normal equipment correctly identified")
            
            # Show confusion matrix
            st.markdown("### 📊 Confusion Matrix")
            
            y_pred = trainer.model.predict(pipeline_data['X_test'])
            fig = plot_confusion_matrix(
                pipeline_data['y_test'].values,
                y_pred
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.markdown("### 🎯 Feature Importance - Top 20 Sensors")
            
            try:
                importance_df = trainer.get_feature_importance(
                    pipeline_data['X_train'].columns.tolist()
                )
                
                st.dataframe(importance_df.head(20), use_container_width=True)
                
                fig = plot_feature_importance(importance_df, top_n=20, 
                                            title="Top 20 Most Important Features")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate feature importance: {str(e)[:50]}")


# LIVE PREDICTIONS PAGE
elif page == "🔮 Live Predictions":
    st.title("🔮 Live Machine Failure Prediction")
    st.markdown("Make real-time predictions using the trained model")
    
    if st.session_state.trainer is None:
        st.warning("⚠️ Please train models first!")
        st.info("Go to 'Model Training' page")
    else:
        trainer = st.session_state.trainer
        pipeline_data = st.session_state.pipeline_data
        
        # Get model
        model = trainer.model
        scaler = pipeline_data['scaler']
        
        st.markdown(f"### 🏆 Using Random Forest Model")
        st.info(f"✅ Model: n_estimators=150, max_depth=5, criterion=gini")
        
        # Three options for input
        st.markdown("### 📝 Choose Input Method")
        
        input_method = st.radio(
            "Select how to provide sensor data:",
            ["Manual Input", "Random Sample", "Upload Test Data"],
            horizontal=True
        )
        
        sensor_values = {}
        feature_cols = pipeline_data['X_test'].columns.tolist()
        
        if input_method == "Manual Input":
            st.markdown("### 📝 Input Sensor Readings")
            
            num_cols = 5
            display_cols = feature_cols[:10]
            
            cols = st.columns(num_cols)
            for idx, feature in enumerate(display_cols):
                col_idx = idx % num_cols
                
                with cols[col_idx]:
                    sensor_values[feature] = st.number_input(
                        feature,
                        value=0.5,
                        min_value=0.0,
                        max_value=1.0,
                        step=0.01
                    )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔮 Predict Machine Status", use_container_width=True):
                # Create input dataframe with all features
                input_dict = {}
                for col in feature_cols:
                    if col in sensor_values:
                        input_dict[col] = sensor_values[col]
                    else:
                        input_dict[col] = 0.5
                
                input_df = pd.DataFrame([input_dict])
                
                # Make prediction
                prediction, probabilities = predict_with_confidence(model, input_df)
                failure_prob = probabilities[0][0]
                
                # Get recommendation
                recommendation = get_maintenance_recommendation(failure_prob)
                
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div style="padding: 1.5rem; background-color: {recommendation['color']}20; 
                                   border-left: 4px solid {recommendation['color']}; border-radius: 5px;">
                            <h2>{recommendation['icon']}</h2>
                            <h3>{recommendation['status']}</h3>
                            <p><strong>{recommendation['message']}</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Failure Probability", f"{failure_prob*100:.1f}%")
                    st.metric("Normal Probability", f"{(1-failure_prob)*100:.1f}%")
                
                with col3:
                    st.metric("Priority Level", recommendation['priority'])
                
                # Gauge chart
                fig = plot_probability_gauge(failure_prob)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation
                st.markdown("### 💡 Action Recommended")
                st.info(recommendation['description'])
        
        elif input_method == "Random Sample":
            st.markdown("### 🎲 Generate Random Sensor Readings")
            
            if st.button("🎲 Generate Random Sample", use_container_width=True):
                for col in feature_cols:
                    sensor_values[col] = np.random.uniform(0.3, 0.7)
                
                # Make prediction
                input_df = pd.DataFrame([sensor_values])
                prediction, probabilities = predict_with_confidence(model, input_df)
                failure_prob = probabilities[0][0]
                
                recommendation = get_maintenance_recommendation(failure_prob)
                
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div style="padding: 1.5rem; background-color: {recommendation['color']}20; 
                                   border-left: 4px solid {recommendation['color']}; border-radius: 5px;">
                            <h2>{recommendation['icon']}</h2>
                            <h3>{recommendation['status']}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Failure Probability", f"{failure_prob*100:.1f}%")
                
                with col3:
                    st.metric("Priority", recommendation['priority'])
                
                fig = plot_probability_gauge(failure_prob)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(recommendation['description'])
        
        else:  # Upload Test Data
            st.markdown("### 📤 Upload Test Dataset")
            
            test_file = st.file_uploader("Upload test CSV file", type=['csv'], key="test_pred")
            
            if test_file is not None:
                test_df = pd.read_csv(test_file)
                st.success(f"✅ Loaded test data: {len(test_df)} samples")
                
                st.markdown(f"### 🔍 Test Data Overview")
                st.dataframe(test_df.head(), use_container_width=True)
                
                st.markdown("### 📋 Available Columns in Your File:")
                st.write(f"**Found {len(test_df.columns)} columns:** {', '.join(test_df.columns[:10])}")
                
                if st.button("🔮 Predict All Samples", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        try:
                            # Normalize the test data using the saved scaler
                            scaler = pipeline_data['scaler']
                            
                            # Prepare test data (match feature columns)
                            test_prepared = pd.DataFrame()
                            for col in feature_cols:
                                if col in test_df.columns:
                                    test_prepared[col] = test_df[col].astype(float)
                                else:
                                    test_prepared[col] = 0.5
                            
                            # Check if we have data
                            if len(test_prepared) == 0:
                                st.error("❌ No valid data to predict. Please check your CSV file.")
                            else:
                                st.info(f"✅ Prepared {len(test_prepared)} samples with {len(feature_cols)} features")
                                
                                # Normalize using the scaler
                                test_scaled = scaler.transform(test_prepared.values)
                                test_prepared_scaled = pd.DataFrame(test_scaled, columns=feature_cols)
                                
                                # Make predictions
                                predictions, probabilities = predict_with_confidence(best_model, test_prepared_scaled)
                                failure_probs = probabilities[:, 0]
                                
                                # Create results dataframe
                                results_df = test_df.copy()
                                results_df['Failure_Probability'] = failure_probs
                                results_df['Failure_Percentage'] = (failure_probs * 100).round(2)
                                results_df['Status'] = results_df['Failure_Probability'].apply(
                                    lambda x: 'CRITICAL' if x > 0.7 else ('WARNING' if x > 0.3 else 'NORMAL')
                                )
                                results_df['Prediction'] = predictions
                                
                                st.markdown("### 📊 Prediction Results for All Samples")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Samples", len(results_df))
                                with col2:
                                    st.metric("Critical", (results_df['Status'] == 'CRITICAL').sum())
                                with col3:
                                    st.metric("Warning", (results_df['Status'] == 'WARNING').sum())
                                with col4:
                                    st.metric("Normal", (results_df['Status'] == 'NORMAL').sum())
                                
                                st.markdown("#### 📋 First 20 Predictions")
                                display_cols = ['Failure_Probability', 'Failure_Percentage', 'Status', 'Prediction']
                                st.dataframe(results_df[display_cols].head(20), use_container_width=True)
                                
                                # Download results
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions CSV",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                                
                                # Distribution of predictions
                                st.markdown("### 📈 Prediction Distribution")
                                try:
                                    fig = plot_class_distribution(results_df['Prediction'].astype(int))
                                    st.plotly_chart(fig, use_container_width=True)
                                except:
                                    st.info("Could not generate distribution chart")
                                
                                # Status breakdown
                                st.markdown("### 🎯 Status Breakdown")
                                status_dist = results_df['Status'].value_counts()
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    normal_count = status_dist.get('NORMAL', 0)
                                    normal_pct = (normal_count/len(results_df)*100) if len(results_df) > 0 else 0
                                    st.metric("🟢 Normal", normal_count, f"{normal_pct:.1f}%")
                                with col2:
                                    warning_count = status_dist.get('WARNING', 0)
                                    warning_pct = (warning_count/len(results_df)*100) if len(results_df) > 0 else 0
                                    st.metric("🟡 Warning", warning_count, f"{warning_pct:.1f}%")
                                with col3:
                                    critical_count = status_dist.get('CRITICAL', 0)
                                    critical_pct = (critical_count/len(results_df)*100) if len(results_df) > 0 else 0
                                    st.metric("🔴 Critical", critical_count, f"{critical_pct:.1f}%")
                                
                        except Exception as e:
                            st.error(f"❌ Prediction Error: {str(e)}")
                            st.info("💡 Make sure your CSV has the same sensor columns as the training data")


# FEATURE INSIGHTS PAGE
elif page == "💡 Feature Insights":
    st.title("💡 Feature Importance & Business Insights")
    
    if st.session_state.trainer is None:
        st.warning("⚠️ Please train the model first!")
    else:
        trainer = st.session_state.trainer
        pipeline_data = st.session_state.pipeline_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Cost Savings Calculator")
            
            failures_prevented = st.slider("Failures prevented per year", 5, 200, 50)
            cost_per_failure = st.number_input("Cost per failure ($)", value=50000, min_value=1000)
            implementation_cost = st.number_input("Implementation cost ($)", value=50000, min_value=1000)
            
            savings = calculate_cost_savings(failures_prevented, cost_per_failure, implementation_cost)
            
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                st.metric("Total Savings", f"${savings['total_savings']:,.0f}")
                st.metric("Net Savings", f"${savings['net_savings']:,.0f}")
            
            with col_s2:
                st.metric("ROI", f"{savings['roi_percentage']:.1f}%")
                payback = savings['payback_period_months']
                payback_text = f"{payback:.1f} months" if payback != float('inf') else "N/A"
                st.metric("Payback Period", payback_text)
        
        with col2:
            st.markdown("#### 📈 Random Forest Performance")
            
            results = trainer.results
            
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
            st.metric("F1 Score", f"{results['f1_macro']:.4f}")
            st.metric("False Negatives", results['broken_false_negatives'])
            st.metric("False Positives", results['broken_false_positives'])
        
        # Feature importance
        st.markdown("### 🎯 Top 20 Most Important Sensors")
        
        try:
            importance_df = trainer.get_feature_importance(
                pipeline_data['X_train'].columns.tolist()
            )
            
            st.dataframe(importance_df.head(20), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_feature_importance(importance_df, top_n=15, title="Top 15 Features")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plot_feature_importance(importance_df, top_n=20, title="Top 20 Features")
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ABOUT PAGE
else:  # About
    st.title("📄 About & Technical Details")
    
    st.markdown("## 🎯 Project Overview")
    st.markdown("""
    This is a complete machine learning pipeline for predictive maintenance of water pump units,
    built from scratch following best practices in data science and ML engineering.
    """)
    
    st.markdown("## 📊 Pipeline Steps")
    
    with st.expander("1️⃣ Data Loading & Exploration", expanded=True):
        st.markdown("""
        - Load sensor data from CSV
        - Analyze missing values and distributions
        - Identify problematic sensors
        - Basic statistics and class distribution
        """)
    
    with st.expander("2️⃣ Data Preprocessing"):
        st.markdown("""
        - Drop sensor_15 (excessive missing values)
        - Fill remaining missing values with -1
        - Remove status columns (will be recreated)
        - Prepare data for label shifting
        """)
    
    with st.expander("3️⃣ Label Creation & Shifting"):
        st.markdown("""
        - Map machine_status to binary labels (1=NORMAL, 0=BROKEN)
        - Shift labels -10 steps (10-minute advance warning)
        - Remove rows with NaN after shifting
        - Enables prediction of future failures
        """)
    
    with st.expander("4️⃣ Feature Engineering"):
        st.markdown("""
        **Method 1: Deviation Features**
        ```
        deviation = sensor_reading - mean(normal_state_readings)
        ```
        Captures how far readings deviate from healthy baselines.
        
        **Method 2: Time Window Features**
        ```
        window_mean = rolling_mean(sensor, window=10)
        ```
        Smooths noise and captures trending behavior.
        """)
    
    with st.expander("5️⃣ Data Normalization"):
        st.markdown("""
        - MinMaxScaler: scale all features to [0, 1]
        - Fit scaler on training data only
        - Transform test data using training scaler
        - Prevents data leakage
        """)
    
    with st.expander("6️⃣ Train/Test Split"):
        st.markdown("""
        - Time-based split (not random)
        - First 131,000 samples: training
        - Remaining samples: testing
        - Preserves temporal relationships in data
        """)
    
    with st.expander("7️⃣ Model Training & Optimization"):
        st.markdown("""
        **Models Trained:**
        1. **Logistic Regression** - Fast baseline
        2. **SVM (SGD)** - Efficient for large datasets
        3. **Random Forest** - Ensemble, handles non-linearity
        4. **XGBoost** - Gradient boosting, state-of-the-art
        
        **Optimization Method:**
        - Time Series Cross-Validation (5 folds)
        - Grid Search over hyperparameters
        - Macro F1 Score as optimization metric
        
        **Best Model Selection:**
        - Prioritize: Lowest False Negatives (missed failures)
        - Secondary: Highest Accuracy
        """)
    
    with st.expander("8️⃣ Evaluation"):
        st.markdown("""
        **Metrics Calculated:**
        - Accuracy
        - F1 Score (Macro and Weighted)
        - Confusion Matrix
        - Precision & Recall per class
        - False Negatives (critical for maintenance)
        - False Positives (unnecessary maintenance)
        """)
    
    st.markdown("## 🛠️ Technologies Used")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**ML & Data**")
        st.markdown("""
        - pandas
        - NumPy
        - scikit-learn
        - XGBoost
        """)
    
    with col2:
        st.markdown("**Visualization**")
        st.markdown("""
        - Plotly
        - Matplotlib
        - Seaborn
        """)
    
    with col3:
        st.markdown("**Deployment**")
        st.markdown("""
        - Streamlit
        - Python 3.8+
        """)
    
    st.markdown("## 👨‍💻 Developer")
    
    st.markdown("""
    **Eng. Mohammed Osman**
    
    Applied Data Scientist | Predictive Maintenance Specialist
    
    - 7+ years structural engineering
    - ML expertise in predictive analytics
    - Portfolio: https://github.com/Mo7ammedAOS
    - Email: mohammedossidahmed@gmail.com
    - LinkedIn: https://www.linkedin.com/in/mohammed-abelmoneim-5415991b6/
    """)