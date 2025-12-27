import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Water Pump Predictive Maintenance",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric label {
        color: white !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
@st.cache_data
def load_data(file):
    """Load and cache uploaded data"""
    df = pd.read_csv(file)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

@st.cache_resource
def load_model():
    """Load trained model and preprocessing components"""
    try:
        model_package = joblib.load('./models/water_pump_model.pkl')
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'water_pump_model.pkl' is in the app directory.")
        return None

def preprocess_data(df):
    """Preprocess raw sensor data"""
    df = df.copy()
    
    if 'machine_status' in df.columns:
        df['labels'] = df['machine_status'].map(lambda x: 1 if x == 'NORMAL' else 0)
    
    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col != 'sensor_15']
    for col in sensor_cols:
        if col in df.columns:
            df[col].fillna(-1, inplace=True)
    
    if 'sensor_15' in df.columns:
        df.drop(columns=['sensor_15'], inplace=True)
    
    return df

def engineer_features(df, model_package):
    """Create features matching the model's expected format"""
    feature_type = model_package.get('feature_type', 'deviation')
    
    if feature_type == 'deviation' and model_package.get('normal_means'):
        features = {}
        sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col != 'sensor_15']
        
        for sensor in sensor_cols:
            if sensor in df.columns:
                if sensor in model_package['normal_means']:
                    features[f'{sensor}_deviation'] = df[sensor] - model_package['normal_means'][sensor]
                else:
                    features[f'{sensor}_deviation'] = df[sensor]
        
        return pd.DataFrame(features, index=df.index)
    else:
        sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col != 'sensor_15']
        return df[sensor_cols].copy()

def calculate_metrics(y_true, y_pred):
    """Calculate all performance metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    # Class-specific metrics
    f1_class = f1_score(y_true, y_pred, average=None)
    precision_class = precision_score(y_true, y_pred, average=None)
    recall_class = recall_score(y_true, y_pred, average=None)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'f1_broken': f1_class[0] if len(f1_class) > 1 else 0,
        'f1_normal': f1_class[1] if len(f1_class) > 1 else f1_class[0],
        'precision_broken': precision_class[0] if len(precision_class) > 1 else 0,
        'precision_normal': precision_class[1] if len(precision_class) > 1 else precision_class[0],
        'recall_broken': recall_class[0] if len(recall_class) > 1 else 0,
        'recall_normal': recall_class[1] if len(recall_class) > 1 else recall_class[0],
        'confusion_matrix': cm,
        'total_predictions': len(y_pred),
        'correct_predictions': (y_true == y_pred).sum(),
        'misclassifications': (y_true != y_pred).sum()
    }

def plot_status_timeline(df):
    """Plot machine status over time"""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    y = df['labels'].values
    x = df.index
    
    # Calculate statistics for display
    normal_pct = (y == 1).sum() / len(y) * 100
    broken_pct = (y == 0).sum() / len(y) * 100
    
    ax.plot(x, y, linewidth=1.5, color='#1f77b4', alpha=0.8)
    ax.fill_between(x, y, alpha=0.3, color='#1f77b4')
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('Status', fontsize=11)
    ax.set_title(f'Machine Status Timeline - Normal: {normal_pct:.1f}% | Broken: {broken_pct:.1f}%', 
                 fontsize=12, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['BROKEN', 'NORMAL'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, {'normal_pct': normal_pct, 'broken_pct': broken_pct}

def plot_sensor_distribution(df, sensor):
    """Plot sensor value distributions"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    normal = df[df['labels'] == 1][sensor].dropna()
    broken = df[df['labels'] == 0][sensor].dropna()
    
    if len(normal) > 0:
        sns.kdeplot(normal, label=f"Normal (n={len(normal)})", fill=True, color='#2ecc71', ax=ax)
    if len(broken) > 0:
        sns.kdeplot(broken, label=f"Broken (n={len(broken)})", fill=True, color='#e74c3c', ax=ax)
    
    # Calculate separation metric
    if len(normal) > 0 and len(broken) > 0:
        mean_diff = abs(normal.mean() - broken.mean())
        std_pooled = np.sqrt((normal.std()**2 + broken.std()**2) / 2)
        separation = mean_diff / std_pooled if std_pooled > 0 else 0
        
        ax.set_title(f'{sensor} - Separation Score: {separation:.2f}', 
                    fontsize=11, fontweight='bold')
    else:
        ax.set_title(f'{sensor} Distribution', fontsize=11, fontweight='bold')
    
    ax.set_xlabel(f'{sensor} readings', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(cm, metrics):
    """Plot confusion matrix with metrics"""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calculate percentages
    cm_pct = cm.astype('float') / cm.sum() * 100
    
    # Annotations with counts and percentages
    annot = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)' 
                      for j in range(cm.shape[1])] 
                      for i in range(cm.shape[0])])
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=['Broken', 'Normal'],
                yticklabels=['Broken', 'Normal'],
                cbar_kws={'label': 'Count'}, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    
    # Add accuracy in title
    accuracy = metrics['accuracy'] * 100
    ax.set_title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def analyze_sensor_importance(df, feature_importance_df):
    """Analyze sensor importance with statistics"""
    sensor_stats = []
    
    for idx, row in feature_importance_df.head(10).iterrows():
        sensor = row['Feature'].replace('_deviation', '')
        
        if sensor in df.columns:
            normal_vals = df[df['labels'] == 1][sensor].dropna()
            broken_vals = df[df['labels'] == 0][sensor].dropna()
            
            stats = {
                'sensor': row['Feature'],
                'importance': row['Importance'],
                'normal_mean': normal_vals.mean() if len(normal_vals) > 0 else 0,
                'broken_mean': broken_vals.mean() if len(broken_vals) > 0 else 0,
                'normal_std': normal_vals.std() if len(normal_vals) > 0 else 0,
                'broken_std': broken_vals.std() if len(broken_vals) > 0 else 0,
                'missing_rate': df[sensor].isna().sum() / len(df) * 100
            }
            
            stats['mean_diff'] = abs(stats['normal_mean'] - stats['broken_mean'])
            sensor_stats.append(stats)
    
    return pd.DataFrame(sensor_stats)

# Main Application
def main():
    st.markdown('<h1 class="main-header">💧 Water Pump Predictive Maintenance</h1>', 
                unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    if model_package is None:
        st.error("Cannot proceed without model file.")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        
        uploaded_file = st.file_uploader("📁 Upload Sensor Data", type=['csv'])
        
        if uploaded_file:
            st.success("✅ File loaded")
        
        st.markdown("---")
        
        page = st.radio("📍 Navigation", 
                       ["📊 Overview", 
                        "🔍 Sensor Analysis",
                        "🤖 Predictions",
                        "📈 Performance"])
        
        st.markdown("---")
        
        # Model info - DYNAMIC
        st.markdown("### 🤖 Model Info")
        model = model_package.get('model')
        if model:
            st.metric("Trees", f"{model.n_estimators}")
            st.metric("Max Depth", f"{model.max_depth}")
            st.metric("Features", f"{model.n_features_in_}")
    
    # Main content
    if uploaded_file is not None:
        # Load and process data
        df = load_data(uploaded_file)
        df_processed = preprocess_data(df.copy())
        
        # Store in session state
        if 'df_processed' not in st.session_state:
            st.session_state.df_processed = df_processed
        
        # Route to pages
        if page == "📊 Overview":
            show_overview(df_processed)
        elif page == "🔍 Sensor Analysis":
            show_sensor_analysis(df_processed)
        elif page == "🤖 Predictions":
            show_predictions(df_processed, model_package)
        elif page == "📈 Performance":
            show_performance(model_package)
    else:
        st.info("👆 Upload a CSV file to begin")
        st.markdown("""
        ### 📋 Required Format
        - `timestamp`: Date/time column
        - `sensor_00` to `sensor_51`: Sensor readings
        - `machine_status`: Status (NORMAL, BROKEN, RECOVERING)
        """)

def show_overview(df):
    """Overview page with key metrics"""
    st.header("📊 Data Overview")
    
    # Calculate dynamic statistics
    total_records = len(df)
    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col != 'sensor_15']
    n_sensors = len(sensor_cols)
    
    # Class distribution
    normal_count = (df['labels'] == 1).sum()
    broken_count = (df['labels'] == 0).sum()
    normal_pct = normal_count / total_records * 100
    broken_pct = broken_count / total_records * 100
    
    # Time range
    if 'timestamp' in df.columns:
        time_range = pd.to_datetime(df['timestamp'])
        duration_days = (time_range.max() - time_range.min()).days
    else:
        duration_days = "N/A"
    
    # Missing values
    missing_total = df[sensor_cols].isna().sum().sum()
    missing_pct = missing_total / (len(df) * n_sensors) * 100
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📝 Total Records", f"{total_records:,}")
    with col2:
        st.metric("🔌 Sensors", f"{n_sensors}")
    with col3:
        st.metric("✅ Normal", f"{normal_count:,}", 
                 delta=f"{normal_pct:.1f}%")
    with col4:
        st.metric("❌ Broken", f"{broken_count:,}", 
                 delta=f"{broken_pct:.1f}%", delta_color="inverse")
    with col5:
        st.metric("⏱️ Duration", f"{duration_days} days" if duration_days != "N/A" else "N/A")
    
    st.markdown("---")
    
    # Timeline
    st.subheader("⏱️ Status Timeline")
    fig, stats = plot_status_timeline(df)
    st.pyplot(fig)
    
    # Imbalance warning
    imbalance_ratio = max(normal_pct, broken_pct) / min(normal_pct, broken_pct)
    if imbalance_ratio > 3:
        st.warning(f"⚠️ Class imbalance detected: {imbalance_ratio:.1f}:1 ratio")
    
    st.markdown("---")
    
    # Missing values analysis
    st.subheader("❓ Data Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Missing Values", f"{missing_total:,}", 
                 delta=f"{missing_pct:.2f}%", delta_color="inverse")
        
        # Top sensors with missing values
        missing_by_sensor = df[sensor_cols].isna().sum().sort_values(ascending=False).head(5)
        if missing_by_sensor.sum() > 0:
            st.write("**Top 5 Sensors with Missing Values:**")
            for sensor, count in missing_by_sensor.items():
                pct = count / len(df) * 100
                st.write(f"- `{sensor}`: {count:,} ({pct:.1f}%)")
    
    with col2:
        # Sensor value ranges
        st.write("**Sensor Value Ranges:**")
        sample_sensors = sensor_cols[:5]
        for sensor in sample_sensors:
            min_val = df[sensor].min()
            max_val = df[sensor].max()
            mean_val = df[sensor].mean()
            st.write(f"- `{sensor}`: [{min_val:.2f}, {max_val:.2f}] (μ={mean_val:.2f})")

def show_sensor_analysis(df):
    """Sensor analysis page"""
    st.header("🔍 Sensor Analysis")
    
    sensor_cols = [col for col in df.columns if col.startswith('sensor_') and col != 'sensor_15']
    
    # Calculate separation scores for all sensors
    st.subheader("📊 Sensor Effectiveness Ranking")
    
    with st.spinner("Calculating sensor separation scores..."):
        sensor_scores = []
        
        for sensor in sensor_cols:
            normal = df[df['labels'] == 1][sensor].dropna()
            broken = df[df['labels'] == 0][sensor].dropna()
            
            if len(normal) > 0 and len(broken) > 0:
                mean_diff = abs(normal.mean() - broken.mean())
                std_pooled = np.sqrt((normal.std()**2 + broken.std()**2) / 2)
                separation = mean_diff / std_pooled if std_pooled > 0 else 0
                
                sensor_scores.append({
                    'Sensor': sensor,
                    'Separation Score': separation,
                    'Normal Mean': normal.mean(),
                    'Broken Mean': broken.mean(),
                    'Difference': mean_diff,
                    'Missing %': df[sensor].isna().sum() / len(df) * 100
                })
        
        scores_df = pd.DataFrame(sensor_scores).sort_values('Separation Score', ascending=False)
    
    # Display top sensors
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.metric("Most Effective Sensor", 
                 scores_df.iloc[0]['Sensor'], 
                 delta=f"Score: {scores_df.iloc[0]['Separation Score']:.2f}")
        
        st.write("**Top 10 Sensors:**")
        st.dataframe(scores_df.head(10).style.background_gradient(
            subset=['Separation Score'], cmap='RdYlGn'
        ), use_container_width=True)
    
    with col2:
        # Plot separation scores
        fig, ax = plt.subplots(figsize=(10, 6))
        top_10 = scores_df.head(10)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_10)))
        
        bars = ax.barh(range(len(top_10)), top_10['Separation Score'], color=colors)
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['Sensor'])
        ax.set_xlabel('Separation Score', fontsize=11)
        ax.set_title('Top 10 Most Predictive Sensors', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Interactive sensor selection
    st.subheader("🎯 Detailed Sensor View")
    
    n_sensors_to_show = st.slider("Number of sensors to analyze", 3, 15, 6)
    
    selected_sensors = st.multiselect(
        "Or select specific sensors:",
        options=sensor_cols,
        default=list(scores_df.head(n_sensors_to_show)['Sensor'])
    )
    
    if not selected_sensors:
        selected_sensors = list(scores_df.head(n_sensors_to_show)['Sensor'])
    
    # Display distributions
    cols_per_row = 2
    for i in range(0, len(selected_sensors), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(selected_sensors):
                with col:
                    sensor = selected_sensors[i + j]
                    fig = plot_sensor_distribution(df, sensor)
                    st.pyplot(fig)

def show_predictions(df, model_package):
    """Predictions page"""
    st.header("🤖 Model Predictions")
    
    # Prediction controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Click the button below to generate predictions for the uploaded data.")
    
    with col2:
        predict_button = st.button("🚀 Run Predictions", type="primary", use_container_width=True)
    
    if predict_button or 'predictions' in st.session_state:
        with st.spinner("Generating predictions..."):
            # Engineer features
            features_df = engineer_features(df, model_package)
            
            # Align features
            expected_features = model_package['feature_columns']
            for feat in expected_features:
                if feat not in features_df.columns:
                    features_df[feat] = 0
            features_df = features_df[expected_features]
            
            # Normalize
            scaler = MinMaxScaler()
            features_normalized = pd.DataFrame(
                scaler.fit_transform(features_df),
                columns=features_df.columns,
                index=features_df.index
            )
            
            # Predict
            predictions = model_package['model'].predict(features_normalized)
            probabilities = model_package['model'].predict_proba(features_normalized)
            
            # Store results
            df['prediction'] = predictions
            df['failure_probability'] = probabilities[:, 0]
            st.session_state.predictions = df
        
        df = st.session_state.predictions
        
        # Calculate dynamic metrics
        total_pred = len(predictions)
        normal_pred = (predictions == 1).sum()
        broken_pred = (predictions == 0).sum()
        avg_risk = df['failure_probability'].mean()
        max_risk = df['failure_probability'].max()
        high_risk_count = (df['failure_probability'] > 0.5).sum()
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Predictions", f"{total_pred:,}")
        with col2:
            st.metric("Predicted Normal", f"{normal_pred:,}", 
                     delta=f"{normal_pred/total_pred*100:.1f}%")
        with col3:
            st.metric("Predicted Broken", f"{broken_pred:,}", 
                     delta=f"{broken_pred/total_pred*100:.1f}%", delta_color="inverse")
        with col4:
            st.metric("Avg Failure Risk", f"{avg_risk*100:.1f}%")
        with col5:
            st.metric("High Risk Periods", f"{high_risk_count:,}", 
                     delta=f"{high_risk_count/total_pred*100:.1f}%", delta_color="inverse")
        
        st.markdown("---")
        
        # Prediction timeline
        st.subheader("📈 Prediction Timeline")
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        x = df.index
        ax.plot(x, df['labels'], label='Actual', linewidth=2, color='#2ecc71', alpha=0.7)
        ax.plot(x, df['prediction'], label='Predicted', linewidth=2, 
               color='#e74c3c', linestyle='--', alpha=0.7)
        
        # Highlight misclassifications
        misclass = df['labels'] != df['prediction']
        if misclass.sum() > 0:
            ax.scatter(x[misclass], df.loc[misclass, 'labels'], 
                      color='orange', s=50, label=f'Misclassified ({misclass.sum()})', 
                      zorder=5, alpha=0.6)
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Status', fontsize=11)
        ax.set_title(f'Actual vs Predicted - Agreement: {(~misclass).sum()/len(df)*100:.1f}%', 
                    fontsize=12, fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['BROKEN', 'NORMAL'])
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Performance metrics
        if 'labels' in df.columns:
            st.subheader("📊 Performance Metrics")
            
            metrics = calculate_metrics(df['labels'], df['prediction'])
            
            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            with col2:
                st.metric("F1 Score (Macro)", f"{metrics['f1_macro']*100:.2f}%")
            with col3:
                st.metric("Precision", f"{metrics['precision']*100:.2f}%")
            with col4:
                st.metric("Recall", f"{metrics['recall']*100:.2f}%")
            
            # Class-specific metrics
            st.write("**Class-Specific Performance:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**BROKEN Class:**")
                st.metric("F1 Score", f"{metrics['f1_broken']*100:.2f}%")
                st.metric("Precision", f"{metrics['precision_broken']*100:.2f}%")
                st.metric("Recall", f"{metrics['recall_broken']*100:.2f}%")
            
            with col2:
                st.write("**NORMAL Class:**")
                st.metric("F1 Score", f"{metrics['f1_normal']*100:.2f}%")
                st.metric("Precision", f"{metrics['precision_normal']*100:.2f}%")
                st.metric("Recall", f"{metrics['recall_normal']*100:.2f}%")
            
            st.markdown("---")
            
            # Confusion matrix
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig = plot_confusion_matrix(metrics['confusion_matrix'], metrics)
                st.pyplot(fig)
            
            with col2:
                st.write("**Error Analysis:**")
                st.metric("Total Errors", f"{metrics['misclassifications']:,}", 
                         delta=f"{metrics['misclassifications']/total_pred*100:.2f}%", 
                         delta_color="inverse")
                
                cm = metrics['confusion_matrix']
                false_positives = cm[1, 0] if cm.shape[0] > 1 else 0
                false_negatives = cm[0, 1] if cm.shape[0] > 1 else 0
                
                st.write(f"**False Positives:** {false_positives:,}")
                st.write(f"- Predicted BROKEN but was NORMAL")
                st.write(f"- Cost: Unnecessary maintenance")
                
                st.write(f"**False Negatives:** {false_negatives:,}")
                st.write(f"- Predicted NORMAL but was BROKEN")
                st.write(f"- Cost: Missed failures (Critical!)")
                
                if false_negatives > false_positives:
                    st.warning("⚠️ More false negatives than false positives - consider adjusting threshold")
        
        st.markdown("---")
        
        # High-risk analysis
        st.subheader("⚠️ High-Risk Periods")
        
        risk_threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)
        high_risk_df = df[df['failure_probability'] > risk_threshold].copy()
        
        if len(high_risk_df) > 0:
            st.warning(f"Found {len(high_risk_df):,} time steps above {risk_threshold*100:.0f}% risk threshold")
            
            # Risk distribution
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df['failure_probability'], linewidth=1, color='#3498db', alpha=0.7)
            ax.fill_between(df.index, df['failure_probability'], 
                           where=(df['failure_probability'] > risk_threshold),
                           color='#e74c3c', alpha=0.5, label=f'High Risk (>{risk_threshold*100:.0f}%)')
            ax.axhline(y=risk_threshold, color='red', linestyle='--', label='Threshold')
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel('Failure Probability', fontsize=11)
            ax.set_title(f'Failure Risk Over Time - {len(high_risk_df)} High-Risk Periods', 
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Show high-risk records
            display_cols = ['timestamp'] if 'timestamp' in high_risk_df.columns else []
            display_cols += ['failure_probability', 'prediction', 'labels']
            
            st.dataframe(
                high_risk_df[display_cols].head(20).style.background_gradient(
                    subset=['failure_probability'], cmap='Reds'
                ),
                use_container_width=True
            )
            
            # Download predictions
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv,
                file_name="pump_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.success(f"✅ No periods above {risk_threshold*100:.0f}% risk threshold")

def show_performance(model_package):
    """Performance and feature importance page"""
    st.header("📈 Model Performance & Feature Importance")
    
    model = model_package['model']
    feature_names = model_package['feature_columns']
    importances = model.feature_importances_
    
    # Feature importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Overall statistics
    total_features = len(feature_names)
    top_10_importance = importance_df.head(10)['Importance'].sum()
    top_20_importance = importance_df.head(20)['Importance'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Features", total_features)
    with col2:
        st.metric("Top 10 Contribution", f"{top_10_importance*100:.1f}%")
    with col3:
        st.metric("Top 20 Contribution", f"{top_20_importance*100:.1f}%")
    with col4:
        most_important = importance_df.iloc[0]
        st.metric("Most Important", 
                 most_important['Feature'].replace('_deviation', ''),
                 delta=f"{most_important['Importance']*100:.2f}%")
    
    st.markdown("---")
    
    # Feature importance visualization
    st.subheader("🏆 Top Feature Importance")
    
    n_features = st.slider("Number of top features to display", 5, 30, 15)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.dataframe(
            importance_df.head(n_features).style.background_gradient(
                subset=['Importance'], cmap='viridis'
            ),
            use_container_width=True
        )
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = importance_df.head(n_features)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_n)))
        
        bars = ax.barh(range(len(top_n)), top_n['Importance'], color=colors)
        ax.set_yticks(range(len(top_n)))
        ax.set_yticklabels([f.replace('_deviation', '') for f in top_n['Feature']], fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title(f'Top {n_features} Most Important Features', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Cumulative importance
    st.subheader("📊 Cumulative Feature Importance")
    
    importance_df['Cumulative'] = importance_df['Importance'].cumsum()
    
    # Find thresholds
    features_80 = (importance_df['Cumulative'] <= 0.8).sum()
    features_90 = (importance_df['Cumulative'] <= 0.9).sum()
    features_95 = (importance_df['Cumulative'] <= 0.95).sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Features for 80%", features_80,
                 delta=f"{features_80/total_features*100:.1f}% of features")
    with col2:
        st.metric("Features for 90%", features_90,
                 delta=f"{features_90/total_features*100:.1f}% of features")
    with col3:
        st.metric("Features for 95%", features_95,
                 delta=f"{features_95/total_features*100:.1f}% of features")
    
    # Plot cumulative importance
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x_vals = range(len(importance_df))
    ax.plot(x_vals, importance_df['Cumulative'], linewidth=2, color='#e74c3c')
    ax.fill_between(x_vals, importance_df['Cumulative'], alpha=0.3, color='#e74c3c')
    
    # Add threshold lines
    ax.axhline(y=0.8, color='green', linestyle='--', 
              label=f'80% ({features_80} features)', linewidth=1.5)
    ax.axhline(y=0.9, color='orange', linestyle='--', 
              label=f'90% ({features_90} features)', linewidth=1.5)
    ax.axhline(y=0.95, color='red', linestyle='--', 
              label=f'95% ({features_95} features)', linewidth=1.5)
    
    # Add vertical lines
    ax.axvline(x=features_80, color='green', linestyle=':', alpha=0.5)
    ax.axvline(x=features_90, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(x=features_95, color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Number of Features', fontsize=11)
    ax.set_ylabel('Cumulative Importance', fontsize=11)
    ax.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Maintenance priority recommendations
    st.subheader("🎯 Sensor Monitoring Priority")
    
    # Categorize by importance
    tier1 = importance_df.head(5)
    tier2 = importance_df.iloc[5:12]
    tier3 = importance_df.iloc[12:20]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔴 Critical Priority")
        st.write(f"**Monitor continuously** ({len(tier1)} sensors)")
        st.metric("Combined Importance", f"{tier1['Importance'].sum()*100:.1f}%")
        for idx, row in tier1.iterrows():
            sensor = row['Feature'].replace('_deviation', '')
            st.write(f"• `{sensor}` ({row['Importance']*100:.2f}%)")
    
    with col2:
        st.markdown("### 🟡 High Priority")
        st.write(f"**Regular checks** ({len(tier2)} sensors)")
        st.metric("Combined Importance", f"{tier2['Importance'].sum()*100:.1f}%")
        for idx, row in tier2.iterrows():
            sensor = row['Feature'].replace('_deviation', '')
            st.write(f"• `{sensor}` ({row['Importance']*100:.2f}%)")
    
    with col3:
        st.markdown("### 🟢 Medium Priority")
        st.write(f"**Periodic inspection** ({len(tier3)} sensors)")
        st.metric("Combined Importance", f"{tier3['Importance'].sum()*100:.1f}%")
        for idx, row in tier3.head(5).iterrows():
            sensor = row['Feature'].replace('_deviation', '')
            st.write(f"• `{sensor}` ({row['Importance']*100:.2f}%)")
        if len(tier3) > 5:
            st.write(f"*...and {len(tier3)-5} more*")
    
    # Download feature importance
    csv = importance_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Feature Importance Data",
        data=csv,
        file_name="feature_importance.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Decision-making insights
    st.subheader("💡 Actionable Insights")
    
    st.markdown(f"""
    ### Key Findings:
    
    1. **Model Efficiency**: Only **{features_80} sensors** ({features_80/total_features*100:.1f}%) 
       contribute to 80% of prediction accuracy
       - **Action**: Focus maintenance resources on these critical sensors
    
    2. **Sensor Redundancy**: {total_features - features_95} sensors contribute less than 5% 
       to predictions
       - **Action**: Consider reducing monitoring costs for low-importance sensors
    
    3. **Top Sensor**: `{importance_df.iloc[0]['Feature'].replace('_deviation', '')}` alone 
       provides {importance_df.iloc[0]['Importance']*100:.2f}% of predictive power
       - **Action**: This sensor should never fail; implement redundant monitoring
    
    4. **Cost Optimization**: Monitoring top {features_90} sensors provides 90% accuracy
       - **Potential Savings**: Could reduce monitoring infrastructure by {(1 - features_90/total_features)*100:.1f}%
    
    5. **Failure Detection**: With current model:
       - Top 10 sensors provide {top_10_importance*100:.1f}% coverage
       - **Recommendation**: Implement real-time alerts on these sensors
    """)

if __name__ == "__main__":
    main()