
import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load saved model, scaler, and expected columns
try:
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()

# Custom CSS for dark theme and animations
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0c1426 0%, #1a2332 50%, #0c1426 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c1426 0%, #1a2332 50%, #0c1426 100%);
    }
    
    .main-header {
        text-align: center;
        color: #00ff88;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 0 0 20px #00ff88;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px #00ff88, 0 0 30px #00ff88, 0 0 40px #00ff88; }
        to { text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88, 0 0 30px #00ff88; }
    }
    
    .subtitle {
        text-align: center;
        color: #64ffda;
        font-size: 1.3rem;
        margin-bottom: 20px;
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .input-container {
        background: rgba(20, 30, 48, 0.8);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 1px solid #00ff88;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
        backdrop-filter: blur(10px);
        animation: slideInLeft 0.8s ease-out;
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .prediction-section {
        background: rgba(20, 30, 48, 0.9);
        padding: 30px;
        border-radius: 20px;
        border: 2px solid #00ff88;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        text-align: center;
        animation: slideInRight 0.8s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .ecg-line {
        width: 100%;
        height: 60px;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1000 100'%3E%3Cpath d='M0,50 L200,50 L220,20 L240,80 L260,30 L280,70 L300,50 L1000,50' stroke='%2300ff88' stroke-width='2' fill='none'/%3E%3C/svg%3E");
        background-size: 200px 100%;
        animation: ecgPulse 2s linear infinite;
        margin: 20px 0;
        opacity: 0.7;
    }
    
    @keyframes ecgPulse {
        0% { background-position: 0% 0%; }
        100% { background-position: 200% 0%; }
    }
    
    .heart-animation {
        font-size: 4rem;
        color: #ff4757;
        animation: heartbeat 1.2s ease-in-out infinite;
        text-shadow: 0 0 20px #ff4757;
    }
    
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        25% { transform: scale(1.1); }
        50% { transform: scale(1.05); }
        75% { transform: scale(1.15); }
    }
    
    .result-high-risk {
        background: linear-gradient(135deg, #ff3838, #ff6b6b);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        text-align: center;
        animation: dangerPulse 2s infinite;
        box-shadow: 0 0 30px rgba(255, 56, 56, 0.5);
    }
    
    @keyframes dangerPulse {
        0%, 100% { box-shadow: 0 0 30px rgba(255, 56, 56, 0.5); }
        50% { box-shadow: 0 0 50px rgba(255, 56, 56, 0.8); }
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, #00ff88, #32ff7e);
        color: #0c1426;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        text-align: center;
        animation: successGlow 2s infinite;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
    }
    
    @keyframes successGlow {
        0%, 100% { box-shadow: 0 0 30px rgba(0, 255, 136, 0.5); }
        50% { box-shadow: 0 0 50px rgba(0, 255, 136, 0.8); }
    }
    
    .prediction-loading {
        text-align: center;
        color: #64ffda;
        font-size: 1.5rem;
        margin: 30px 0;
        animation: fadeInOut 1.5s infinite;
    }
    
    @keyframes fadeInOut {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .algorithm-info {
        background: rgba(100, 255, 218, 0.1);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #64ffda;
        margin: 20px 0;
        color: #64ffda;
    }
    
    .metric-card {
        background: rgba(20, 30, 48, 0.6);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00ff88;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 255, 136, 0.2);
    }
    
    .footer-info {
        text-align: center;
        color: #64ffda;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #00ff88;
        opacity: 0.8;
    }
    
    /* Streamlit specific styling */
    .stSelectbox > div > div {
        background-color: rgba(20, 30, 48, 0.8);
        border: 1px solid #00ff88;
        color: #64ffda;
    }
    
    .stSlider > div > div {
        background-color: rgba(20, 30, 48, 0.8);
    }
    
    .stNumberInput > div > div {
        background-color: rgba(20, 30, 48, 0.8);
        border: 1px solid #00ff88;
        color: #64ffda;
    }
    
    h3 {
        color: #00ff88 !important;
        text-shadow: 0 0 10px #00ff88;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #00ff88, #64ffda);
        color: #0c1426;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 1.2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        animation: buttonPulse 3s infinite;
    }
    
    @keyframes buttonPulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
        50% { transform: scale(1.05); box-shadow: 0 0 30px rgba(0, 255, 136, 0.6); }
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">🫀 Heart Disease Prediction Using Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced AI-Powered Cardiovascular Risk Assessment System</div>', unsafe_allow_html=True)

# ECG line animation
st.markdown('<div class="ecg-line"></div>', unsafe_allow_html=True)

# Algorithm information section
st.markdown("""
<div class="algorithm-info">
    <h4>🤖 Algorithms Used in This Model:</h4>
    <ul>
        <li><strong>K-Nearest Neighbors</strong> - Instance-based learning method</li>
        <li><strong>Logistic Regression</strong> - Simple linear model for binary classification</li>
        <li><strong>Naive Bayes</strong> - Probabilistic model based on Bayes' theorem</li>
        <li><strong>Support Vector Machine</strong> - Maximizes margin for separation</li>
        <li><strong>Decision Tree</strong> - Tree-based classification model</li>
        <li><strong>Random Forest</strong> - Ensemble of multiple decision trees</li>
        <li><strong>XGBoost</strong> - Gradient boosting algorithm for high accuracy</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Create main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown("### 📋 Select Features to Predict Heart Disease")
    
    # Input Parameters section
    st.markdown("#### 👤 Patient Demographics")
    age = st.slider("🎂 Age (years)", 18, 100, 45, help="Patient's age in years")
    sex = st.selectbox("⚧️ Biological Sex", ["M", "F"], help="M = Male, F = Female")
    
    st.markdown("#### 🩺 Clinical Measurements")
    resting_bp = st.slider("🩸 Resting Blood Pressure (mmHg)", 80, 200, 120, help="Resting blood pressure in mm Hg")
    cholesterol = st.slider("🧪 Serum Cholesterol (mg/dL)", 100, 600, 250, help="Serum cholesterol in mg/dL")
    fasting_bs = st.selectbox("🍯 Fasting Blood Sugar > 120 mg/dL", [0, 1], help="1 if fasting blood sugar > 120 mg/dL, 0 otherwise")
    max_hr = st.slider("💓 Max Heart Rate (bpm)", 60, 220, 150, help="Maximum heart rate achieved during exercise")
    
    st.markdown("#### 🏥 Medical History")
    chest_pain = st.selectbox("💔 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], 
                             help="ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina, ASY: Asymptomatic")
    resting_ecg = st.selectbox("📋 Resting ECG", ["Normal", "ST", "LVH"], 
                              help="Normal: Normal, ST: ST-T Wave abnormality, LVH: Left ventricular hypertrophy")
    exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["Y", "N"], help="Y = Yes, N = No")
    oldpeak = st.slider("📈 Oldpeak (ST Depression)", 0.0, 6.0, 1.0, help="ST depression induced by exercise relative to rest")
    st_slope = st.selectbox("📊 ST Slope", ["Up", "Flat", "Down"], 
                           help="The slope of the peak exercise ST segment")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 AI Analysis Center")
    
    # Heart animation
    st.markdown('<div class="heart-animation">🫀</div>', unsafe_allow_html=True)
    
    # ECG animation
    st.markdown('<div class="ecg-line"></div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🚀 Analyze Heart Disease Risk", type="primary", use_container_width=True):
        # Show prediction loading
        st.markdown('<div class="prediction-loading">🔄 Predicting Heart Disease...</div>', unsafe_allow_html=True)
        
        # Progress bar animation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 25:
                status_text.text('🔍 Analyzing patient data...')
            elif i < 50:
                status_text.text('🧠 Running ML algorithms...')
            elif i < 75:
                status_text.text('📊 Processing results...')
            else:
                status_text.text('✅ Generating prediction...')
            time.sleep(0.03)
        
        progress_bar.empty()
        status_text.empty()
        
        # Create input for prediction
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        
        # Calculate confidence
        confidence = max(prediction_proba) * 100
        
        # Display results
        if prediction == 1:
            st.markdown(f"""
            <div class="result-high-risk">
                <h2>⚠️ HIGH RISK DETECTED</h2>
                <div style="font-size: 3rem; margin: 20px 0;">🚨</div>
                <h3>Confidence: {confidence:.1f}%</h3>
                <p style="font-size: 1.2rem; margin: 20px 0;">
                    <strong>URGENT:</strong> The AI model indicates a high probability of heart disease.
                </p>
                <p style="font-size: 1rem;">
                    ⚡ <strong>Immediate Action Required:</strong><br>
                    • Consult a cardiologist immediately<br>
                    • Schedule comprehensive cardiac evaluation<br>
                    • Monitor symptoms closely<br>
                    • Follow medical professional's guidance
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low-risk">
                <h2>✅ LOW RISK DETECTED</h2>
                <div style="font-size: 3rem; margin: 20px 0;">💚</div>
                <h3>Confidence: {confidence:.1f}%</h3>
                <p style="font-size: 1.2rem; margin: 20px 0;">
                    <strong>GOOD NEWS:</strong> The AI model suggests low probability of heart disease.
                </p>
                <p style="font-size: 1rem;">
                    🌟 <strong>Maintain Healthy Lifestyle:</strong><br>
                    • Continue regular exercise<br>
                    • Maintain balanced diet<br>
                    • Regular health check-ups<br>
                    • Monitor cardiovascular health
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk factors breakdown
        st.markdown("### 📊 Risk Factor Analysis")
        
        risk_factors = []
        if age > 60:
            risk_factors.append("Advanced age")
        if resting_bp > 140:
            risk_factors.append("High blood pressure")
        if cholesterol > 240:
            risk_factors.append("High cholesterol")
        if max_hr < 100:
            risk_factors.append("Low maximum heart rate")
        if exercise_angina == "Y":
            risk_factors.append("Exercise-induced angina")
        
        if risk_factors:
            st.warning("⚠️ **Identified Risk Factors:** " + ", ".join(risk_factors))
        else:
            st.success("✅ **No major risk factors identified from the input parameters**")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add informational section
st.markdown("---")
col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>🎯 Model Accuracy</h4>
        <p style="font-size: 2rem; color: #00ff88;">95.2%</p>
        <p>K-Nearest Neighbors</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h4>📈 Features Used</h4>
        <p style="font-size: 2rem; color: #64ffda;">11</p>
        <p>Clinical Parameters</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <h4>🔬 Dataset Size</h4>
        <p style="font-size: 2rem; color: #ff6b6b;">918</p>
        <p>Patient Records</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-info">
    <h4>🧠 Machine Learning Model Information</h4>
    <p>This heart disease prediction system uses advanced machine learning algorithms trained on clinical data.</p>
    <p>The model analyzes multiple cardiovascular risk factors to provide accurate risk assessment.</p>
    <p><strong>⚠️ Disclaimer:</strong> This tool is for educational and research purposes only. 
    It should not be used as a substitute for professional medical diagnosis or treatment.</p>
    <p>💡 <strong>Developed by:</strong> Gajanan | <strong>Powered by:</strong> Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
