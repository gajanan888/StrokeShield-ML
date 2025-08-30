
import streamlit as st
import pandas as pd
import joblib
import time

# Load saved model, scaler, and expected columns
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# Custom CSS for animations and styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #e74c3c;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .subtitle {
        text-align: center;
        color: #34495e;
        font-size: 1.2rem;
        margin-bottom: 30px;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .input-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        animation: slideIn 0.8s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .prediction-button {
        background: linear-gradient(45deg, #e74c3c, #c0392b);
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: bounce 2s infinite;
        margin: 20px auto;
        display: block;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    .result-container {
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        animation: zoomIn 0.6s ease-out;
    }
    
    @keyframes zoomIn {
        from { transform: scale(0.8); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        box-shadow: 0 0 20px rgba(238, 90, 82, 0.4);
    }
    
    .low-risk {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        box-shadow: 0 0 20px rgba(64, 192, 87, 0.4);
    }
    
    .heart-icon {
        font-size: 3rem;
        animation: heartbeat 1.5s infinite;
    }
    
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #e74c3c;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .input-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .input-section:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

# Main header with animation
st.markdown('<div class="main-header">❤️ Heart Health Predictor ❤️</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">🔍 AI-Powered Heart Disease Risk Assessment by Gajanan</div>', unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Basic Information")
    age = st.slider("🎂 Age", 18, 100, 40)
    sex = st.selectbox("⚧️ Sex", ["M", "F"])
    chest_pain = st.selectbox("💔 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 🫀 Heart Metrics")
    max_hr = st.slider("💓 Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("📈 Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("📊 ST Slope", ["Up", "Flat", "Down"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 🩺 Medical Readings")
    resting_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("🧪 Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("🍯 Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("📋 Resting ECG", ["Normal", "ST", "LVH"])
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction button with custom styling
st.markdown('<div style="text-align: center; margin: 30px 0;">', unsafe_allow_html=True)

if st.button("🔮 Predict Heart Disease Risk", type="primary"):
    # Show loading animation
    with st.spinner('🧠 AI is analyzing your health data...'):
        time.sleep(2)  # Simulate processing time
        
        # Create a raw input dictionary
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

        # Show result with animation
        if prediction == 1:
            st.markdown("""
            <div class="result-container high-risk">
                <div class="heart-icon">💔</div>
                <h2>⚠️ High Risk of Heart Disease</h2>
                <p>Please consult with a healthcare professional for proper evaluation and guidance.</p>
                <p><strong>Remember:</strong> This is an AI prediction and should not replace professional medical advice.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-container low-risk">
                <div class="heart-icon">💚</div>
                <h2>✅ Low Risk of Heart Disease</h2>
                <p>Great news! Your inputs suggest a lower risk of heart disease.</p>
                <p><strong>Keep it up:</strong> Maintain a healthy lifestyle and regular check-ups!</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Add footer information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin-top: 30px;">
    <p>🤖 <strong>Powered by Machine Learning</strong> | Built with ❤️ using Streamlit</p>
    <p><em>Disclaimer: This tool is for educational purposes only and should not replace professional medical advice.</em></p>
</div>
""", unsafe_allow_html=True)
