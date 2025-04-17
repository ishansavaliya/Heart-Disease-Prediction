import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration and styling
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern look with Times New Roman font and one task per row with aligned containers
st.markdown("""
<style>
    /* Overall app styling */
    .main {
        background-color: #0e1117;
        color: #f3f4f6;
        padding: 0 !important;
    }
    
    /* Font styling - Times New Roman */
    html, body, label, .stMarkdown, p, h1, h2, h3, h4, h5, h6, button, input, select {
        font-family: "Times New Roman", Times, serif !important;
    }
    
    /* Headers styling */
    h1 {
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 2.8rem;
        font-weight: bold;
    }
    
    h3 {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        margin-top: 1rem;
        width: 90%;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Instruction text styling */
    .instruction-text {
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Parameter labels */
    .stSlider label, .stSelectbox label, .stRadio label {
        font-size: 1.2rem;
        color: white;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Slider styling */
    .stSlider [data-baseweb="slider"] {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stSlider [data-baseweb="thumb"] {
        background-color: #f43f5e;
    }
    .stSlider [data-baseweb="track"] {
        background-color: #4b5563;
        height: 6px;
    }
    .stSlider [data-baseweb="track-highlight"] {
        background-color: #f43f5e;
        height: 6px;
    }
    
    /* Button styling */
    div.stButton > button {
        background-color: #f43f5e;
        color: white;
        border-radius: 5px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        border: none;
        width: 100%;
        max-width: 300px;
        margin: 2rem auto;
        font-size: 1.2rem;
        display: block;
        font-family: "Times New Roman", Times, serif !important;
    }
    div.stButton > button:hover {
        background-color: #e11d48;
    }
    
    /* Select box styling */
    .stSelectbox [data-baseweb="select"] {
        background-color: #1f2937;
        border-radius: 5px;
        border: 1px solid #4b5563;
    }
    
    /* Results styling for persisting result even after button click */
    .result-container {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 2rem auto;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        font-family: "Times New Roman", Times, serif !important;
        width: 90%;
        max-width: 600px;
    }
    .healthy {
        background-color: rgba(16, 185, 129, 0.2);
        border: 1px solid rgb(16, 185, 129);
        color: rgb(16, 185, 129);
    }
    .risk {
        background-color: rgba(50, 23, 28, 0.9) !important;
        background: rgba(50, 23, 28, 0.9) !important;
        border: 1px solid #f43f5e;
        color: #f43f5e;
    }
    
    /* Make inputs a bit wider */
    .stSlider, .stSelectbox, .stRadio {
        width: 90%;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.9rem;
        margin: 3rem auto 1rem auto;
        padding-top: 1rem;
        border-top: 1px solid #374151;
        font-family: "Times New Roman", Times, serif !important;
        width: 90%;
        max-width: 600px;
    }
    
    /* Expander styling - improved alignment to match screenshot */
    div.st-expander {
        border: 1px solid #2d3748;
        border-radius: 5px;
        background-color: #111827;
        padding: 0;
        margin: 1rem auto;
        width: 90% !important;
        max-width: 600px;
    }
    .streamlit-expanderHeader {
        font-size: 1.2rem;
        font-weight: normal;
        color: white;
        background-color: #111827 !important;
        border: none !important;
        border-radius: 0 !important;
    }
    .streamlit-expanderContent {
        background-color: #111827 !important;
        border-top: 1px solid #2d3748 !important;
        padding: 1rem 1rem 0.5rem 1rem !important;
    }
    
    /* Make recommendation text aligned with other containers */
    .recommendation-text {
        width: 90%;
        max-width: 600px;
        margin: 0 auto;
        font-size: 1.1rem;
        padding: 0.5rem 0;
    }
    
    /* For warning icon in result */
    .warning-icon {
        color: #f43f5e;
        font-size: 1.5rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    
    /* Force expander width */
    section[data-testid="stExpander"] {
        width: 90% !important;
        max-width: 600px !important;
        margin: 1rem auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
model = pickle.load(open('heart_disease_model.pkl', 'rb'))


# Define the function to predict heart disease
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    return prediction


# Define the Streamlit app
def app():
    # Header
    st.markdown('<h1>Heart Disease Prediction</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown('<p class="instruction-text">Please input the following parameters to predict the presence of heart disease:</p>', unsafe_allow_html=True)
    
    # Create vertical layout - one input per row
    age = st.slider('Age', 20, 100, 57)
    
    sex_options = [(1, 'Male'), (0, 'Female')]
    sex = st.selectbox('Sex', options=sex_options, format_func=lambda x: x[1])
    sex = sex[0]  # Extract numeric value
    
    cp_options = [
        (0, 'Typical Angina (chest pain due to reduced blood flow to heart)'), 
        (1, 'Atypical Angina (chest pain not typical of heart disease)'), 
        (2, 'Non-anginal Pain (chest pain not related to heart)'), 
        (3, 'Asymptomatic (no symptoms)')
    ]
    cp = st.selectbox('Chest Pain Type', options=cp_options, format_func=lambda x: x[1])
    cp = cp[0]  # Extract numeric value
    
    trestbps = st.slider('Resting Blood Pressure (mm Hg) (pressure when heart at rest)', 80, 200, 99)
    
    chol = st.slider('Cholesterol (mg/dL) (fat-like substance in blood)', 100, 600, 312)
    
    fbs_options = [(0, 'No'), (1, 'Yes')]
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL (high blood sugar level after fasting)', options=fbs_options, format_func=lambda x: x[1])
    fbs = fbs[0]  # Extract numeric value
    
    restecg_options = [
        (0, 'Normal (normal ECG results)'), 
        (1, 'ST-T Wave Abnormality (sign of reduced blood flow)'), 
        (2, 'Left Ventricular Hypertrophy (enlarged heart muscle)')
    ]
    restecg = st.selectbox('Resting ECG Results (electrocardiogram test at rest)', options=restecg_options, format_func=lambda x: x[1])
    restecg = restecg[0]  # Extract numeric value
    
    thalach = st.slider('Maximum Heart Rate Achieved (fastest heart beats during exercise)', 70, 220, 150)
    
    exang_options = [(0, 'No'), (1, 'Yes')]
    exang = st.selectbox('Exercise Induced Angina (chest pain during exercise)', options=exang_options, format_func=lambda x: x[1])
    exang = exang[0]  # Extract numeric value
    
    # Additional Parameters section styled to match the screenshot
    with st.expander("Additional Parameters"):
        oldpeak = st.slider('ST Depression Induced by Exercise (abnormal ECG during exercise)', 0.0, 6.0, 1.0, 0.1)
        
        slope_options = [
            (0, 'Upsloping (improving blood flow during exercise)'), 
            (1, 'Flat (no change in blood flow)'), 
            (2, 'Downsloping (worsening blood flow during exercise)')
        ]
        slope = st.selectbox('Slope of Peak Exercise ST Segment (pattern in ECG)', options=slope_options, format_func=lambda x: x[1])
        slope = slope[0]  # Extract numeric value
        
        ca_options = [
            (0, '0 (no major vessels affected)'), 
            (1, '1 (one major vessel affected)'), 
            (2, '2 (two major vessels affected)'), 
            (3, '3 (three major vessels affected)')
        ]
        ca = st.selectbox('Number of Major Vessels Colored by Flourosopy (number of blocked blood vessels)', options=ca_options, format_func=lambda x: x[1])
        ca = ca[0]  # Extract numeric value
        
        thal_options = [
            (0, 'Normal (normal blood flow)'), 
            (1, 'Fixed Defect (permanently reduced blood flow)'), 
            (2, 'Reversible Defect (temporarily reduced blood flow)'), 
            (3, 'Unknown')
        ]
        thal = st.selectbox('Thalassemia (blood disorder affecting red blood cells)', options=thal_options, format_func=lambda x: x[1])
        thal = thal[0]  # Extract numeric value
    
    # Initialize session state to store prediction result
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
        st.session_state.prediction_result = None
    
    # Center the predict button
    predict_button = st.button('Predict')
    
    # When the button is clicked or if a prediction was already made
    if predict_button:
        with st.spinner('Analyzing your information...'):
            # Add a small delay to simulate processing
            import time
            time.sleep(1)
            
            result = predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
            st.session_state.prediction_made = True
            st.session_state.prediction_result = int(result[0])
    
    # Display results if prediction has been made
    if st.session_state.prediction_made:
        if st.session_state.prediction_result == 0:
            st.markdown('<div class="result-container healthy">✅ No Heart Disease Detected</div>', unsafe_allow_html=True)
            st.markdown('<h3>Recommendation</h3>', unsafe_allow_html=True)
            st.markdown('<div class="recommendation-text">Your parameters indicate a lower risk of heart disease. Continue maintaining a healthy lifestyle with regular exercise and balanced diet.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-container risk">⚠️ Heart Disease Risk Detected!</div>', unsafe_allow_html=True)
            st.markdown('<h3>Recommendation</h3>', unsafe_allow_html=True)
            st.markdown('<div class="recommendation-text">Your parameters indicate a higher risk of heart disease. We recommend consulting a healthcare professional for a thorough evaluation.</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">This prediction tool is for informational purposes only and should not replace professional medical advice.</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    app()