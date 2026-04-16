import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# --- CUSTOM CSS FOR ANIMATIONS ---
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        animation: fadeIn 1.5s ease-out;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- HEADER ---
st.markdown('<h1 class="main-header">Experience-Based Predictor</h1>', unsafe_allow_html=True)
st.write("---")

# --- INPUT SECTION ---
st.subheader("Enter Details")
col1, col2 = st.columns([2, 1])

with col1:
    years_exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
    
with col2:
    st.info("Input your total professional experience to see the predicted outcome.")

# --- PREDICTION LOGIC ---
if st.button("Predict Result"):
    with st.spinner('Calculating using SVR Model...'):
        time.sleep(1) # Visual delay for effect
        
        # Prepare data for model
        features = np.array([[years_exp]])
        prediction = model.predict(features)
        
        # Display Result
        st.balloons()
        st.success(f"### Prediction Complete!")
        
        # Using a nice metric display
        st.metric(label="Predicted Value", value=f"{prediction[0]:,.2f}")
        
        # Data visualization
        st.write("#### Insights")
        chart_data = pd.DataFrame({
            'Experience': [years_exp],
            'Prediction': [prediction[0]]
        })
        st.bar_chart(chart_data)

# --- FOOTER ---
st.write("---")
st.caption("Model Engine: SVR (RBF Kernel) | Scikit-Learn v1.6.1")
