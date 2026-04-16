import streamlit as st
import pandas as pd
import pickle
import time
from streamlit_lottie import st_lottie
import requests

# Page Configuration
st.set_page_config(page_title="Customer Predictor", page_icon="🎯", layout="centered")

# Custom CSS for styling and animations
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# Helper function for animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

# Load the Model
@st.cache_resource
def load_model():
    with open('model (2).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# Header Section
with st.container():
    left_column, right_column = st.columns([2, 1])
    with left_column:
        st.title("🎯 Target Audience Predictor")
        st.subheader("Predicting customer behavior using K-Nearest Neighbors")
    with right_column:
        st_lottie(lottie_coding, height=150, key="coding")

st.write("---")

# User Input Section
st.write("### 👤 Enter Customer Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)

with col2:
    salary = st.number_input("Estimated Annual Salary ($)", min_value=0, value=50000, step=1000)

# Pre-processing input for the model 
# Based on the model metadata: 0 for Female, 1 for Male (typical encoding)
gender_encoded = 1 if gender == "Male" else 0

# Create DataFrame for prediction 
input_data = pd.DataFrame([[gender_encoded, age, salary]], 
                          columns=['Gender', 'Age', 'EstimatedSalary'])

# Prediction Button
if st.button("🚀 Run Prediction"):
    with st.spinner('Calculating likelihood...'):
        time.sleep(1.5)  # Aesthetic delay
        prediction = model.predict(input_data)
        
        st.write("---")
        if prediction[0] == 1:
            st.balloons()
            st.success("### ✅ Result: Likely to Purchase!")
            st.write("This customer matches the profile of your typical buyers.")
        else:
            st.warning("### ❌ Result: Unlikely to Purchase")
            st.write("This customer is less likely to engage with this specific offer.")

# Footer
st.markdown("<br><hr><center>Powered by Scikit-Learn 1.6.1 & Streamlit</center>", unsafe_allow_html=True)
