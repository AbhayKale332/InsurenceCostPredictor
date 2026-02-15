import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

version = "1.0"

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="Insurance Predictor",
    page_icon="🏥",
    layout="centered"
)

# 2. Sidebar for metadata and links
with st.sidebar:
    st.header("About Project")
    st.markdown("This tool predicts insurance costs based on personal health data.")
    st.markdown("---")
    st.markdown("""
    <a href="https://github.com/akashch1512/InsurenceCostPredictor" target="_blank">
        <button style="
            background-color:#24292e;
            color:white;
            width: 100%;
            padding:8px 16px;
            border:none;
            border-radius:5px;
            cursor: pointer;
            font-weight: bold;">
            View Source on GitHub
        </button>
    </a>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-learn")

# 3. Load model (Cached to prevent reloading on every interaction)
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open(f"./TrainedModels/v{version}/insurance_model.pkl", "rb"))
        cols = pickle.load(open(f"./TrainedModels/v{version}/model_columns.pkl", "rb"))
        return model, cols
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the 'TrainedModels' folder exists.")
        return None, None

model, model_columns = load_model()

# 4. Main App UI
st.title("🏥 Insurance Cost Predictor")
st.markdown("Enter your details below to get an estimated premium calculation.")

st.divider() # Visual separation

# Input Section - Organized into Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")
    age = st.slider("Age", 18, 100, 30, help="Age of the primary beneficiary")
    sex = st.radio("Sex", ["male", "female"], horizontal=True)
    bmi = st.slider("BMI", 10.0, 50.0, 25.0, help="Body Mass Index")

with col2:
    st.subheader("Policy Details")
    children = st.number_input("Number of Children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["no", "yes"], help="Do you smoke cigarettes regularly?")
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.markdown("<br>", unsafe_allow_html=True) # Add some spacing

# 5. Prediction Logic
if st.button("Calculate Premium", type="primary", use_container_width=True):
    
    if model:
        with st.spinner("Analyzing profile..."):
            # Simulate a brief delay for UX (optional, remove in production if unwanted)

            # Create dataframe with zeros
            input_data = pd.DataFrame(columns=model_columns)
            input_data.loc[0] = 0

            # Fill numeric
            input_data["age"] = age
            input_data["bmi"] = bmi
            input_data["children"] = children

            # Encode binary
            input_data["sex"] = 1 if sex == "female" else 0
            input_data["smoker"] = 1 if smoker == "yes" else 0

            # Encode region
            if f"region_{region}" in input_data.columns:
                input_data[f"region_{region}"] = 1

            try:
                # Predict (log scale)
                log_prediction = model.predict(input_data)
                
                # Convert back
                prediction = np.expm1(log_prediction)
                
                # 6. Display Result using Metric
                st.markdown("---")
                st.markdown("### 📋 Prediction Result")
                
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                     st.metric(
                        label="Estimated Cost", 
                        value=f"₹ {prediction[0]:,.2f}",
                        delta="Yearly Premium"
                    )
                
                with result_col2:
                    st.info("Note: This is an AI-generated estimate based on the data provided and should not be considered a final quote from an insurance provider.")

            except Exception as e:
                st.error(f"Error during prediction: {e}")