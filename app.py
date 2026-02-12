import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.markdown("""
<a href="https://github.com/yourusername/your-repo-name" target="_blank">
    <button style="background-color:#24292e;color:white;padding:8px 16px;border:none;border-radius:5px;">
        View Source Code on GitHub
    </button>
</a>
""", unsafe_allow_html=True)


# Load model
model = pickle.load(open("./TrainedModels/insurance_model.pkl", "rb"))
model_columns = pickle.load(open("./TrainedModels/model_columns.pkl", "rb"))

st.title("Akash's Insurance Cost Predictor")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Insurance Cost"):

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

    # Encode region (same as training with drop_first)
    if f"region_{region}" in input_data.columns:
        input_data[f"region_{region}"] = 1

    # Predict (log scale)
    log_prediction = model.predict(input_data)

    # Convert back
    prediction = np.expm1(log_prediction)

    st.success(f"Predicted Insurance Cost: ₹ {prediction[0]:,.2f}")
