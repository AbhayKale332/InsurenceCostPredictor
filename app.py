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

# 2. Sidebar for metadata, theme toggle, and links
with st.sidebar:
    st.header("About Project")
    st.markdown("This tool predicts insurance costs based on personal health data.")
    
    st.markdown("---")
    theme = st.radio("🎨 UI Theme", ["System Default", "Light Mode", "Dark Mode"], index=0)
    
    st.markdown("---")
    st.markdown("""
    <a href="https://github.com/akashch1512/InsurenceCostPredictor" target="_blank" style="text-decoration:none;">
        <button style="
            background-color:#24292e;
            color:white;
            width: 100%;
            padding:10px 16px;
            border:none;
            border-radius:8px;
            cursor: pointer;
            transition: 0.3s;
            font-weight: 600;">
            View Source on GitHub
        </button>
    </a>
    """, unsafe_allow_html=True)
    st.caption("Built with Streamlit & Scikit-learn")

# Apply Theme and Clean UI Customizations
if theme == "Light Mode":
    theme_css = """
        .stApp { background-color: #ffffff; }
        .stMarkdown, .stText, .stTitle, h1, h2, h3, h4, h5, h6, p, label, .stMetricValue, .stMetricLabel { color: #1e1e1e !important; }
        [data-testid="stSidebar"] { background-color: #f4f6f9; }
    """
elif theme == "Dark Mode":
    theme_css = """
        .stApp { background-color: #121212; }
        .stMarkdown, .stText, .stTitle, h1, h2, h3, h4, h5, h6, p, label, .stMetricValue, .stMetricLabel { color: #f0f0f0 !important; }
        [data-testid="stSidebar"] { background-color: #1a1a1a; }
    """
else:
    theme_css = ""

st.markdown(f"""
    <style>
    {theme_css}
    /* Clean and Modern UI */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    .stAppDeployButton {{
        display: none !important;
    }}
    .stButton > button {{
        border-radius: 8px;
        transition: 0.3s;
        font-weight: 600;
        border: 1px solid rgba(128,128,128,0.2);
    }}
    .stButton > button:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 2.3rem;
        font-weight: 700;
        color: #1f77b4 !important;
    }}
    </style>
""", unsafe_allow_html=True)

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
    
    # Improved Male/Female selectbox with signs and colors
    st.markdown("<b>Sex</b>", unsafe_allow_html=True)
    sex = st.selectbox(
        "Sex", 
        ["male", "female"],
        format_func=lambda x: "🟦 👨 Male (♂)" if x == "male" else "🟥 👩 Female (♀)",
        label_visibility="collapsed"
    )
    
    bmi = st.slider("BMI", 10.0, 50.0, 25.0, help="Body Mass Index")
    
    # We will compute the fallback text in Python
    if bmi < 18.5:
        fallback_html = "🔵 Underweight (BMI < 18.5)"
        fallback_bg = "#cce5ff"
        fallback_color = "#004085"
    elif bmi < 25.0:
        fallback_html = "🟢 Normal Weight (18.5 - 24.9)"
        fallback_bg = "#d4edda"
        fallback_color = "#155724"
    elif bmi < 30.0:
        fallback_html = "🟡 Overweight (25.0 - 29.9)"
        fallback_bg = "#fff3cd"
        fallback_color = "#856404"
    elif bmi < 35.0:
        fallback_html = "🟠 Obese Class I (30.0 - 34.9)"
        fallback_bg = "#ffeeba"
        fallback_color = "#856404"
    elif bmi < 40.0:
        fallback_html = "🔴 Obese Class II (35.0 - 39.9)"
        fallback_bg = "#f8d7da"
        fallback_color = "#721c24"
    else:
        fallback_html = "🟤 Obese Class III (Morbid/Severe)"
        fallback_bg = "#f5c6cb"
        fallback_color = "#721c24"
        
    st.markdown(f"""
        <div id="fluid-bmi-category" style="
            padding: 12px; 
            border-radius: 8px; 
            font-weight: bold; 
            margin-top: -10px; 
            margin-bottom: 10px;
            background-color: {fallback_bg};
            color: {fallback_color};
            border: 1px solid {fallback_color};
            transition: all 0.3s;
        ">
            {fallback_html}
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Policy Details")
    children = st.number_input("Number of Children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ["no", "yes"], help="Do you smoke cigarettes regularly?")
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.markdown("<br>", unsafe_allow_html=True) # Add some spacing

# 5. Prediction Logic
if st.button("Calculate Premium", type="primary", use_container_width=True):
    
    if model:
        # Custom elegant loading animation
        loader_html = """
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 40px;">
                <div style="width: 50px; height: 50px; border-radius: 50%; border: 4px solid rgba(31, 119, 180, 0.2); border-top-color: #1f77b4; animation: spin 1s linear infinite;"></div>
                <p style="margin-top: 15px; font-weight: 600; font-size: 1.1rem; color: #1f77b4;">Analyzing Health Profile...</p>
            </div>
            <style>
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            </style>
        """
        loader = st.empty()
        loader.markdown(loader_html, unsafe_allow_html=True)
        time.sleep(1.5)
        loader.empty()
        
        if True: # Retain indentation for the block below

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

# Javascript for Fluid BMI Slider Update
import streamlit.components.v1 as components
components.html("""
<script>
    const doc = window.parent.document;
    
    function updateCategory() {
        if(!doc) return;
        const labels = doc.querySelectorAll('div[data-testid="stSlider"]');
        let bmiValue = null;
        
        labels.forEach((widget) => {
            const label = widget.querySelector('label');
            if (label && label.innerText.includes("BMI")) {
                const thumb = widget.querySelector('[data-testid="stThumbValue"]');
                if(thumb) {
                    bmiValue = parseFloat(thumb.innerText);
                }
            }
        });

        const displays = doc.querySelectorAll("#fluid-bmi-category");
        if (displays.length > 0 && bmiValue !== null) {
            const display = displays[displays.length - 1]; 
            let text = ""; let bg = ""; let color = "";
            
            if (bmiValue < 18.5) {
                text = "🔵 Underweight (BMI < 18.5)"; bg = "#cce5ff"; color = "#004085";
            } else if (bmiValue >= 18.5 && bmiValue < 25.0) {
                text = "🟢 Normal Weight (18.5 - 24.9)"; bg = "#d4edda"; color = "#155724";
            } else if (bmiValue >= 25.0 && bmiValue < 30.0) {
                text = "🟡 Overweight (25.0 - 29.9)"; bg = "#fff3cd"; color = "#856404";
            } else if (bmiValue >= 30.0 && bmiValue < 35.0) {
                text = "🟠 Obese Class I (30.0 - 34.9)"; bg = "#ffeeba"; color = "#856404";
            } else if (bmiValue >= 35.0 && bmiValue < 40.0) {
                text = "🔴 Obese Class II (35.0 - 39.9)"; bg = "#f8d7da"; color = "#721c24";
            } else {
                text = "🟤 Obese Class III (Morbid/Severe)"; bg = "#f5c6cb"; color = "#721c24";
            }
            
            if (display.innerHTML !== text) {
                display.innerHTML = text;
                display.style.backgroundColor = bg;
                display.style.color = color;
                display.style.border = `1px solid ${color}`;
            }
        }
    }
    
    // Run observer periodically for fluid feeling
    setInterval(updateCategory, 50);
</script>
""", height=0, width=0)