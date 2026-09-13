import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="centered"
)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("advertising_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Keep `advertising_model.pkl` and `scaler.pkl` in the same folder.")
    st.stop()

# Title
st.title("📈 Sales Prediction ")
st.markdown("Predict **Sales** from **TV** and **Radio** advertising budgets.")

st.divider()

# User inputs
st.subheader("🎯 Enter Advertising Budgets")

tv = st.number_input(
    "TV Budget",
    min_value=0.0,
    value=None,
    step=1.0,
    placeholder="Enter TV budget"
)

radio = st.number_input(
    "Radio Budget",
    min_value=0.0,
    value=None,
    step=1.0,
    placeholder="Enter Radio budget"
)

# Predict Sales
if st.button(" Predict Sales", use_container_width=True):
    if tv is None or radio is None:
        st.warning(" Please enter both TV and Radio budgets.")
    else:
        input_df = pd.DataFrame({"TV": [tv], "Radio": [radio]})
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.success(f"###  Predicted Sales: **{prediction:.2f} units**")