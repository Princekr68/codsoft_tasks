import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load model and columns
@st.cache_resource
def load_artifacts():
    model = joblib.load("titanic_model.pkl")
    feature_columns = joblib.load("features_columns.pkl")
    return model, feature_columns

try:
    model, feature_columns = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Keep `titanic_model.pkl` and `features_columns.pkl` in the same folder.")
    st.stop()

# Title
st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to predict survival.")

st.divider()

# Inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

with col2:
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=15.0, step=0.5)
    embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])

# Predict
if st.button("🔮 Predict Survival", use_container_width=True):
    row = {col: 0 for col in feature_columns}
    row['Pclass'] = pclass
    row['Sex'] = 0 if sex == 'male' else 1
    row['Age'] = age
    row['Fare'] = fare
    # SibSp and Parch default 0 (hidden from user)

    embarked_col = f'Embarked_{embarked}'
    if embarked_col in row:
        row[embarked_col] = 1

    input_df = pd.DataFrame([row])[feature_columns]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"### ✅ Survived\n**Survival Probability: {probability:.1%}**")
    else:
        st.error(f"### ❌ Not Survived\n**Survival Probability: {probability:.1%}**")