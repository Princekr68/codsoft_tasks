import pandas as pd
import streamlit as st
import joblib
from io import StringIO

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp { background-color: #FFFFFF; color: #1F2937; }
    .main-title { text-align: center; color: #1F2937; font-size: 34px; font-weight: 700; }
    .sub-title { text-align: center; color: #6B7280; font-size: 16px; }
    .result-fraud { background-color: #FEE2E2; color: #B91C1C; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: 700; border: 2px solid #B91C1C; }
    .result-safe { background-color: #DCFCE7; color: #15803D; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: 700; border: 2px solid #15803D; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Load Model and Scaler ----------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- Fixed Column Names ----------------
COLUMN_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------------- Page Title ----------------
st.markdown("<p class='main-title'>💳 Credit Card Fraud Detection</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload CSV file or paste transaction data below</p>", unsafe_allow_html=True)
st.divider()

# ---------------- Input Method ----------------
input_method = st.radio(
    "Choose input method:",
    ["📁 Upload CSV File", "📝 Paste Data "],
    horizontal=True
)

data = None
is_single_row = False

# ---------------- CSV Upload ----------------
if input_method == "📁 Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Drop 'Class' column if it exists
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        st.dataframe(data.head(), use_container_width=True)

# ---------------- Manual Paste ----------------
else:
    st.write("Paste only data row (Time, V1...V28, Amount values, comma separated)")
    pasted_data = st.text_area(
        "Paste your data here",
        height=150,
        placeholder="0,-1.35,-0.07,...,149.62"
    )
    if pasted_data.strip() != "":
        full_data = ",".join(COLUMN_NAMES) + "\n" + pasted_data
        data = pd.read_csv(StringIO(full_data))
        is_single_row = True

st.divider()

# ---------------- Predict ----------------
if st.button("🔍 Predict"):
    if data is None or data.shape[0] == 0:
        st.warning("Please upload a CSV file or paste data first.")
    else:
        # Normalize Amount and Time columns
        data["normAmount"] = scaler.transform(data[["Amount"]])
        data["normTime"] = scaler.transform(data[["Time"]])
        data = data.drop(["Amount", "Time"], axis=1)

        # Predict
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        # ---------------- Single Row: Just Show Result ----------------
        if is_single_row:
            pred = predictions[0]
            prob = probabilities[0] * 100

            if pred == 1:
                st.markdown(
                    f"<div class='result-fraud'>⚠️ FRAUD TRANSACTION<br><span style='font-size:16px;'>Fraud Probability: {prob:.2f}%</span></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-safe'>✅ GENUINE TRANSACTION<br><span style='font-size:16px;'>Fraud Probability: {prob:.2f}%</span></div>",
                    unsafe_allow_html=True
                )

        # ---------------- Multiple Rows: Show Summary + Separate Tables ----------------
        else:
            data["Prediction"] = ["Fraud" if p == 1 else "Genuine" for p in predictions]
            data["Fraud Probability (%)"] = (probabilities * 100).round(2)

            fraud_count = int((predictions == 1).sum())
            genuine_count = int((predictions == 0).sum())

            st.subheader("📊 Summary")
            st.write(f"⚠️ Total Fraud Transactions: **{fraud_count}**")
            st.write(f"✅ Total Genuine Transactions: **{genuine_count}**")

            st.divider()

            # Fraud Table
            fraud_data = data[data["Prediction"] == "Fraud"]
            st.subheader("⚠️ Fraud Transactions")
            if fraud_data.shape[0] > 0:
                st.dataframe(fraud_data, use_container_width=True)
            else:
                st.success("No fraud transactions found. ✅")

            st.divider()

            # Genuine Table
            genuine_data = data[data["Prediction"] == "Genuine"]
            st.subheader("✅ Genuine Transactions")
            if genuine_data.shape[0] > 0:
                st.dataframe(genuine_data, use_container_width=True)
            else:
                st.warning("No genuine transactions found.")

st.divider()
