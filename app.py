import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

# Load model, scaler, and feature columns
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# Streamlit setup
st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📊 Telco Customer Churn Prediction App (Random Forest Only)")

st.subheader("📥 Enter Customer Details:")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72, value=12, step=1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=750.0, value=70.0, step=0.1)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0, step=10.0)

# Encode inputs
input_dict = {
    'gender': 1 if gender == "Male" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == "Yes" else 0,
    'MultipleLines': 1 if multiple_lines == "Yes" else 0,
    'OnlineSecurity': 1 if online_security == "Yes" else 0,
    'OnlineBackup': 1 if online_backup == "Yes" else 0,
    'DeviceProtection': 1 if device_protection == "Yes" else 0,
    'TechSupport': 1 if tech_support == "Yes" else 0,
    'StreamingTV': 1 if streaming_tv == "Yes" else 0,
    'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
}

# Create dataframe and align columns
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Scale input
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=model_columns)

# Predict
if st.button("Predict Churn"):
    prediction = rf_model.predict(input_scaled)
    probability = rf_model.predict_proba(input_scaled)[0][1]

    st.subheader("🔎 Prediction Result:")
    st.success(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.info(f"📈 Probability of Churn: **{probability:.2%}**")

    # SHAP Explanation
    st.subheader("🔍 Why this prediction?")
    explainer = shap.Explainer(rf_model, input_scaled)
    shap_values = explainer(input_scaled)

    # Handle multi-output format
    shap_single = shap_values[0]
    if hasattr(shap_single, "values") and len(shap_single.values.shape) == 2:
        shap_single.values = shap_single.values[:, 1]
        shap_single.base_values = shap_single.base_values[1]

    # Waterfall plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_single, max_display=10, show=False)
    st.pyplot(fig)
