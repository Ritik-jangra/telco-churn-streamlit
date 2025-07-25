import streamlit as st
import pandas as pd
import joblib

# Load models and scaler
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")
svm_model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")  # List of columns used during training

# Page config
st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📊 Telco Customer Churn Prediction App")

# Sidebar model selection
model_choice = st.sidebar.selectbox("Choose ML Model", ("Random Forest", "Logistic Regression", "SVM"))

st.subheader("📥 Enter Customer Details:")

# --- Input form ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
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
monthly_charges = st.slider("Monthly Charges", 0.0, 750.0, 70.0, step=0.1)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1500.0, step=10.0)

# --- Dataframe construction ---
input_dict = {
    'gender': gender,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])

# One-hot encoding
input_encoded = pd.get_dummies(input_df)

# Align columns with training columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Scale input
input_scaled = pd.DataFrame(scaler.transform(input_encoded), columns=model_columns)

# Predict
if st.button("Predict Churn"):
    if model_choice == "Random Forest":
        model = rf_model
    elif model_choice == "Logistic Regression":
        model = lr_model
    else:
        model = svm_model

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]  # probability of churn (class 1)

    st.subheader("🔎 Prediction Result:")
    st.success(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    st.info(f"📈 Probability of Churn: **{probability:.2%}**")
