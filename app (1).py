import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("Predict the likelihood of a customer leaving based on their service profile.")

# ADVANCED ERROR HANDLING: This pinpoints exact loading issues
try:
    model = joblib.load('telco_churn_model.pkl')
except Exception as e:
    st.error(f"CRITICAL SYSTEM ERROR: Could not load the model file.")
    st.error(f"Error Details: {e}")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.header("Customer Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    senior_citizen = st.selectbox("Senior Citizen?", [0, 1])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=125.0, value=50.0)
    total_charges = tenure * monthly_charges 
    st.info(f"Estimated Total Charges: ${total_charges:.2f}")

with col2:
    st.header("Service & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    if internet_service == "No":
        tech_support = "No internet service"
        online_security = "No internet service"
        st.write("Tech Support: N/A")
        st.write("Online Security: N/A")
    else:
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'InternetService': [internet_service],
    'TechSupport': [tech_support],
    'OnlineSecurity': [online_security],
    'PaymentMethod': [payment_method],
    'PaperlessBilling': [paperless_billing],
    'SeniorCitizen': [senior_citizen]
})

st.markdown("---")

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    st.subheader("Prediction Output")
    
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability: {prediction_proba:.1%})")
    else:
        st.success(f"✅ Low Risk of Churn. (Probability: {prediction_proba:.1%})")
