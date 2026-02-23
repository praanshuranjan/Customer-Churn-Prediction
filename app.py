import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load('telco_churn_model.pkl')
except:
    st.error("Model not found. Please run the training cell first.")

st.title("Telco Customer Churn Predictor")
st.markdown("Predict the likelihood of a customer leaving based on their service profile.")

col1, col2 = st.columns(2)

# Collect inputs
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

    # Conditional logic to handle 'No internet service' categories naturally
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

# Prepare input data matching the pipeline expectations
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

    st.subheader("Key Drivers Insight")
    st.write("According to our models, the most common drivers for churn in this dataset are:")
    st.markdown("""
    * **Contract Type:** Month-to-month contracts have a significantly higher churn rate.
    * **Internet Service:** Fiber optic customers tend to churn more frequently than DSL.
    * **Tenure:** The first few months are the highest risk period.
    """)
