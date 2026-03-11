import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("credit_model.pkl")

st.title("Credit Risk Prediction System")

st.write("Enter loan applicant information")

# Inputs
credit_amount = st.number_input("Credit Amount", min_value=0)
month_duration = st.number_input("Loan Duration (Months)", min_value=1)
age = st.number_input("Age", min_value=18)

# Simple demo inputs (since dataset has many features)
status_account = st.slider("Account Status Score", 0, 3)
credit_history = st.slider("Credit History Score", 0, 4)

if st.button("Predict Credit Risk"):

    # Create feature array (must match training feature order)
    features = np.array([[status_account,
                          month_duration,
                          credit_history,
                          0,   # purpose placeholder
                          credit_amount,
                          0,0,0,0,0,
                          0,0,
                          age,
                          0,0,0,0,0,0,0]])

    # Prediction
    prediction = model.predict(features)

    # Default probability
    prob = model.predict_proba(features)

    default_probability = prob[0][1]

    # Risk score (inverse probability)
    risk_score = int((1 - default_probability) * 100)

    st.subheader("Results")

    st.write("Default Probability:", round(default_probability, 2))

    st.write("Risk Score:", risk_score, "/100")

    if prediction[0] == 1:
        st.error("Loan Decision: High Risk (Reject Loan)")
    else:
        st.success("Loan Decision: Low Risk (Approve Loan)")