import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained models and encoders
encoder = pickle.load(open('model/encoder_final.pkl', 'rb'))
scaler = pickle.load(open('model/scaler_final.pkl', 'rb'))
model = pickle.load(open('model/final_model.pkl', 'rb'))

# Function to make predictions
def predict_fraud(amount, type_, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, step_days):
    # Feature engineering
    diff_new_old_origin = newbalanceOrig - oldbalanceOrg
    diff_new_old_dest = newbalanceDest - oldbalanceDest

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([{
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'step_days': step_days,
        'diff_new_old_orgin': diff_new_old_origin,
        'diff_new_old_dest': diff_new_old_dest,
        'type_CASH_IN': 1 if type_ == "CASH_IN" else 0,
        'type_CASH_OUT': 1 if type_ == "CASH_OUT" else 0,
        'type_DEBIT': 1 if type_ == "DEBIT" else 0,
        'type_PAYMENT': 1 if type_ == "PAYMENT" else 0,
        'type_TRANSFER': 1 if type_ == "TRANSFER" else 0
    }])

    # Align input data to match model's expected feature order
    feature_order = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'step_days',
        'diff_new_old_orgin', 'diff_new_old_dest',
        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]
    input_data = input_data[feature_order]

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data_scaled)

    return "Fraud Detected" if prediction[0] == 1 else "Safe Transaction"

# Streamlit application
st.title("Fraud Detection System")
st.markdown("Enter the transaction details to check if it's fraudulent or legitimate.")

# Input fields
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01, format="%.2f")
type_ = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
oldbalanceOrg = st.number_input("Original Account Balance (Before Transaction)", min_value=0.0, step=0.01, format="%.2f")
newbalanceOrig = st.number_input("Original Account Balance (After Transaction)", min_value=0.0, step=0.01, format="%.2f")
oldbalanceDest = st.number_input("Destination Account Balance (Before Transaction)", min_value=0.0, step=0.01, format="%.2f")
newbalanceDest = st.number_input("Destination Account Balance (After Transaction)", min_value=0.0, step=0.01, format="%.2f")
step_days = st.number_input("Step (Transaction Day)", min_value=0, step=1, format="%d")

# Prediction button
if st.button("Predict"):
    try:
        # Get the prediction
        result = predict_fraud(amount, type_, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, step_days)
        if result == "Fraud Detected":
            st.error(result)
        else:
            st.success(result)
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
