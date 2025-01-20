from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the pre-trained models and encoders
encoder = pickle.load(open('model/encoder_final.pkl', 'rb'))
scaler = pickle.load(open('model/scaler_final.pkl', 'rb'))
model = pickle.load(open('model/final_model.pkl', 'rb'))

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the result page
@app.route('/templates/result.html', methods=['POST'])
def result():
    try:
        # Retrieve form data
        amount = float(request.form['amount'])
        type_ = request.form['type']
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])
        step_days = int(request.form['step_days'])

        # Feature engineering to match training
        diff_new_old_origin = newbalanceOrig - oldbalanceOrg
        diff_new_old_dest = newbalanceDest - oldbalanceDest

        # Create a DataFrame with all required features
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
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest','step_days',
            'diff_new_old_orgin', 'diff_new_old_dest',
            'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
        ]
        input_data = input_data[feature_order]

        # Scale the data
        input_data_scaled = scaler.transform(input_data)

        # Predict using the model
        prediction = model.predict(input_data_scaled)

        # Interpret the result
        if prediction[0] == 1:
            result_text = "Fraud Detected"
            result_class = "fraud"
        else:
            result_text = "Safe Transaction"
            result_class = "legitimate"

        # Render the result page
        return render_template('result.html', result=result_text, resultClass=result_class)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
