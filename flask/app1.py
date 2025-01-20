from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
import numpy as np
from io import BytesIO

# Load the pre-trained models and encoders
encoder = pickle.load(open('encoder_1.pkl', 'rb'))
scaler = pickle.load(open('scaler_1.pkl', 'rb'))
model = pickle.load(open('final_model_rf.pkl', 'rb'))

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('csv.html')

# Route to upload CSV and get fraud prediction
@app.route('/csv.html', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the file part exists
        if 'file' not in request.files:
            print("No file part")
            return 'No file part', 400
        
        file = request.files['file']
        
        # Check if no file was selected
        if file.filename == '':
            print("No selected file")
            return 'No selected file', 400

        # Check if file is a CSV
        if file and file.filename.endswith('.csv'):
            try:
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(file)

                # Check that the required columns are present
                required_columns = [
                    'amount', 'type', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                    'newbalanceDest', 'step_days'
                ]
                for column in required_columns:
                    if column not in df.columns:
                        return render_template('error.html', error_message=f'Missing column: {column}')

                # Feature engineering
                df['diff_new_old_orgin'] = df['newbalanceOrig'] - df['oldbalanceOrg']
                df['diff_new_old_dest'] = df['newbalanceDest'] - df['oldbalanceDest']

                # One-hot encoding for transaction types
                df = pd.get_dummies(df, columns=['type'], drop_first=True)

                # Ensure all required columns are present after encoding
                input_columns = [
                    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                    'step_days', 'diff_new_old_orgin', 'diff_new_old_dest',
                    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
                ]
                # Align the DataFrame to have the same columns as the model was trained on
                df = df.reindex(columns=input_columns, fill_value=0)

                # Scale the data
                df_scaled = scaler.transform(df)

                # Make predictions
                predictions = model.predict(df_scaled)

                # Add predictions to the DataFrame
                df['fraud_prediction'] = predictions
                df['fraud_prediction'] = df['fraud_prediction'].map({1: 'Fraud', 0: 'Legitimate'})

                # Convert the result into a CSV in-memory
                output = BytesIO()
                df.to_csv(output, index=False)
                output.seek(0)

                # Provide the output CSV file for download
                return send_file(output, as_attachment=True, download_name="fraud_detection_results.csv", mimetype="text/csv")

            except Exception as e:
                print(f"Error processing CSV: {e}")
                return f"Error processing CSV: {e}", 500
        else:
            print("Invalid file type")
            return 'Invalid file type. Please upload a CSV file.', 400

    return render_template('csv.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
