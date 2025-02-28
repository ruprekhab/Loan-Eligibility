import pandas as pd
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS
from pprint import pprint
app = Flask(__name__)
CORS(app) 

# Load trained model
model = joblib.load("xgboost_model.pkl")  # Replace with your trained model file

# Load feature columns from training
feature_columns = joblib.load("feature_columns.pkl")

# HTML for the homepage
home_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Prediction API</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            text-align: center;
            padding: 50px;
        }
        h1 {
            color: #333;
        }
        p {
            font-size: 20px;
        }
    </style>
</head>
<body>
    <h1>Loan Prediction API</h1>
    <p>The API is working</p>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(home_html)

#make route to load index.html
@app.route("/prediction_page")
def prediction_page():
    return render_template("index.html")


@app.route("/loan_prediction", methods=["POST","GET"])
def predict():
    try:
        # Get input data
        data = request.json
        
        pprint(data)
        # Convert to DataFrame
        data ={'age': float(data["age"]),
                'credit_history_length': float(data["credit_history_length"]),
                'employment_duration': float(data["employment_duration"]),
                'home_ownership': data["home_ownership"],
                'income': float(data["income"]),
                'interest_rate': float(data["interest_rate"]),
                'loan_amount': float(data["loan_amount"]),
                'loan_grade': data["loan_grade"],
                'loan_income_percentage': float(data["loan_income_percentage"]),
                'loan_purpose': data["loan_purpose"],
                'past_default_status': data["past_default_status"] }
        
        pprint(data)
        
        input_df = pd.DataFrame([data])
        
        
        
        
        # Apply one-hot encoding
        categorical_cols = ["home_ownership", "loan_purpose", "loan_grade", "past_default_status"]
        input_df = pd.get_dummies(input_df, columns=categorical_cols)
        
        print(f'input_df \n {input_df.head()}')
        # Align columns with training data
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Add missing columns with 0
        print(f'input_df \n {input_df.head()}')
        # Ensure column order matches training
        input_df = input_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        print(f"prediction   :   {prediction}")
        return jsonify({"prediction": int(prediction)})
        

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
