from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Model path
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "fraud_model.pkl")

# Safe convert function
def get_val(x):
    return float(x) if x else 0

# Type encoding (VERY IMPORTANT)
type_map = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CREDIT": 4
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")

@app.route("/submit", methods=["POST"])
def submit():
    try:
        # Load model
        model = joblib.load(model_path)

        # Get type value properly
        type_input = request.form.get("type")
        type_value = type_map.get(type_input, 0)

        # Prepare input data (correct order)
        data = [
            get_val(request.form.get("step")),
            type_value,
            get_val(request.form.get("amount")),
            get_val(request.form.get("oldbalanceOrg")),
            get_val(request.form.get("newbalanceOrig")),
            get_val(request.form.get("oldbalanceDest")),
            get_val(request.form.get("newbalanceDest")),
            0  # isFlaggedFraud
        ]

        # Prediction
        prediction = model.predict([data])[0]

        # Result
        result = "Fraud" if prediction == 1 else "Not Fraud"

        return render_template("submit.html", result=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)