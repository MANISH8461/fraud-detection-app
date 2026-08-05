"""
app.py
------
Flask web app for the fraud classifier.
Loads fraud_model.pkl (created by train_model.py) and serves
a simple form where a user enters transaction details and
gets a Fraud / Not Fraud prediction.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------
# Load the trained model bundle once, at startup
# ---------------------------------------------------------
bundle = joblib.load("fraud_model.pkl")
model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]  # e.g. ["amount", "account_age_days", "num_prev_transactions"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            # Read form inputs in the same order the model was trained on
            values = [float(request.form[feat]) for feat in features]
            X = np.array(values).reshape(1, -1)

            proba = model.predict_proba(X)[0][1]  # probability of class 1 (fraud)
            is_fraud = proba > threshold

            probability = round(proba * 100, 2)
            prediction = "🚨 Fraud" if is_fraud else "✅ Not Fraud"

        except Exception as e:
            error = f"Invalid input: {e}"

    return render_template(
        "index.html",
        features=features,
        prediction=prediction,
        probability=probability,
        error=error,
    )


if __name__ == "__main__":
    # debug=True only for local testing — Render runs this via gunicorn (see Procfile),
    # so this block doesn't execute in production
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
