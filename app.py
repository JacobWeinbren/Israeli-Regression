from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)
model_jewish = joblib.load("output/best_model_jewish.joblib")
model_arab = joblib.load("output/best_model_arab.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sector = data.pop("sector", None)

    if sector not in [1, 2]:
        return jsonify({"error": "Invalid sector"}), 400

    input_df = pd.DataFrame([data])

    if sector == 1:
        predictions = model_jewish.predict_proba(input_df)
    elif sector == 2:
        predictions = model_arab.predict_proba(input_df)

    return jsonify(predictions.tolist())


@app.route("/")
def home():
    return "Hello World", 200


if __name__ == "__main__":
    app.run()
