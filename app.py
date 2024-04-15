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
        model = model_jewish
    elif sector == 2:
        model = model_arab

    predictions = model.predict_proba(input_df)
    prediction_keys = model.classes_.tolist()

    predictions_list = predictions[0].tolist()

    result = dict(zip(prediction_keys, predictions_list))

    return jsonify(result)


@app.route("/")
def home():
    return "Hello World", 200


if __name__ == "__main__":
    app.run()
