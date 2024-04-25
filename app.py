from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load models
model_jewish = joblib.load("output/best_model_jewish.joblib")
model_arab = joblib.load("output/best_model_arab.joblib")

# Load encoders
encoder_jewish = joblib.load("output/encoder_jewish.joblib")
encoder_arab = joblib.load("output/encoder_arab.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sector = data.pop("sector", None)
    if sector not in [1, 2]:
        return jsonify({"error": "Invalid sector"}), 400

    input_df = pd.DataFrame([data])

    if sector == 1:
        model = model_jewish
        encoder = encoder_jewish
    elif sector == 2:
        model = model_arab
        encoder = encoder_arab

    predictions = model.predict_proba(input_df)
    prediction_keys = model.classes_.tolist()

    # Use the encoder to get original labels
    original_labels = encoder.inverse_transform(prediction_keys)

    predictions_list = predictions[0].tolist()

    # Create result dictionary using original labels
    result = dict(zip(original_labels, predictions_list))

    return jsonify(result)


@app.route("/")
def home():
    return "Hello World", 200


if __name__ == "__main__":
    app.run()
