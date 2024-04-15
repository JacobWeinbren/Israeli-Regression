from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("output/pipeline.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame(data)
    predictions = model.predict_proba(input_df)
    return jsonify(predictions.tolist())


@app.route("/")
def home():
    return "Hello World", 200


if __name__ == "__main__":
    app.run()
