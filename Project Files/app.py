from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("Rainfall.pkl")
scaler = joblib.load("scale.pkl")
num_imputer = joblib.load("imputer.pkl")       
encoder_dict = joblib.load("encoder.pkl")
FEATURE_ORDER = ['Location', 'MinTemp', 'MaxTemp', 'Humidity3pm', 'WindDir3pm']
NUM_COLS = ['MinTemp', 'MaxTemp', 'Humidity3pm']

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.form.to_dict()
        df = pd.DataFrame([input_data])

        for col, encoder in encoder_dict.items():
            if col in df.columns:
                val = df[col][0]
                if val not in encoder.classes_:
                    return f"Error: Unknown value '{val}' for {col}"
                df[col] = encoder.transform(df[col])

        for col in NUM_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df[NUM_COLS] = num_imputer.transform(df[NUM_COLS])
        df = df[FEATURE_ORDER]

        scaled_data = scaler.transform(df.values.astype(float))
        prediction = model.predict(scaled_data)

        if prediction[0] == 1:
            return render_template("chance.html")
        else:
            return render_template("noChance.html")

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)
