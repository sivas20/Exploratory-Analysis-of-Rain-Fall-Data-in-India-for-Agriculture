from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained classification model
model = pickle.load(open("Rainfall.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values in correct order
        features = [float(x) for x in request.form.values()]
        final_input = np.array([features])

        prediction = model.predict(final_input)[0]

        if prediction == 1:
            return render_template("chance.html")
        else:
            return render_template("noChance.html")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
