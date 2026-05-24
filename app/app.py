import sys
import os

# ======================================================
# FIX PYTHON MODULE PATH
# ======================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# ======================================================
# IMPORTS
# ======================================================

from flask import Flask, request, jsonify

from src.predict import predict_student_performance

# ======================================================
# CREATE FLASK APP
# ======================================================

app = Flask(__name__)

# ======================================================
# HOME ROUTE
# ======================================================

@app.route("/")

def home():

    return jsonify({

        "message": "Student Performance Prediction API Running"

    })

# ======================================================
# PREDICTION ROUTE
# ======================================================

@app.route("/predict", methods=["POST"])

def predict():

    try:

        # Get JSON input
        data = request.get_json()

        # Run prediction
        result = predict_student_performance(data)

        # Return prediction response
        return jsonify(result)

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400

# ======================================================
# RUN FLASK SERVER
# ======================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",
        port=5000,
        debug=True

    )