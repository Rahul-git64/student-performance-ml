from flask import Flask, request, jsonify

from datetime import datetime

import json
import os
import time

from src.predict import predict_student_performance

# ======================================================
# APP SETUP
# ======================================================

app = Flask(__name__)

# ======================================================
# LOG DIRECTORY
# ======================================================

os.makedirs("logs", exist_ok=True)

LOG_FILE = "logs/prediction_logs.jsonl"

# ======================================================
# HOME ROUTE
# ======================================================

@app.route("/")

def home():

    return jsonify({

        "message": "Student Performance Prediction API Running",

        "status": "healthy"

    })

# ======================================================
# HEALTH CHECK
# ======================================================

@app.route("/health")

def health():

    return jsonify({

        "status": "healthy",

        "timestamp": str(datetime.utcnow())

    })

# ======================================================
# PREDICTION ROUTE
# ======================================================

@app.route("/predict", methods=["POST"])

def predict():

    start_time = time.time()

    try:

        # ==============================================
        # GET INPUT DATA
        # ==============================================

        data = request.get_json()

        # ==============================================
        # RUN PREDICTION
        # ==============================================

        result = predict_student_performance(data)

        # ==============================================
        # LATENCY
        # ==============================================

        latency = round(

            time.time() - start_time,
            4

        )

        # ==============================================
        # LOG ENTRY
        # ==============================================

        log_entry = {

            "timestamp": str(datetime.utcnow()),

            "input": data,

            "prediction": result["prediction"],

            "probabilities": result["probabilities"],

            "latency_seconds": latency

        }

        with open(LOG_FILE, "a") as log_file:

            log_file.write(

                json.dumps(log_entry) + "\n"

            )

        # ==============================================
        # RESPONSE
        # ==============================================

        return jsonify({

            "prediction": result["prediction"],

            "probabilities": result["probabilities"],

            "latency_seconds": latency

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400

# ======================================================
# RUN APP
# ======================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )