from flask import Flask, request, jsonify

from datetime import datetime

import json
import os
import time

from prometheus_client import (

    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST

)

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
# PROMETHEUS METRICS
# ======================================================

REQUEST_COUNT = Counter(

    "prediction_requests_total",
    "Total prediction requests"

)

PREDICTION_ERRORS = Counter(

    "prediction_errors_total",
    "Total prediction failures"

)

PREDICTION_LATENCY = Histogram(

    "prediction_latency_seconds",
    "Prediction latency"

)

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
# PROMETHEUS METRICS ROUTE
# ======================================================

@app.route("/metrics")

def metrics():

    return generate_latest(), 200, {

        "Content-Type": CONTENT_TYPE_LATEST

    }

# ======================================================
# PREDICTION ROUTE
# ======================================================

@app.route("/predict", methods=["POST"])

def predict():

    REQUEST_COUNT.inc()

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

        PREDICTION_LATENCY.observe(latency)

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

        PREDICTION_ERRORS.inc()

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