from flask import Flask, request, jsonify

from src.predict import predict_student_performance

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

        # Return response
        return jsonify(result)

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 400

# ======================================================
# RUN SERVER
# ======================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )