"""
Prediction Pipeline
===================

PURPOSE:
- Load trained model artifacts
- Accept new student input
- Scale features
- Predict grade
- Return probabilities

USED BY:
- Flask API
- Streamlit frontend
- Deployment services
"""

# =========================================================
# IMPORTS
# =========================================================

import joblib
import pandas as pd

# =========================================================
# LOAD ARTIFACTS
# =========================================================

print("Loading model artifacts...")

model = joblib.load("models/student_model.pkl")

scaler = joblib.load("models/student_scaler.pkl")

label_encoder = joblib.load(
    "models/student_label_encoder.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

print("✓ Artifacts loaded successfully")

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_student_performance(student_data):

    """
    Predict student grade and probabilities

    Parameters:
        student_data (dict)

    Returns:
        dict
    """

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([student_data])

    # Ensure correct feature order
    input_df = input_df[feature_columns]

    # Scale features
    input_scaled = scaler.transform(input_df)

    # Predict encoded class
    prediction = model.predict(input_scaled)[0]

    # Predict probabilities
    probabilities = model.predict_proba(input_scaled)[0]

    # Decode prediction label
    predicted_grade = label_encoder.inverse_transform(
        [prediction]
    )[0]

    # Format probabilities
    probability_dict = {

        label_encoder.classes_[i]: round(float(prob), 4)

        for i, prob in enumerate(probabilities)

    }

    # Final response
    result = {

        "prediction": predicted_grade,

        "probabilities": probability_dict

    }

    return result

# =========================================================
# SAMPLE TEST INPUT
# =========================================================

if __name__ == "__main__":

    sample_student = {

        "study_hours": 8,
        "attendance_pct": 92,
        "prev_gpa": 3.7,
        "assignments_done": 95,
        "sleep_hours": 7,
        "internet_hours": 2,
        "family_support": 5,
        "part_time_job": 0,
        "extracurricular": 1,
        "parent_edu": 3,
        "gender": 0,
        "school_type": 1,
        "distance_km": 5,
        "quiz_avg": 88,
        "midterm_score": 85

    }

    result = predict_student_performance(
        sample_student
    )

    print("\nPrediction Result:\n")

    print(result)