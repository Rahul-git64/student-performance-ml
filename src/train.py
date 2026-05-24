"""
Student Performance Predictor
=============================

FINAL PRODUCTION-ORIENTED TRAINING PIPELINE

FEATURES:
- Reads dataset from CSV
- Trains Logistic Regression model
- Evaluates model
- Saves deployment artifacts
- Saves confusion matrix
- Modular deployment-friendly structure

PROJECT STRUCTURE:
student-performance-ml/
│
├── data/
│   └── student_dataset.csv
│
├── models/
│
├── outputs/
│
├── src/
│   └── train.py
│
└── requirements.txt
"""

# =========================================================
# IMPORTS
# =========================================================

import os
import warnings
import joblib

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = "data/student_dataset.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("✓ Dataset loaded successfully")
print(f"Dataset shape: {df.shape}")

# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\nGrade Distribution:\n")
print(df["grade"].value_counts())

# =========================================================
# PREPROCESSING
# =========================================================

print("\nPreprocessing data...")

# Label Encoders
le_parent = LabelEncoder()
le_gender = LabelEncoder()
le_school = LabelEncoder()
le_grade = LabelEncoder()

# Copy dataframe
dm = df.copy()

# Encode categorical columns
dm["parent_edu"] = le_parent.fit_transform(dm["parent_edu"])
dm["gender"] = le_gender.fit_transform(dm["gender"])
dm["school_type"] = le_school.fit_transform(dm["school_type"])

# Encode target
dm["grade_enc"] = le_grade.fit_transform(dm["grade"])

# =========================================================
# FEATURES
# =========================================================

FEATURES = [

    "study_hours",
    "attendance_pct",
    "prev_gpa",
    "assignments_done",
    "sleep_hours",
    "internet_hours",
    "family_support",
    "part_time_job",
    "extracurricular",
    "parent_edu",
    "gender",
    "school_type",
    "distance_km",
    "quiz_avg",
    "midterm_score"

]

TARGET = "grade_enc"

X = dm[FEATURES]
y = dm[TARGET]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n✓ Train/Test split completed")

print(f"Train size: {X_train.shape}")
print(f"Test size : {X_test.shape}")

# =========================================================
# FEATURE SCALING
# =========================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODEL TRAINING
# =========================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("✓ Model training completed")

# =========================================================
# MODEL EVALUATION
# =========================================================

print("\nEvaluating model...")

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

auc = roc_auc_score(
    y_test,
    y_prob,
    multi_class="ovr"
)

cv_score = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
).mean()

# =========================================================
# RESULTS
# =========================================================

print("\n" + "─" * 55)
print("LOGISTIC REGRESSION RESULTS")
print("─" * 55)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"AUC Score: {auc:.4f}")
print(f"CV Score : {cv_score:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=le_grade.classes_
))

# =========================================================
# SAVE MODEL ARTIFACTS
# =========================================================

print("\nSaving model artifacts...")

joblib.dump(
    model,
    f"{MODEL_DIR}/student_model.pkl"
)

joblib.dump(
    scaler,
    f"{MODEL_DIR}/student_scaler.pkl"
)

joblib.dump(
    le_grade,
    f"{MODEL_DIR}/student_label_encoder.pkl"
)

joblib.dump(
    FEATURES,
    f"{MODEL_DIR}/feature_columns.pkl"
)

print("✓ Model artifacts saved")

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le_grade.classes_,
    yticklabels=le_grade.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    f"{OUTPUT_DIR}/confusion_matrix.png",
    dpi=120,
    bbox_inches="tight"
)

plt.close()

print("✓ Confusion matrix saved")

# =========================================================
# SAMPLE PREDICTIONS
# =========================================================

print("\n── SAMPLE PREDICTIONS ──\n")

samples = pd.DataFrame([

    {
        "study_hours": 9,
        "attendance_pct": 95,
        "prev_gpa": 3.9,
        "assignments_done": 97,
        "sleep_hours": 8,
        "internet_hours": 1,
        "family_support": 5,
        "part_time_job": 0,
        "extracurricular": 1,
        "parent_edu": 3,
        "gender": 0,
        "school_type": 1,
        "distance_km": 4,
        "quiz_avg": 90,
        "midterm_score": 88
    },

    {
        "study_hours": 3,
        "attendance_pct": 60,
        "prev_gpa": 2.1,
        "assignments_done": 45,
        "sleep_hours": 5,
        "internet_hours": 7,
        "family_support": 2,
        "part_time_job": 1,
        "extracurricular": 0,
        "parent_edu": 1,
        "gender": 1,
        "school_type": 0,
        "distance_km": 25,
        "quiz_avg": 42,
        "midterm_score": 40
    }

])

# Scale sample inputs
samples_scaled = scaler.transform(samples)

# Predict
predictions = model.predict(samples_scaled)
probabilities = model.predict_proba(samples_scaled)

# Display predictions
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):

    grade = le_grade.inverse_transform([pred])[0]

    probs = " | ".join([
        f"{le_grade.classes_[j]}: {p*100:.1f}%"
        for j, p in enumerate(prob)
    ])

    print(f"Student {i+1}")
    print(f"Prediction: {grade}")
    print(f"Probabilities: {probs}\n")

# =========================================================
# FINAL OUTPUT
# =========================================================

print("✓ Pipeline completed successfully")

print(f"\nModels saved in : {MODEL_DIR}")
print(f"Outputs saved in: {OUTPUT_DIR}")

print("\nGenerated files:")

print("""
MODELS/
- student_model.pkl
- student_scaler.pkl
- student_label_encoder.pkl
- feature_columns.pkl

OUTPUTS/
- confusion_matrix.png
""")