import os
import yaml
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# ======================================================
# LOAD PARAMETERS
# ======================================================

with open("params.yaml", "r") as file:

    params = yaml.safe_load(file)

TEST_SIZE = params["model"]["test_size"]

RANDOM_STATE = params["model"]["random_state"]

MAX_ITER = params["model"]["max_iter"]

DATASET_PATH = params["paths"]["dataset"]

MODEL_DIR = params["paths"]["model_dir"]

OUTPUT_DIR = params["paths"]["output_dir"]

# ======================================================
# CREATE DIRECTORIES
# ======================================================

os.makedirs(MODEL_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# LOAD DATASET
# ======================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("✓ Dataset loaded successfully")

print(f"Dataset shape: {df.shape}")

# ======================================================
# DISPLAY CLASS DISTRIBUTION
# ======================================================

print("\nGrade Distribution:\n")

print(df["grade"].value_counts())

# ======================================================
# FEATURES + TARGET
# ======================================================

X = df.drop("grade", axis=1)

y = df["grade"]

# ======================================================
# HANDLE CATEGORICAL FEATURES
# ======================================================

print("\nEncoding categorical features...")

feature_encoders = {}

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(X[column])

    feature_encoders[column] = encoder

print("✓ Categorical encoding completed")

# ======================================================
# TARGET ENCODING
# ======================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

print("\nPreprocessing data...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded

)

print("\n✓ Train/Test split completed")

print(f"Train size: {X_train.shape}")

print(f"Test size : {X_test.shape}")

# ======================================================
# FEATURE SCALING
# ======================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ======================================================
# MODEL TRAINING
# ======================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(

    max_iter=MAX_ITER,
    random_state=RANDOM_STATE

)

model.fit(X_train_scaled, y_train)

print("✓ Model training completed")

# ======================================================
# PREDICTIONS
# ======================================================

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)

# ======================================================
# EVALUATION
# ======================================================

accuracy = accuracy_score(y_test, y_pred)

auc = roc_auc_score(

    y_test,
    y_prob,
    multi_class="ovr"

)

cv_scores = cross_val_score(

    model,
    scaler.transform(X),
    y_encoded,
    cv=5

)

# ======================================================
# RESULTS
# ======================================================

print("\nEvaluating model...")

print("\n" + "─" * 55)

print("LOGISTIC REGRESSION RESULTS")

print("─" * 55)

print(f"\nAccuracy : {accuracy:.4f}")

print(f"AUC Score: {auc:.4f}")

print(f"CV Score : {cv_scores.mean():.4f}")

print("\nClassification Report:\n")

print(

    classification_report(

        y_test,
        y_pred,
        target_names=label_encoder.classes_

    )

)

# ======================================================
# SAVE ARTIFACTS
# ======================================================

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

    label_encoder,
    f"{MODEL_DIR}/student_label_encoder.pkl"

)

joblib.dump(

    list(X.columns),
    f"{MODEL_DIR}/feature_columns.pkl"

)

joblib.dump(

    feature_encoders,
    f"{MODEL_DIR}/feature_encoders.pkl"

)

print("✓ Model artifacts saved")

# ======================================================
# CONFUSION MATRIX
# ======================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(label_encoder.classes_))

plt.xticks(

    tick_marks,
    label_encoder.classes_,
    rotation=45

)

plt.yticks(

    tick_marks,
    label_encoder.classes_

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(

            j,
            i,
            cm[i, j],
            ha="center",
            va="center"

        )

plt.tight_layout()

plt.savefig(

    f"{OUTPUT_DIR}/confusion_matrix.png"

)

print("✓ Confusion matrix saved")

# ======================================================
# SAMPLE PREDICTIONS
# ======================================================

print("\n── SAMPLE PREDICTIONS ──")

sample_1 = X.iloc[[0]]

sample_2 = X.iloc[[1]]

samples = [sample_1, sample_2]

for idx, sample in enumerate(samples, start=1):

    sample_scaled = scaler.transform(sample)

    pred = model.predict(sample_scaled)

    probs = model.predict_proba(sample_scaled)[0]

    prediction = label_encoder.inverse_transform(pred)[0]

    print(f"\nStudent {idx}")

    print(f"Prediction: {prediction}")

    prob_text = " | ".join([

        f"{label_encoder.classes_[i]}: {p*100:.1f}%"

        for i, p in enumerate(probs)

    ])

    print(f"Probabilities: {prob_text}")

# ======================================================
# FINAL STATUS
# ======================================================

print("\n✓ Pipeline completed successfully")

print(f"\nModels saved in : {MODEL_DIR}")

print(f"Outputs saved in: {OUTPUT_DIR}")

print("\nGenerated files:")

print("\nMODELS/")

print("- student_model.pkl")

print("- student_scaler.pkl")

print("- student_label_encoder.pkl")

print("- feature_columns.pkl")

print("- feature_encoders.pkl")

print("\nOUTPUTS/")

print("- confusion_matrix.png")