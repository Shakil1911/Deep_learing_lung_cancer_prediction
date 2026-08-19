import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "lung_cancer.csv"
MODEL_PATH = "lung_cancer_deep_model.keras"
SCALER_PATH = "lung_cancer_scaler.pkl"


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("LUNG CANCER DEEP LEARNING MODEL EVALUATION")
print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DATASET_PATH,
    MODEL_PATH,
    SCALER_PATH
]

for file_path in required_files:

    if not os.path.exists(file_path):

        print(
            f"\nERROR: File not found -> {file_path}"
        )

        exit()


print("\nAll required files found.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data = pd.read_csv(DATASET_PATH)

print(
    "Dataset shape:",
    data.shape
)


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

data.columns = (
    data.columns
    .astype(str)
    .str.strip()
    .str.upper()
)


print("\nColumns:")

print(data.columns.tolist())


# ============================================================
# TARGET
# ============================================================

TARGET = "LUNG_CANCER"


# ============================================================
# SAME PREPROCESSING AS TRAINING
# ============================================================

print("\nCleaning dataset...")


# Remove unnecessary spaces
for column in data.columns:

    if data[column].dtype == "object":

        data[column] = (
            data[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )


# ============================================================
# GENDER
# ============================================================

if "GENDER" in data.columns:

    data["GENDER"] = data["GENDER"].map({

        "M": 1,
        "MALE": 1,

        "F": 0,
        "FEMALE": 0

    })


# ============================================================
# YES / NO COLUMNS
# ============================================================

yes_no_columns = [

    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
    "LUNG_CANCER"

]


for column in yes_no_columns:

    if column in data.columns:

        data[column] = data[column].replace({

            "YES": 1,
            "NO": 0,

            "Y": 1,
            "N": 0

        })


# ============================================================
# CONVERT ALL COLUMNS TO NUMERIC
# ============================================================

print(
    "\nConverting all columns to numeric..."
)


for column in data.columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# ============================================================
# REMOVE NaN
# ============================================================

data = data.dropna()

data = data.reset_index(
    drop=True
)


print(
    "\nFinal dataset shape:",
    data.shape
)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(
    columns=[TARGET]
)

y = data[TARGET].astype(int)


print("\nFeatures:")

for i, feature in enumerate(
    X.columns,
    start=1
):

    print(
        f"{i}. {feature}"
    )


print(
    "\nFeature shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print(
    "\nTarget distribution:"
)

print(
    y.value_counts()
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# LOAD SCALER
# ============================================================

print(
    "\nLoading scaler..."
)

scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# SCALE TEST DATA
# ============================================================

X_test_scaled = scaler.transform(
    X_test
)


print(
    "Test data scaled successfully."
)


# ============================================================
# LOAD MODEL
# ============================================================

print(
    "\nLoading Deep Learning model..."
)

model = tf.keras.models.load_model(
    MODEL_PATH
)


print(
    "Model loaded successfully."
)


# ============================================================
# PREDICTION
# ============================================================

print(
    "\nGenerating predictions..."
)

y_probability = model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# ============================================================
# SHOW PROBABILITY RANGE
# ============================================================

print(
    "\nPrediction probability statistics:"
)

print(
    "Minimum:",
    round(float(y_probability.min()), 4)
)

print(
    "Maximum:",
    round(float(y_probability.max()), 4)
)

print(
    "Mean:",
    round(float(y_probability.mean()), 4)
)


# ============================================================
# DEFAULT THRESHOLD
# ============================================================

THRESHOLD = 0.50

y_pred = (
    y_probability >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(

        y_test,

        y_pred,

        target_names=[
            "NO LUNG CANCER",
            "LUNG CANCER"
        ],

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=[
        "No Cancer",
        "Cancer"
    ]

)

display.plot(
    ax=ax
)

ax.set_title(
    "Lung Cancer Prediction - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved:"
)

print(
    "confusion_matrix.png"
)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(

    y_test,

    y_probability

)


plt.figure(
    figsize=(7, 6)
)

plt.plot(

    fpr,

    tpr,

    label=f"Deep Learning (AUC = {roc_auc:.3f})"

)

plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--"

)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Lung Cancer Prediction"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.close()


print(
    "ROC curve saved:"
)

print(
    "roc_curve.png"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "Accuracy": accuracy,

    "Precision": precision,

    "Recall": recall,

    "F1_Score": f1,

    "ROC_AUC": roc_auc,

    "Threshold": THRESHOLD

}


pd.DataFrame(
    [metrics]
).to_csv(

    "model_metrics.csv",

    index=False

)


print(
    "Metrics saved:"
)

print(
    "model_metrics.csv"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nGenerated files:"
)

print(
    "1. confusion_matrix.png"
)

print(
    "2. roc_curve.png"
)

print(
    "3. model_metrics.csv"
)

print(
    "\nDone!"
)