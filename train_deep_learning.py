
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = "lung_cancer.csv"

MODEL_PATH = "lung_cancer_deep_model.keras"
SCALER_PATH = "lung_cancer_scaler.pkl"


print("=" * 60)
print("LUNG CANCER DEEP LEARNING MODEL")
print("=" * 60)


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    print(f"\nERROR: Dataset not found!")
    print(f"Please put '{DATASET_PATH}' in the same folder.")
    exit()


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)

print("\nOriginal columns:")
print(data.columns.tolist())


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

data.columns = data.columns.str.strip().str.upper()

print("\nCleaned columns:")
print(data.columns.tolist())


# ============================================================
# 5. CHECK TARGET COLUMN
# ============================================================

TARGET = "LUNG_CANCER"

if TARGET not in data.columns:
    print(f"\nERROR: '{TARGET}' column not found!")
    print("Available columns:")
    print(data.columns.tolist())
    exit()


# ============================================================
# 6. REMOVE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

print(data.isnull().sum())

data = data.dropna()

print("\nDataset shape after removing missing values:")
print(data.shape)


# ============================================================
# 7. CONVERT TEXT VALUES TO NUMERIC
# ============================================================

print("\nConverting categorical values...")


# Remove unnecessary spaces
for column in data.columns:
    if data[column].dtype == "object":
        data[column] = data[column].astype(str).str.strip().str.upper()


# Gender
if "GENDER" in data.columns:
    data["GENDER"] = data["GENDER"].map({
        "M": 1,
        "MALE": 1,
        "F": 0,
        "FEMALE": 0
    })


# YES / NO columns
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
# 8. CONVERT ALL COLUMNS TO NUMERIC
# ============================================================

for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")


# Remove rows that became NaN
data = data.dropna()


# ============================================================
# 9. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=[TARGET])
y = data[TARGET]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(TARGET)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 10. CHECK TARGET VALUES
# ============================================================

print("\nTarget values:")
print(y.value_counts())


# ============================================================
# 11. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 12. FEATURE SCALING
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Save scaler
joblib.dump(scaler, SCALER_PATH)

print(f"Scaler saved as: {SCALER_PATH}")


# ============================================================
# 13. BUILD DEEP LEARNING MODEL
# ============================================================

print("\nBuilding Neural Network...")


model = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 14. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print("\nModel summary:")
model.summary()


# ============================================================
# 15. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# 16. TRAIN MODEL
# ============================================================

print("\nStarting model training...")
print("=" * 60)


history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# 17. EVALUATE MODEL
# ============================================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)


print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# 18. PREDICTIONS
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=0
)


y_pred = (y_probability >= 0.5).astype(int).flatten()


# ============================================================
# 19. ACCURACY
# ============================================================

accuracy_score_value = accuracy_score(
    y_test,
    y_pred
)


print(
    f"\nAccuracy Score: "
    f"{accuracy_score_value * 100:.2f}%"
)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 22. SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n")
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(f"\nModel: {MODEL_PATH}")
print(f"Scaler: {SCALER_PATH}")


# ============================================================
# 23. FINAL INFORMATION
# ============================================================

print("\nFeatures used by the model:")

for i, feature in enumerate(X.columns, start=1):
    print(f"{i}. {feature}")


print("\nTraining completed successfully!")

print("=" * 60)
