import streamlit as st
import numpy as np
import joblib
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "lung_cancer_deep_model.keras"
    )

    scaler = joblib.load(
        "lung_cancer_scaler.pkl"
    )

    return model, scaler


try:

    model, scaler = load_model()

except Exception as e:

    st.error("❌ Model or scaler could not be loaded.")
    st.error(str(e))
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🫁 Lung Cancer Prediction System")

st.write(
    "Enter the patient's information below and click "
    "**Predict Lung Cancer** to get the model prediction."
)

st.warning(
    "⚠️ This application is for educational and research "
    "purposes only. It is NOT a medical diagnosis."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("📊 Model Information"):

    st.write("**Model:** Deep Neural Network")
    st.write("**Dataset:** Lung Cancer Dataset")
    st.write("**Dataset Size:** 1,200 records")
    st.write("**Features:** 15")
    st.write("**Test Accuracy:** 78.33%")
    st.write("**Precision:** 76.84%")
    st.write("**Recall:** 70.87%")
    st.write("**F1-Score:** 73.74%")
    st.write("**ROC-AUC:** 0.8357")


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.subheader("👤 Patient Information")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


# ============================================================
# SYMPTOMS AND RISK FACTORS
# ============================================================

st.subheader("🩺 Symptoms and Risk Factors")

smoking = st.selectbox(
    "Smoking",
    ["No", "Yes"]
)

yellow_fingers = st.selectbox(
    "Yellow Fingers",
    ["No", "Yes"]
)

anxiety = st.selectbox(
    "Anxiety",
    ["No", "Yes"]
)

peer_pressure = st.selectbox(
    "Peer Pressure",
    ["No", "Yes"]
)

chronic_disease = st.selectbox(
    "Chronic Disease",
    ["No", "Yes"]
)

fatigue = st.selectbox(
    "Fatigue",
    ["No", "Yes"]
)

allergy = st.selectbox(
    "Allergy",
    ["No", "Yes"]
)

wheezing = st.selectbox(
    "Wheezing",
    ["No", "Yes"]
)

alcohol_consuming = st.selectbox(
    "Alcohol Consuming",
    ["No", "Yes"]
)

coughing = st.selectbox(
    "Coughing",
    ["No", "Yes"]
)

shortness_of_breath = st.selectbox(
    "Shortness of Breath",
    ["No", "Yes"]
)

swallowing_difficulty = st.selectbox(
    "Swallowing Difficulty",
    ["No", "Yes"]
)

chest_pain = st.selectbox(
    "Chest Pain",
    ["No", "Yes"]
)


# ============================================================
# CONVERSION FUNCTIONS
# IMPORTANT:
# This must match train_deep_learning.py
# ============================================================

def yes_no(value):

    if value == "Yes":
        return 2

    return 1


# Gender:
# M / Male = 1
# F / Female = 0

gender_value = 1 if gender == "Male" else 0


# ============================================================
# FEATURE ORDER
# MUST MATCH TRAINING MODEL
# ============================================================

features = np.array([
    [
        gender_value,
        age,
        yes_no(smoking),
        yes_no(yellow_fingers),
        yes_no(anxiety),
        yes_no(peer_pressure),
        yes_no(chronic_disease),
        yes_no(fatigue),
        yes_no(allergy),
        yes_no(wheezing),
        yes_no(alcohol_consuming),
        yes_no(coughing),
        yes_no(shortness_of_breath),
        yes_no(swallowing_difficulty),
        yes_no(chest_pain)
    ]
], dtype=float)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
    "GENDER",
    "AGE",
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
    "CHEST_PAIN"
]


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Lung Cancer",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        scaled_features = scaler.transform(features)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        probability = float(
            model.predict(
                scaled_features,
                verbose=0
            )[0][0]
        )


        # Make sure probability stays between 0 and 1

        probability = min(
            max(probability, 0.0),
            1.0
        )


        probability_percent = probability * 100


        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        if probability >= 0.50:

            prediction = "Lung Cancer"

        else:

            prediction = "No Lung Cancer"


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("🔬 Prediction Result")


        if prediction == "Lung Cancer":

            st.error(
                "⚠️ Model Prediction: LUNG CANCER"
            )

        else:

            st.success(
                "✅ Model Prediction: NO LUNG CANCER"
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        st.metric(
            "Model Probability",
            f"{probability_percent:.2f}%"
        )


        st.progress(probability)


        # ----------------------------------------------------
        # RISK INTERPRETATION
        # ----------------------------------------------------

        if probability < 0.30:

            risk_level = "Lower model score"

        elif probability < 0.50:

            risk_level = "Intermediate model score"

        elif probability < 0.70:

            risk_level = "Higher model score"

        else:

            risk_level = "Very high model score"


        st.info(
            f"Model score interpretation: **{risk_level}**"
        )


        # ----------------------------------------------------
        # IMPORTANT WARNING
        # ----------------------------------------------------

        st.warning(
            "⚠️ This probability is produced by a machine "
            "learning model. It should NOT be interpreted "
            "as a clinical diagnosis or actual medical "
            "probability. Please consult a qualified "
            "healthcare professional for medical evaluation."
        )


        # ----------------------------------------------------
        # ENTERED INFORMATION
        # ----------------------------------------------------

        with st.expander(
            "📋 View Entered Information"
        ):

            st.write(f"**Age:** {age}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Smoking:** {smoking}")
            st.write(f"**Yellow Fingers:** {yellow_fingers}")
            st.write(f"**Anxiety:** {anxiety}")
            st.write(f"**Peer Pressure:** {peer_pressure}")
            st.write(f"**Chronic Disease:** {chronic_disease}")
            st.write(f"**Fatigue:** {fatigue}")
            st.write(f"**Allergy:** {allergy}")
            st.write(f"**Wheezing:** {wheezing}")
            st.write(f"**Alcohol Consuming:** {alcohol_consuming}")
            st.write(f"**Coughing:** {coughing}")
            st.write(
                f"**Shortness of Breath:** "
                f"{shortness_of_breath}"
            )
            st.write(
                f"**Swallowing Difficulty:** "
                f"{swallowing_difficulty}"
            )
            st.write(f"**Chest Pain:** {chest_pain}")


    except Exception as e:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Lung Cancer Prediction System | "
    "Deep Learning Project | "
    "Educational & Research Use Only"
)