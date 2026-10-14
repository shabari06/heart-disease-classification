import streamlit as st
import requests

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CardioCheck | Heart Risk Estimator",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = "http://backend:8000/predict"


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #142b45, #245b70);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.5rem;
        font-size: 2.4rem;
    }

    .hero p {
        color: #e1edf5;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 1.3rem;
        font-weight: 650;
        margin: 1rem 0;
    }

    .disclaimer {
        padding: 1rem;
        border-radius: 10px;
        background: #fff5df;
        color: #694b13;
        margin-top: 1.5rem;
        font-size: 0.9rem;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        font-weight: 600;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>❤️ CardioCheck</h1>
    <p>
        Heart Disease Classification using Machine Learning
        <br>
        Enter the requested health parameters to obtain a model prediction.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("### Patient Information")
st.caption("Enter the available values as accurately as possible.")


# ==========================================
# INPUT FORM
# ==========================================

with st.form("patient_form"):

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Basic Information")

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=56
        )

        # These options are provisional.
        # Confirm their encoding using the dataset documentation.
        sex_label = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=1,
            max_value=300,
            value=120
        )

        cholesterol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            min_value=1,
            max_value=1000,
            value=240
        )

        fasting_bs_label = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            ["No", "Yes"]
        )

        st.markdown("#### Cardiac Measurements")

        max_heart_rate = st.number_input(
            "Maximum Heart Rate Achieved (bpm)",
            min_value=1,
            max_value=300,
            value=150
        )

        oldpeak = st.number_input(
            "Oldpeak (ST Depression)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1
        )

    with right:
        st.markdown("#### Clinical Indicators")

        chest_pain_label = st.selectbox(
            "Chest Pain Type",
            [
                "Type 0",
                "Type 1",
                "Type 2",
                "Type 3"
            ]
        )

        ecg_label = st.selectbox(
            "Resting ECG Result",
            [
                "Category 0",
                "Category 1",
                "Category 2"
            ]
        )

        exercise_angina_label = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"]
        )

        slope_label = st.selectbox(
            "ST Segment Slope",
            [
                "Category 0",
                "Category 1",
                "Category 2"
            ]
        )

        major_vessels = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3, 4]
        )

        thal_label = st.selectbox(
            "Thalassemia Test Category",
            [
                "Category 0",
                "Category 1",
                "Category 2",
                "Category 3"
            ]
        )

    st.markdown("---")

    submitted = st.form_submit_button(
        "Generate Prediction",
        type="primary",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if submitted:

    # IMPORTANT:
    # These mappings must match the encoding used during training.
    # Verify them against your dataset documentation.

    patient_data = {
        "age": int(age),
        "sex": 1 if sex_label == "Male" else 0,
        "cp": int(chest_pain_label.split()[-1]),
        "trestbps": int(resting_bp),
        "chol": int(cholesterol),
        "fbs": 1 if fasting_bs_label == "Yes" else 0,
        "restecg": int(ecg_label.split()[-1]),
        "thalach": int(max_heart_rate),
        "exang": 1 if exercise_angina_label == "Yes" else 0,
        "oldpeak": float(oldpeak),
        "slope": int(slope_label.split()[-1]),
        "ca": int(major_vessels),
        "thal": int(thal_label.split()[-1])
    }

    try:
        with st.spinner("Analyzing the submitted parameters..."):

            response = requests.post(
                API_URL,
                json=patient_data,
                timeout=15
            )

        if response.status_code == 200:

            result = response.json()
            prediction = int(result["prediction"])

            st.markdown("---")
            st.markdown("## Prediction Summary")

            if prediction == 1:
                st.warning(
                    "### The model predicts a higher likelihood "
                    "of heart disease."
                )
            else:
                st.success(
                    "### The model predicts a lower likelihood "
                    "of heart disease."
                )

            st.write(
                "This result represents the classification produced "
                "by the trained machine learning model."
            )

            with st.expander("View submitted parameters"):
                st.json(patient_data)

        else:
            st.error(
                f"The prediction API returned HTTP {response.status_code}."
            )
            st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the prediction service. "
            "Please make sure the FastAPI server is running."
        )

    except requests.exceptions.Timeout:
        st.error("The prediction request timed out.")

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")


# ==========================================
# DISCLAIMER
# ==========================================

st.markdown("""
<div class="disclaimer">
    <strong>Important notice:</strong>
    This application is an educational machine learning project.
    Its predictions are not clinically validated and must not be used
    to diagnose heart disease or guide medical treatment.
    Consult a qualified healthcare professional for medical concerns.
</div>
""", unsafe_allow_html=True)