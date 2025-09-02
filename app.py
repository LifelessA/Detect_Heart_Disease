import streamlit as st
import pandas as pd
import joblib

# --- 1. Load the Trained Model ---
try:
    pipeline = joblib.load('heart_disease_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file 'heart_disease_pipeline.pkl' not found. Please run the Jupyter Notebook to train and save the model first.")
    st.stop()

# --- 2. Page Configuration and Title ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

st.title("❤️ Heart Disease Prediction")
st.markdown("This application uses a machine learning model to predict the likelihood of heart disease based on patient data. Please enter the patient's information in the sidebar.")

# --- 3. User Input in Sidebar ---
st.sidebar.header("Patient Information")

# --- Feature Mappings for User-Friendly Input ---
sex_map = {1: 'Male', 0: 'Female'}
chest_pain_map = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal Pain', 4: 'Asymptomatic'}
fasting_bs_map = {1: '> 120 mg/dL', 0: '<= 120 mg/dL'}
resting_ecg_map = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Probable or Definite Left Ventricular Hypertrophy'}
exercise_angina_map = {1: 'Yes', 0: 'No'}
st_slope_map = {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}

# --- Input Fields ---
with st.sidebar:
    age = st.slider("Age", 29, 77, 54)
    sex_label = st.selectbox("Sex", options=list(sex_map.values()))
    sex = [k for k, v in sex_map.items() if v == sex_label][0]

    cp_label = st.selectbox("Chest Pain Type", options=list(chest_pain_map.values()))
    chest_pain_type = [k for k, v in chest_pain_map.items() if v == cp_label][0]

    resting_bp_s = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)
    cholesterol = st.slider("Serum Cholesterol (mg/dl)", 126, 564, 240)
    
    fasting_bs_label = st.selectbox("Fasting Blood Sugar", options=list(fasting_bs_map.values()))
    fasting_blood_sugar = [k for k, v in fasting_bs_map.items() if v == fasting_bs_label][0]

    resting_ecg_label = st.selectbox("Resting ECG Results", options=list(resting_ecg_map.values()))
    resting_ecg = [k for k, v in resting_ecg_map.items() if v == resting_ecg_label][0]

    max_heart_rate = st.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    
    exercise_angina_label = st.selectbox("Exercise Induced Angina", options=list(exercise_angina_map.values()))
    exercise_angina = [k for k, v in exercise_angina_map.items() if v == exercise_angina_label][0]

    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, 0.1)

    st_slope_label = st.selectbox("Slope of the Peak Exercise ST Segment", options=list(st_slope_map.values()))
    st_slope = [k for k, v in st_slope_map.items() if v == st_slope_label][0]


# --- 4. Prediction Logic and Display ---
if st.sidebar.button("Predict Diagnosis", type="primary"):
    # Create a DataFrame from the inputs
    input_data = {
        'age': [age],
        'sex': [sex],
        'chest pain type': [chest_pain_type],
        'resting bp s': [resting_bp_s],
        'cholesterol': [cholesterol],
        'fasting blood sugar': [fasting_blood_sugar],
        'resting ecg': [resting_ecg],
        'max heart rate': [max_heart_rate],
        'exercise angina': [exercise_angina],
        'oldpeak': [oldpeak],
        'ST slope': [st_slope]
    }
    input_df = pd.DataFrame(input_data)

    # Make prediction
    try:
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0]

        # Display result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"High Risk of Heart Disease Detected (Probability: {prediction_proba[1]*100:.2f}%)")
            st.markdown("The model indicates a high probability of heart disease. It is strongly recommended to consult a medical professional for a comprehensive evaluation.")
        else:
            st.success(f"Low Risk of Heart Disease Detected (Probability: {prediction_proba[0]*100:.2f}%)")
            st.markdown("The model indicates a low probability of heart disease based on the data provided. Maintaining a healthy lifestyle is always recommended.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- 5. Disclaimer ---
st.markdown("---")
st.warning("**Disclaimer:** This is a machine learning tool and not a substitute for professional medical advice. The prediction is based on the patterns in the dataset used for training.")

