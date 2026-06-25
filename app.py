import streamlit as st
import pandas as pd
from src.predictor import predict_heart_disease

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

st.markdown(
    """
    <div style='text-align:center; padding-bottom:20px;'>
        <h1>Heart Disease Prediction</h1>
        <p>Enter the patient's health details to get a prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 100, 55)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
resting_bp = st.sidebar.slider("Resting BP", 90, 200, 130)
cholesterol = st.sidebar.slider("Cholesterol", 100, 600, 220)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
restecg = st.sidebar.selectbox("Rest ECG", [0, 1, 2])
max_hr = st.sidebar.slider("Max Heart Rate", 70, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia", [1, 2, 3])

input_data = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": resting_bp,
    "chol": cholesterol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": max_hr,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal,
}

if st.button("Predict", use_container_width=True):
    prediction, probability = predict_heart_disease(input_data)

    if prediction == 1:
        result = "High Risk of Heart Disease"
        color = "#ef4444"
    else:
        result = "Low Risk of Heart Disease"
        color = "#22c55e"

    st.markdown(
        f"""
        <div style='padding:20px; border-radius:12px; background-color:{color}; color:white; text-align:center;'>
            <h2>{result}</h2>
            <p>Prediction Probability: {probability * 100:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Input Summary")
    st.dataframe(pd.DataFrame([input_data]))
