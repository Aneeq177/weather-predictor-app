# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the newly trained model and encoders
try:
    model = joblib.load('weather_predictor_model.joblib')
    encoders = joblib.load('weather_encoders.joblib')
except FileNotFoundError:
    st.error("Model or encoders not found. Please run train_model.py first.")
    st.stop()

# --- Streamlit App ---

st.set_page_config(layout="wide")
st.title("Accurate Weather Event Predictor")
st.write("Enter real-time weather conditions to predict the current weather event.")

# --- User Inputs in Columns ---
st.subheader("Enter Current Weather Conditions")

col1, col2 = st.columns(2)

with col1:
    temp = st.number_input("Temperature (°C)", value=25.0)
    dew_point = st.number_input("Dew Point Temperature (°C)", value=20.0)
    humidity = st.slider("Relative Humidity (%)", 0, 100, 70)

with col2:
    wind_speed = st.number_input("Wind Speed (km/h)", value=15.0)
    visibility = st.number_input("Visibility (km)", value=10.0)
    pressure = st.slider("Pressure (kPa)", 98.0, 105.0, 101.0)

# --- Prediction Logic ---
if st.button("Predict Weather"):
    # 1. Create a DataFrame with the exact feature names the model was trained on
    input_data = pd.DataFrame({
        'Temp_C': [temp],
        'Dew Point Temp_C': [dew_point],
        'Rel Hum_%': [humidity],
        'Wind Speed_km/h': [wind_speed],
        'Visibility_km': [visibility],
        'Press_kPa': [pressure]
    })

    # 2. Make prediction
    prediction_encoded = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)

    # 3. Decode the prediction to get the weather name (e.g., 'Fog', 'Rain')
    weather_category = encoders['Weather'].inverse_transform([prediction_encoded])[0]

    st.subheader("Prediction Result")
    st.success(f"The predicted weather event is: **{weather_category}**")

    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    # Decode the column names for the chart
    proba_df.columns = encoders['Weather'].inverse_transform(proba_df.columns)
    st.bar_chart(proba_df.T)