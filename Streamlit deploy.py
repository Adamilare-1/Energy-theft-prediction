# -*- coding: utf-8 -*-
"""
Created on Mon May 19 04:40:27 2025

@author: Adamilare
"""

import streamlit as st
import pandas as pd
import joblib
import os
import requests
from datetime import datetime

# === Download model from Google Drive if not present ===
MODEL_URL = "https://drive.google.com/uc?export=download&id=1gk7RZOZ9zHh-qfzzp1qh5D3k7nROC-En"
MODEL_PATH = "electricity_theft_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.info("🔄 Downloading model, please wait...")
    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    st.success("✅ Model downloaded successfully!")

# === Load the trained pipeline model ===
model = joblib.load(MODEL_PATH)

# Create input widgets
st.title('⚡ Electricity Theft Detection System')
st.markdown("""
This app predicts potential electricity theft using machine learning. 
Enter the customer details below to analyze theft risk.
""")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.header("Consumption Data")
        billed_cons = st.number_input("Billed Consumption (kWh)", min_value=0.0, value=300.0)
        actual_cons = st.number_input("Actual Consumption (kWh)", min_value=0.0, value=500.0)
        avg_voltage = st.number_input("Average Voltage (V)", min_value=0.0, value=200.0)
        peak_cons = st.number_input("Peak Time Consumption (kWh)", min_value=0.0, value=200.0)
        off_peak_cons = st.number_input("Off-Peak Consumption (kWh)", min_value=0.0, value=300.0)

    with col2:
        st.header("Customer Details")
        power_factor = st.number_input("Power Factor", min_value=0.0, max_value=2.0, value=0.95)
        num_outages = st.number_input("Number of Outages", min_value=0, value=2)
        customer_type = st.selectbox("Customer Type", ["Residential", "Commercial", "Industrial"])
        location = st.selectbox("Location", ["urban", "suburban", "rural"])
        payment_hist = st.selectbox("Payment History", ["Good", "Poor"])
        peak_time = st.time_input("Peak Usage Time", value=datetime.strptime("20:00", "%H:%M"))

    usage_month = st.date_input("Usage Month", value=datetime(2024, 1, 1))

    submitted = st.form_submit_button("Predict Theft Risk")

if submitted:
    try:
        # Process time inputs
        peak_hour = peak_time.hour
        time_of_day = "night"
        if 5 <= peak_hour < 12:
            time_of_day = "morning"
        elif 12 <= peak_hour < 17:
            time_of_day = "afternoon"
        elif 17 <= peak_hour < 21:
            time_of_day = "evening"

        # Process month into year/month_num FIRST
        input_month = pd.to_datetime(usage_month)
        year = input_month.year
        month_num = input_month.month

        # Construct feature DataFrame (with year/month_num instead of raw date)
        input_data = pd.DataFrame([{
            'billed_consumption': float(billed_cons),
            'actual_consumption': float(actual_cons),
            'average_voltage': float(avg_voltage),
            'peak_time_consumption': float(peak_cons),
            'off_peak_consumption': float(off_peak_cons),
            'power_factor': float(power_factor),
            'num_outages': int(num_outages),
            'customer_type': str(customer_type),  # Keep as string
            'location': str(location),            # Keep as string
            'payment_history': str(payment_hist), # Keep as string
            'year': int(year),                    # Add processed year
            'month_num': int(month_num),          # Add processed month
            'peak_hour': int(peak_hour),
            'time_of_day': str(time_of_day),      # Keep as string
            'consumption_diff': float(actual_cons - billed_cons),
            'consumption_ratio': float(actual_cons / billed_cons if billed_cons != 0 else 0),
            'peak_offpeak_ratio': float(peak_cons / off_peak_cons if off_peak_cons != 0 else 0)
        }])

        # Debug: Show processed features
        st.write("Processed Input Features:", input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        
        
        # Display results
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error(f"🚨 High Risk of Theft Detected ({proba*100:.1f}% probability)")
            st.markdown("""
                **Recommended Actions:**
                - Schedule physical inspection  
                - Verify meter functionality  
                - Analyze consumption patterns  
            """)
        else:
            st.success(f"✅ Normal Usage Detected ({proba*100:.1f}% probability)")
            st.markdown("""
                **Status:** No suspicious activity detected
            """)

        # Show input features
        with st.expander("View Processed Input Features"):
            st.dataframe(input_data.T.style.background_gradient(cmap='Blues'))

    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Sidebar info
st.sidebar.header("About")
st.sidebar.markdown("""
**Machine Learning Model Details:**
- Model Type: Ensemble Classifier  
- Features: 25 engineered features  
- Accuracy: ~92% (test set)  
- Precision: ~89%  
- Recall: ~93%
""")

st.sidebar.markdown("""
**How to Use:**
1. Fill all input fields  
2. Click 'Predict Theft Risk'  
3. View results and recommendations  
""")
