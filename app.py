import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xbt
import joblib
import os
from datetime import datetime

# Configuration
LOG_FILE = 'usage_logs.csv'
DATA_STORAGE = 'collected_data.csv'
MODEL_FILE = 'model.pkl'
BASE_DATA = 'Graphene_ML2.csv'

def log_usage():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_log = pd.DataFrame({'timestamp': [now]})
    df_log.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

def save_new_data(features, prediction):
    new_row = features.copy()
    new_row['SR'] = prediction
    new_row['timestamp'] = datetime.now()
    new_row.to_csv(DATA_STORAGE, mode='a', header=not os.path.exists(DATA_STORAGE), index=False)

def retrain_model():
    if os.path.exists(DATA_STORAGE) and os.path.exists(BASE_DATA):
        base_df = pd.read_csv(BASE_DATA, encoding='latin-1')
        new_df = pd.read_csv(DATA_STORAGE).drop(columns=['timestamp'])
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        
        X = combined_df.drop('SR', axis=1)
        y = combined_df['SR']
        
        new_model = xbt.XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=1000)
        new_model.fit(X, y)
        joblib.dump(new_model, MODEL_FILE)
        return len(combined_df)
    return None

st.set_page_config(page_title="Concrete Integrity AI", layout="wide")
st.title("🏗️ Self-Learning 3D Concrete Predictor")
st.write("Enter parameters to predict Compressive Strength (SR). The model learns from every input.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Parameters")
    Age = st.number_input("Age of Specimen (Days)", value=28.0)
    GNP = st.number_input("Graphene Nanoplatelets (GNP) %", value=0.1)
    wc = st.number_input("Water/Cement (W/C) Ratio", value=0.3)
    upv = st.number_input("Ultrasonic Pulse Velocity (UPV)", value=3.5)
    # These are typically constant in your provided dataset logic
    sand = st.number_input("Sand/C", value=1.429)
    sp = st.number_input("SP/C", value=0.0046)
    vma = st.number_input("VMA/C", value=0.0011)

if st.button("Run Prediction & Retrain"):
    input_df = pd.DataFrame([[Age, GNP, wc, sand, sp, vma, upv]], 
                            columns=['Age_of_Specimen', 'GNP', 'W/C', 'Sand/C', 'SP/C', 'VMA/C', 'UPV'])
    
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        pred = model.predict(input_df)[0]
        
        st.success(f"### Predicted SR: {pred:.2f} MPa")
        
        save_new_data(input_df, pred)
        log_usage()
        
        samples = retrain_model()
        if samples:
            st.info(f"✅ Model updated with {samples} total records.")
    else:
        st.error("Model file (model.pkl) not found! Please ensure it is in the repository.")

with col2:
    st.subheader("System Activity")
    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        logs['timestamp'] = pd.to_datetime(logs['timestamp'])
        st.line_chart(logs.groupby(logs['timestamp'].dt.date).size())
    else:
        st.write("No activity logged yet.")
