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

def log_usage():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_log = pd.DataFrame({'timestamp': [now]})
    df_log.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

def save_new_data(features, prediction):
    # Save new user inputs to a local CSV for future retraining
    new_row = features.copy()
    new_row['SR'] = prediction
    new_row['timestamp'] = datetime.now()
    new_row.to_csv(DATA_STORAGE, mode='a', header=not os.path.exists(DATA_STORAGE), index=False)

def retrain_model():
    # In a real scenario, you'd use ground truth, but here we adapt to new inputs
    if os.path.exists(DATA_STORAGE) and os.path.exists('Graphene_ML2.csv'):
        base_df = pd.read_csv('Graphene_ML2.csv')
        new_df = pd.read_csv(DATA_STORAGE).drop(columns=['timestamp'])
        combined_df = pd.concat([base_df, new_df], ignore_index=True)
        
        X = combined_df.drop('SR', axis=1)
        y = combined_df['SR']
        
        new_model = xbt.XGBRegressor(learning_rate=0.1, max_depth=3, n_estimators=1000)
        new_model.fit(X, y)
        joblib.dump(new_model, MODEL_FILE)
        return len(combined_df)
    return None

st.title("Self-Learning 3D Printing Integrity Predictor")
st.write("The model automatically learns from new inputs provided to the system.")

# Inputs
Age = st.number_input("Age_of_Specimen", value=28.0)
GNP = st.number_input("GNP Content", value=0.1)
wc = st.number_input("W/C Ratio", value=0.3)
sand = st.number_input("Sand/C", value=1.4)
sp = st.number_input("SP/C", value=0.004)
vma = st.number_input("VMA/C", value=0.001)
upv = st.number_input("UPV", value=3.5)

if st.button("Predict & Train"):
    input_df = pd.DataFrame([[Age, GNP, wc, sand, sp, vma, upv]], 
                            columns=['Age_of_Specimen', 'GNP', 'W/C', 'Sand/C', 'SP/C', 'VMA/C', 'UPV'])
    
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        pred = model.predict(input_df)[0]
        st.success(f"Predicted SR: {pred:.4f} kΩ-cm")
        
        # 1. Save data for retraining
        save_new_data(input_df, pred)
        # 2. Log usage
        log_usage()
        # 3. Retrain (In production, this might be scheduled, here it is immediate for demo)
        total_samples = retrain_model()
        st.info(f"Model retrained! Now utilizing {total_samples} data points.")
    else:
        st.error("Initial model.pkl not found. Please export it from the notebook.")

# Trend View
st.divider()
st.subheader("Usage Activity")
if os.path.exists(LOG_FILE):
    logs = pd.read_csv(LOG_FILE)
    logs['timestamp'] = pd.to_datetime(logs['timestamp'])
    st.line_chart(logs.groupby(logs['timestamp'].dt.date).size())
