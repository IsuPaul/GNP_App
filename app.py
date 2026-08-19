import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xbt
import joblib
import os

st.title("3D Printing Cement Mortar Integrity Predictor")
st.write("Input the parameters below to predict Surface Resistivity (SR).")

# Create input fields for each feature in your model
days = st.number_input("Days", min_value=1.0, max_value=100.0, value=28.0)
graphene = st.number_input("Graphene Content", min_value=0.0, max_value=5.0, value=0.1)
wc = st.number_input("W/C Ratio", min_value=0.2, max_value=0.5, value=0.3)
sand_c = st.number_input("Sand/C", min_value=1.0, max_value=2.0, value=1.429)
sp_c = st.number_input("SP/C", min_value=0.0, max_value=0.1, value=0.004)
sta_c = st.number_input("VMA/C", min_value=0.0, max_value=0.1, value=0.001)
upv = st.number_input("UPV", min_value=2.0, max_value=5.0, value=3.5)

if st.button("Predict Strength"):
    if os.path.exists('model.pkl'):
        # Load the pre-trained model
        model = joblib.load('model.pkl')
        
        # Create a dataframe from inputs matching the training feature order
        input_data = pd.DataFrame([[days, graphene, wc, sand_c, sp_c, sta_c, upv]],
                                  columns=['Days', 'Graphene', 'W/C', 'Sand/C', 'SP/C', 'VMA/C', 'UPV'])
        
        # Perform prediction
        prediction = model.predict(input_data)
        
        st.success(f"Predicted Surface Resistivity (SR): {prediction[0]:.4f} kΩ-cm")
    else:
        st.error("Model file 'model.pkl' not found. Please ensure it is in the same directory as app.py.")
