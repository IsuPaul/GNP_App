import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xbt
import joblib
import os

st.title("3D Printing Cement Mortar Integrity Predictor")
st.write("Input the parameters below to predict Surface Resistivity (SR).")

# By setting step=0.0, Streamlit allows any number of decimal places
days = st.number_input("Days", step=0.0)
graphene = st.number_input("Graphene Content", step=0.0)
wc = st.number_input("W/C Ratio", step=0.0)
sand_c = st.number_input("Sand/C", step=0.0)
sp_c = st.number_input("SP/C", step=0.0)
vma_c = st.number_input("VMA/C", step=0.0)
upv = st.number_input("UPV", step=0.0)

if st.button("Predict Strength"):
    if os.path.exists('model.pkl'):
        # Load the pre-trained model
        model = joblib.load('model.pkl')

        # Create a dataframe from inputs matching the training feature order
        input_data = pd.DataFrame([[days, graphene, wc, sand_c, sp_c, vma_c, upv]],
                                  columns=['Days', 'Graphene', 'W/C', 'Sand/C', 'SP/C', 'VMA/C', 'UPV'])

        # Perform prediction
        prediction = model.predict(input_data)

        # Output the raw prediction value without rounding constraints
        st.success(f"Predicted Surface Resistivity (SR): {prediction[0]} kΩ-cm")
    else:
        st.error("Model file 'model.pkl' not found. Please ensure it is in the same directory as app.py.")
