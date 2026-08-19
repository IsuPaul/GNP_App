import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xbt
import joblib
import os

st.title("3D Printing Cement Mortar Integrity Predictor")
st.write("Input the parameters below to predict Surface Resistivity (SR).")

# value=None ensures the field starts empty
# step=None allows the browser to handle arbitrary precision without forced rounding
days = st.number_input("Days", value=None, step=None)
graphene = st.number_input("Graphene Content", value=None, step=None)
wc = st.number_input("W/C Ratio", value=None, step=None)
sand_c = st.number_input("Sand/C", value=None, step=None)
sp_c = st.number_input("SP/C", value=None, step=None)
vma_c = st.number_input("VMA/C", value=None, step=None)
upv = st.number_input("UPV", value=None, step=None)

if st.button("Predict Strength"):
    # Check if all inputs are filled
    inputs = [days, graphene, wc, sand_c, sp_c, vma_c, upv]
    if any(val is None for val in inputs):
        st.warning("Please enter values for all fields before predicting.")
    elif os.path.exists('model.pkl'):
        # Load the pre-trained model
        model = joblib.load('model.pkl')

        # Create a dataframe from inputs matching the training feature order
        input_data = pd.DataFrame([[days, graphene, wc, sand_c, sp_c, vma_c, upv]],
                                  columns=['Days', 'Graphene', 'W/C', 'Sand/C', 'SP/C', 'VMA/C', 'UPV'])

        # Perform prediction
        prediction = model.predict(input_data)

        # Display result with high precision
        st.success(f"Predicted Surface Resistivity (SR): {prediction[0]} kΩ-cm")
    else:
        st.error("Model file 'model.pkl' not found.")
