import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# Load model
with open('svm_drug_model.pkl', 'rb') as file:
    model, drug_encoder, sex_encoder, bp_encoder, cholesterol_encoder = pickle.load(file)

# Streamlit app
st.title("Drug Type Prediction")

# Get user input for each variable
age_input = st.slider('Enter Age:', 15, 75, 51)
sex_input = st.selectbox('Select Sex:', ['M', 'F'])
bp_input = st.selectbox('Select Blood Pressure Levels (BP):', ['LOW', 'NORMAL','HIGH'])
cholesterol_input = st.selectbox('Select Cholesterol Levels:', ['NORMAL','HIGH'])
Na_to_K_input = st.number_input('Na to Potassium Ration:', min_value=0.00, max_value=50.00, step=0.001, format="%.3f")

# Create a DataFrame with user input
x_new = pd.DataFrame({
    'Age': [age_input],
    'Sex': [sex_input],
    'BP': [bp_input],
    'Cholesterol': [cholesterol_input],
    'Na_to_K': [Na_to_K_input]
})

# Encoding
x_new['Sex'] = sex_encoder.transform(x_new['Sex'])
x_new['BP'] = bp_encoder.transform(x_new['BP'])
x_new['Cholesterol'] = cholesterol_encoder.transform(x_new['Cholesterol'])

# Prediction function
def predict_drug():
    y_pred_new = model.predict(x_new)
    result = drug_encoder.inverse_transform(y_pred_new)
    return result[0]

# "Predict" button
if st.button('Predict'):
    result = predict_drug()
    st.subheader('Prediction Result:')
    st.write(f'{result}')
