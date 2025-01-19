# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 04:15:47 2025

@author: luthr
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model, scaler, and encoder
model = joblib.load('student_performance_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Define the feature columns (same as in your dataset)
features = [
    'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
    'Mjob', 'Fjob', 'reason', 'nursery', 'internet', 'Dalc', 'Walc', 'health', 
    'absences', 'G1', 'G2'
]

# Streamlit app layout
st.title("Student Performance Prediction")

# Create input fields for the user
school = st.selectbox('School', ['GP', 'MS'])
sex = st.selectbox('Sex', ['M', 'F'])
age = st.slider('Age', 15, 22)
address = st.selectbox('Address', ['U', 'R'])
famsize = st.selectbox('Family Size', ['GT3', 'LE3'])
Pstatus = st.selectbox('Parental Status', ['T', 'A'])
Medu = st.slider('Mother Education (0-4)', 0, 4)
Fedu = st.slider('Father Education (0-4)', 0, 4)
Mjob = st.selectbox('Mother Job', ['at_home', 'health', 'other', 'services', 'teacher'])
Fjob = st.selectbox('Father Job', ['at_home', 'health', 'other', 'services', 'teacher'])
reason = st.selectbox('Reason for Choosing School', ['course', 'home', 'reputation', 'other'])
nursery = st.selectbox('Attended Nursery', ['yes', 'no'])
internet = st.selectbox('Internet Access at Home', ['yes', 'no'])
Dalc = st.slider('Workday Alcohol Consumption (1-5)', 1, 5)
Walc = st.slider('Weekend Alcohol Consumption (1-5)', 1, 5)
health = st.slider('Current Health Status (1-5)', 1, 5)
absences = st.slider('Number of School Absences', 0, 93)
G1 = st.slider('Grade 1', 0, 20)
G2 = st.slider('Grade 2', 0, 20)

# Create a dataframe for the input values
input_data = pd.DataFrame({
    'school': [school],
    'sex': [sex],
    'age': [age],
    'address': [address],
    'famsize': [famsize],
    'Pstatus': [Pstatus],
    'Medu': [Medu],
    'Fedu': [Fedu],
    'Mjob': [Mjob],
    'Fjob': [Fjob],
    'reason': [reason],
    'nursery': [nursery],
    'internet': [internet],
    'Dalc': [Dalc],
    'Walc': [Walc],
    'health': [health],
    'absences': [absences],
    'G1': [G1],
    'G2': [G2]
})

# Encode categorical columns
categorical_cols = input_data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    input_data[col] = encoder.transform(input_data[col])

# Scale numerical columns
numerical_cols = input_data.select_dtypes(include=["int64", "float64"]).columns
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Make a prediction
if st.button('Predict'):
    prediction = model.predict(input_data[features])
    st.subheader(f"Predicted Final Grade (G3): {prediction[0]:.2f}")

