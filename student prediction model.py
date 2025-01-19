# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 04:15:47 2025

@author: luthr
"""
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# App title
st.title("Student Performance Prediction")

# Path to the dataset on your computer
dataset_path = r"C:\path\to\your\dataset.csv"  # Replace with the actual path to your dataset

# Step 1: Load the dataset
try:
    data = pd.read_csv(dataset_path)
    st.write("Dataset Loaded Successfully!")
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Step 2: Validate dataset columns
    required_columns = [
        'school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
        'Mjob', 'Fjob', 'reason', 'nursery', 'internet', 'Dalc', 'Walc', 'health', 
        'absences', 'G1', 'G2', 'G3'
    ]
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        st.error(f"The dataset is missing the following columns: {missing_cols}")
    else:
        # Step 3: Preprocess the dataset
        X = data.drop(columns=["G3"])
        y = data["G3"]

        # Encode categorical variables
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        # Scale numerical variables
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

        # Step 4: Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 5: Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Step 6: Evaluate the model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Model Performance:")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Step 7: Make predictions for the entire dataset
        data['Predicted G3'] = model.predict(X)

        # Display predictions
        st.subheader("Predictions")
        st.dataframe(data[["G3", "Predicted G3"]])

        # Download option for predictions
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="student_predictions.csv",
            mime="text/csv",
        )

except FileNotFoundError:
    st.error(f"The file at {dataset_path} was not found. Please check the path.")
