# Q1. IMPORT REQUIRED LIBRARIES
# Streamlit is used to create an interactive web application
import streamlit as st
# Pandas is used for data handling and DataFrame creation
import pandas as pd
# Joblib is used to load the saved Machine Learning model
import joblib


# Q2. LOAD MODEL AND PREPROCESSING FILES
model = joblib.load("LR_model.pkl")
scaler = joblib.load("scaler.pkl")
encoded_columns = joblib.load("columns.pkl")


# Q3. STREAMLIT PAGE CONFIGURATION
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)

# Q4. DISPLAY TITLE AND DESCRIPTION
st.title(" Ford Car Price Predictor")
st.write("Enter the car details below to predict the selling price.")

# Q5. NUMERICAL INPUT FIELDS

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2025,
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    max_value=300000,
    value=50000
)

tax = st.number_input(
    "Road Tax",
    min_value=0,
    max_value=1000,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.5,
    max_value=10.0,
    value=1.5
)

# Q6. CATEGORICAL INPUT FIELDS
transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual", "Semi-Auto"])

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Electric", "Other"]
)


# Q7. PREDICTION BUTTON
predict_button = st.button(
    " Predict Price")


# Q8. CREATE DATAFRAME AND ENCODE DATA
input_data = pd.DataFrame({
        "year": [year],
        "mileage": [mileage],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize],
        "transmission": [transmission],
        "fuelType": [fuelType]
    })

    # One-Hot Encoding
input_data = pd.get_dummies(input_data)

    # Match training columns
input_data = input_data.reindex(
        columns=encoded_columns,
        fill_value=0
    )

#Q9 AND Q10. COMPLETE PREDICTION APP
if predict_button:

    try:

        # Create DataFrame from user input
        input_data = pd.DataFrame({
            "year": [year],
            "mileage": [mileage],
            "tax": [tax],
            "mpg": [mpg],
            "engineSize": [engineSize],
            "transmission": [transmission],
            "fuelType": [fuelType]
        })
        # Apply One-Hot Encoding
        input_data = pd.get_dummies(input_data)
        # Match training columns
        input_data = input_data.reindex(
            columns=encoded_columns,
            fill_value=0
        )
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        # Predict the car price
        prediction = model.predict(input_scaled)
        # Display success message
        st.success("Prediction Successful! ")
        # Display predicted price
        st.metric(
            "Predicted Car Price",
            f"£ {prediction[0]:,.2f}"
        )
        # Display additional information
        st.info(
            "Price predicted using a trained Linear Regression model."
        )
    except Exception as error:

        # Handle errors during prediction
        st.error(
            "An error occurred while predicting the price."
        )
        st.write("Error Details:", error)