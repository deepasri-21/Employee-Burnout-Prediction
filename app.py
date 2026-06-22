import streamlit as st
import pickle
import pandas as pd


# Load Model

model = pickle.load(
    open(
        "model/burnout_model.pkl",
        "rb"
    )
)


scaler = pickle.load(
    open(
        "model/scaler.pkl",
        "rb"
    )
)


# Title

st.title("Employee Burnout Prediction System")

st.write(
    "Predict employee burnout risk using Machine Learning"
)



# User Inputs


gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)


company = st.selectbox(
    "Company Type",
    ["Service","Product"]
)


wfh = st.selectbox(
    "WFH Setup Available",
    ["Yes","No"]
)


designation = st.number_input(
    "Designation",
    min_value=0,
    value=1
)


resource = st.number_input(
    "Resource Allocation",
    min_value=0.0,
    value=5.0
)


fatigue = st.number_input(
    "Mental Fatigue Score",
    min_value=0.0,
    value=5.0
)



# Encoding same as training

if gender == "Male":
    gender = 1
else:
    gender = 0


if company == "Service":
    company = 1
else:
    company = 0


if wfh == "Yes":
    wfh = 1
else:
    wfh = 0



# Prediction

if st.button("Predict Burnout"):


    input_data = pd.DataFrame(
        [[
            gender,
            company,
            wfh,
            designation,
            resource,
            fatigue
        ]],
        columns=[
            "Gender",
            "Company Type",
            "WFH Setup Available",
            "Designation",
            "Resource Allocation",
            "Mental Fatigue Score"
        ]
    )


    # Scaling

    input_scaled = scaler.transform(
        input_data
    )


    # Prediction

    prediction = model.predict(
        input_scaled
    )


    result = prediction[0]


    st.success(
        f"Predicted Burn Rate: {result:.2f}"
    )


    if result >= 0.6:

        st.error(
            "High Burnout Risk"
        )


    elif result >= 0.3:

        st.warning(
            "Medium Burnout Risk"
        )


    else:

        st.success(
            "Low Burnout Risk"
        )