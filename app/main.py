import joblib
import os

import streamlit as st
import numpy as np

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model_files/model_1.pkl"
scaler_path = f"{working_dir}/trained_model_files/scaler_1.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


def preprocess_input(input_set):
    # TODO: Handle the input before processing it to model.
    input_data_as_numpy_array = np.asarray(input_set)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    input_data_std = scaler.transform(input_data_reshaped)

    return input_data_std


def predict(input_set):
    # TODO: Preprocess and process the data to Predict the final Outcome
    preprocessed_data = preprocess_input(input_set)
    prediction = model.predict(preprocessed_data)
    return prediction[0]


st.title("Machine Learning Mini Project")
st.subheader("Banks can predict whether a customer would default on loans or not")

dependents = st.number_input("No. of dependents: ", placeholder="Enter the number of dependents", step=1,
                             value=None)

graduate = st.selectbox(
    "Are you a Graduate?",
    ("YES", "NO"),
)

st.write("You selected:", graduate)
graduate_true = 1 if graduate == "YES" else 0

self_employed = st.selectbox(
    "Are you self-employed?",
    ("YES", "NO"),
)

st.write("You selected:", self_employed)
self_employed_true = 1 if self_employed == "YES" else 0

income = st.number_input("Income per annum ", placeholder="Enter the income per annum", value=None)
loan = st.number_input("Loan Amount", placeholder="Enter the total loan amount", value=None)

loan_term = st.number_input("Loan Term", placeholder="Enter the loan term in year", value=None)
cibil_score = st.number_input("CIBIL Score", placeholder="Enter the CIBIL Score", value=None)

res_assests = st.number_input("Residential Asset Value", placeholder="Enter the residential asset value",
                              value=None)
com_assests = st.number_input("Commercial Asset Value", placeholder="Enter the commercial asset value",
                              value=None)

input_data = (dependents, graduate_true, self_employed_true, income, loan, loan_term, cibil_score, res_assests, com_assests)


if st.button('Submit'):
    # Check if all fields are filled
    if dependents is None:
        st.error('Please enter the no. of Dependents.')
    elif income is None:
        st.error('Please enter your annual income.')
    elif loan is None:
        st.error('Please enter a valid loan amount.')
    elif loan_term is None:
        st.error('Please enter a valid loan term.')
    elif graduate_true not in [0, 1]:
        st.error('Please select your education status.')
    elif self_employed_true not in [0, 1]:
        st.error('Please select your employment status.')
    elif cibil_score is None:
        st.error('Please enter the CIBIL Score.')
    elif res_assests is None:
        st.error('Please enter the Residential Asset Value.')
    elif com_assests is None:
        st.error('Please enter the Commercial Asset Value.')
    else:
        prediction = predict(input_data)
        if prediction:
            st.success("Will not default on Loan.")
        else:
            st.warning("Will default on Loan.")
