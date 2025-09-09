import streamlit as st
import tensorflow as tf
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import pandas as pd 
import pickle

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the pre-trained encoders
with open('gender_lable_encoder.pkl','rb') as file:
    lable_encoder_gender = pickle.load(file)

with open('OneHotEncoder_geaography.pkl','rb') as file:
    geo_encoder = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler_encoder = pickle.load(file)

# UI to get user input
geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', lable_encoder_gender.classes_)
age = st.selectbox('Age', list(range(18, 93)))
balance = st.selectbox('Balance', [0, 50000, 70000, 100000, 150000, 200000])
credit_score = st.selectbox('CreditScore', list(range(300, 851)))
tenure = st.selectbox('Tenure', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
num_of_products = st.selectbox('NumOfProducts', [1, 2, 3, 4])
has_cr_card = st.selectbox('HasCrCard', [0, 1])
is_active_member = st.selectbox('IsActiveMember', [0, 1])
estimated_salary = st.selectbox('EstimatedSalary', [0, 50000, 100000, 101348.88, 150000, 200000])

# Prepare the input data in a DataFrame
# Note: The order of columns here must match the order in the trained model
# The training data in exp.ipynb shows the order is:
# CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [lable_encoder_gender.transform([gender])[0]], # Transform gender here
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Apply one-hot encoding for the geography column separately
geo_encoded = geo_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=geo_encoder.get_feature_names_out(['Geography']))

# Concatenate the numerical/gender data with the one-hot encoded geography data
final_input_df = pd.concat([input_df, geo_encoded_df], axis=1)

# Scale the final DataFrame
# The scaler was trained on a NumPy array, so transform the DataFrame to a NumPy array before scaling
input_data_scaled = scaler_encoder.transform(final_input_df)

# Make the prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display the prediction result
st.write(f"Prediction Probability: {prediction_prob:.4f}")
if prediction_prob > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write("The customer is not likely to churn.")