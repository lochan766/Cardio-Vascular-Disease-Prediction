import streamlit as st
import pandas as pd
from models import cardio_predict
import matplotlib.pyplot as plt
import seaborn as sns
import requests

st.header('Cardiovascular Disease Prediction')
st.subheader('Using Logistic Regression')

features, target, X, Y, scaler, model, Y_pred, cr, cm = cardio_predict()

LOGISTIC_API_URL = 'https://cardio-vascular-disease-prediction-v6sw.onrender.com/cardio-predict-logistic'


# -----------------------------
# User Input
# -----------------------------

st.sidebar.header('User Cardio Features')


# Age
age = st.sidebar.slider(
    'Age',
    min_value=26,
    max_value=70,
    value=30
)


# Gender
gender_dict = {
    1: 'Female',
    2: 'Male'
}

gender = st.sidebar.radio(
    'Gender',
    options=list(gender_dict.keys()),
    format_func=lambda x: gender_dict.get(x)
)


# Height
height = st.sidebar.slider(
    'Height',
    min_value=155,
    max_value=200,
    value=160
)


# Weight
weight = st.sidebar.slider(
    'Weight',
    min_value=35,
    max_value=120,
    value=55
)


# Systolic Blood Pressure
ap_hi = st.sidebar.slider(
    'Systolic Pressure',
    min_value=90,
    max_value=200,
    value=120
)


# Diastolic Blood Pressure
ap_lo = st.sidebar.slider(
    'Diastolic Pressure',
    min_value=50,
    max_value=90,
    value=80
)


# Cholesterol
cholesterol_dict = {
    1: 'Low Cholesterol',
    2: 'Mild Cholesterol',
    3: 'High Cholesterol'
}

cholesterol = st.sidebar.radio(
    'Cholesterol',
    options=list(cholesterol_dict.keys()),
    format_func=lambda x: cholesterol_dict.get(x)
)


# Glucose
glucose_dict = {
    1: 'Low Glucose',
    2: 'Mild Glucose',
    3: 'High Glucose'
}

gluc = st.sidebar.radio(
    'Glucose',
    options=list(glucose_dict.keys()),
    format_func=lambda x: glucose_dict.get(x)
)


# Smoking
smoke_dict = {
    0: 'Does not Smoke',
    1: 'Does Smoke'
}

smoke = st.sidebar.radio(
    'Smoke',
    options=list(smoke_dict.keys()),
    format_func=lambda x: smoke_dict.get(x)
)


# Alcohol
alcohol_dict = {
    0: 'Does not Drink Alcohol',
    1: 'Drinks Alcohol'
}

alco = st.sidebar.radio(
    'Alcohol',
    options=list(alcohol_dict.keys()),
    format_func=lambda x: alcohol_dict.get(x)
)


# Physical Activity
active_dict = {
    0: 'Does not do PA',
    1: 'Does PA'
}

active = st.sidebar.radio(
    'Physical Activities',
    options=list(active_dict.keys()),
    format_func=lambda x: active_dict.get(x)
)

# -----------------------------
# Calling API
# -----------------------------

if st.button('Predict Cardio'):

    payload = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }

    try:
        response = requests.post(LOGISTIC_API_URL, json=payload)

        if response.status_code == 200:

            result = response.json()

            if result['prediction'] == 0:
                st.write('No disease found.')
                st.success(
                    'No Cardiovascular Disease Found. '
                    'Likely to be Healthy.'
                )

            else:
                st.write('Disease Found.')
                st.warning(
                    'Cardiovascular Disease Found. '
                    'Likely to be Unhealthy.'
                )

        else:
            st.error(
                f'API Status Code: {response.status_code}'
            )

    except requests.exceptions.RequestException:
        st.error(f'API Server Error: {response.status_code}')
# -----------------------------
# Prediction
# -----------------------------

# if st.button('Predict Cardio'):

#     data = pd.DataFrame([[
#         age,
#         gender,
#         height,
#         weight,
#         ap_hi,
#         ap_lo,
#         cholesterol,
#         gluc,
#         smoke,
#         alco,
#         active
#     ]], columns=features)

#     # Debugging
#     st.write("Input Data:")
#     st.write(data)

#     # st.write("Data Types:")
#     # st.write(data.dtypes)

#     # st.write("Features:")
#     # st.write(features)

#     # Scaling
#     data_scale = scaler.transform(data)

#     # Prediction
#     prediction = model.predict(data_scale)

#     if prediction[0] == 0:
#         st.success('No Cardiovascular Disease found.')
    # else:
    #     st.warning('Cardiovascular Disease found.')


# -----------------------------
# Visualization
# -----------------------------

st.subheader('Visualization')

fig, axes = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='1.0f',
    ax=axes,
    xticklabels=['Predicted Healthy [0]', 'Predicted Unhealthy [1]'],
    yticklabels=['Actual Healthy [0]', 'Actual Unhealthy [1]']
)

axes.set_title('Actual Cardio vs. Predicted Cardio')
axes.set_xlabel('Predicted')
axes.set_ylabel('Actual')

st.pyplot(fig)

plt.close(fig)


# -----------------------------
# Classification Report
# -----------------------------

st.subheader('Classification Report')

data = pd.DataFrame(cr).transpose()

st.dataframe(
    data.style.format(precision=2)
)