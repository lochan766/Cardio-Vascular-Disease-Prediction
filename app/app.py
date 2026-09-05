#API
from fastapi import FastAPI
import pandas as pd

from app.schema import Cardio
from app.models import load_logistic_models, load_svm_models


app = FastAPI()

model, scaler = load_logistic_models()

svm_model, svm_scaler = load_svm_models()


#get(read), post(insert/create), put(update), delete(remove)

#get-home
@app.get('/')
def home():
    return 'Welcome to Cardiovascular Disease Prediction using Logistic Regression.'


# POST - Prediction

# logistic
# --------------------------------

@app.post("/cardio-predict-logistic")
def predict(data: Cardio):

    # Create DataFrame from user input
    data = pd.DataFrame([
         data.model_dump()#convert user data to json fromat
          
    ])

 # Scale input using trained scaler
 
    scale_data = scaler.transform(data)
    # Make prediction
    prediction = model.predict(scale_data)[0]

    # Convert prediction to readable result
    
    return {
            "prediction": int(prediction),
            "status": 'Likely to be Healthy' if prediction == 0 else "Likely to be Unhealthy."
        }
#SVM 

@app.post("/cardio-predict-svm")
def predict(data: Cardio):

    # Create DataFrame from user input
    data = pd.DataFrame([
         data.model_dump()#convert user data to json fromat
          
    ])

 # Scale input using trained scaler
 
    svm_scale_data = svm_scaler.transform(data)
    # Make prediction
    svm_prediction = model.predict(svm_scale_data)[0]

    # Convert prediction to readable result
    
    return {
            "svm_prediction": int(svm_prediction),
            "status": 'Likely to be Healthy' if svm_prediction == 0 else "Likely to be Unhealthy"
        }





