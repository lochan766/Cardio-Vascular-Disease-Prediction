
# ML Models

# import os
# from pyexpat import model
# import pandas as pd
import joblib

# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report


LOGISTIC_MODEL_PATH = "models/logistic/logistic_model.pkl"
LOGISTIC_SCALER_PATH = "models/logistic/logistic_scaler.pkl"

SVM_MODEL_PATH = "models/svm/svm_model.pkl"
SVM_SCALER_PATH = "models/svm/svm_scaler.pkl"

# #Load ML-Binary Models
def load_logistic_models():
    model = joblib.load(LOGISTIC_MODEL_PATH)
    scaler = joblib.load(LOGISTIC_SCALER_PATH)
    return model, scaler

def load_svm_models():
    svm_model = joblib.load(SVM_MODEL_PATH)
    svm_scaler = joblib.load(SVM_SCALER_PATH)

    return svm_model, svm_scaler




# -----------------------------
# Load Dataset
# -----------------------------

# df = pd.read_csv("data/Cardiovascular_Disease.csv")


# # Convert age from days to years
# df["age"] = df["age"]//365


# # Remove unrealistic blood pressure values
# df = df[
#     (df["ap_hi"].between(90, 200)) &
#     (df["ap_lo"].between(50, 90))
# ]

# df = df[(df['height'].between(155, 200)) & (df['weight'].between(35, 120))]
# # # -----------------------------
# # # Model Paths
# # # -----------------------------

# SVM_MODEL_PATH = "models/svm/svm_model.pkl"
# SVM_SCALER_PATH = "models/svm/svm_scaler.pkl"

# #take  sample

# df_sample = df.sample(n=5000, random_state = 42)




# # # -----------------------------
# # # Train SVM
# # # -----------------------------

# def cardio_predict():

#     features = [
#         "age",
#         "gender",
#         "height",
#         "weight",
#         "ap_hi",
#         "ap_lo",
#         "cholesterol",
#         "gluc",
#         "smoke",
#         "alco",
#         "active"
#     ]

#     target = "cardio"


#     # Features and target
#     X = df_sample[features]
#     Y = df_sample[target]


#     # Train-test split
#     X_train, X_test, Y_train, Y_test = train_test_split(
#         X,
#         Y,
#         test_size=0.2,
#         random_state=42,
#         stratify=Y
#     )


#     # -----------------------------
#     # Standardization
#     # -----------------------------

#     scaler = StandardScaler()

#     X_train_scale = scaler.fit_transform(X_train)
#     X_test_scale = scaler.transform(X_test)


#     # -----------------------------
#     # SVM model
#     # -----------------------------

#     model = SVC(kernel='rbf', C=1.0, gamma='scale')
   


#     # Train model
#     model.fit(X_train_scale, Y_train)


#     # -----------------------------
#     # Prediction
#     # -----------------------------

#     model.predict(X_test_scale)


#     # -----------------------------
#     # Save Model and Scaler
#     # -----------------------------

#     joblib.dump(model, SVM_MODEL_PATH)
#     joblib.dump(scaler, SVM_SCALER_PATH)


#     return model, scaler



