import pandas as  pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("data/Cardiovascular_Disease.csv")

df['age'] = df['age']// 365
df = df[(df['ap_hi'].between(90, 200)) & (df['ap_lo'].between(50, 90))]

def cardio_predict():
    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
       'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'

    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 42, stratify = Y
    )

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train) # learn (mean and sd) + implement (xi - mean/sd)
    X_test_scale = scaler.transform(X_test) #implement

    model = LogisticRegression(solver= 'liblinear', class_weight = 'balanced', random_state=42)

    model.fit(X_train_scale, Y_train)
    Y_pred = model.predict(X_test_scale)
    cr = classification_report(Y_test, Y_pred, output_dict=True)
    cm = confusion_matrix(Y_test, Y_pred)

    return features, target, X, Y, scaler, model, Y_pred, cr, cm 


def cardio_predict_svm():
    df_sample = df.sample(n= 5000, random_state= 42)

    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo','cholesterol',
                 'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'  

    X = df_sample[features]
    Y = df_sample[target] 

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 42, stratify = Y
   )   

    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train) 
    X_test_scale = scaler.transform(X_test) 

    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scale, Y_train)
    Y_pred = model.predict(X_test_scale)    

    cm = confusion_matrix(Y_test, Y_pred)
    cr = classification_report(Y_test, Y_pred, output_dict=True)      

    return features, target, X, Y, scaler, model, Y_pred, cm, cr
   