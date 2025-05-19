import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('heart_Satyam.csv')
    return data

data = load_data()

# Title and description
st.title("Heart Disease Prediction App")
st.markdown("Enter the required parameters to predict the likelihood of heart disease.")

# Sidebar - User Input
def user_input_features():
    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ("Male", "Female"))
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 90, 200, 120)
    chol = st.slider("Serum Cholestoral (chol)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.slider("ST Depression Induced (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
    
    sex = 1 if sex == "Male" else 0

    features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame([features])

input_df = user_input_features()

# Model training or loading
@st.cache_resource
def train_model():
    df = data.copy()
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf

model = train_model()

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output
st.subheader("Prediction Result")
heart_disease_result = 'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'
st.write(f"🔍 **Prediction:** {heart_disease_result}")
st.write(f"📊 **Confidence:** {np.max(prediction_proba[0])*100:.2f}%")
