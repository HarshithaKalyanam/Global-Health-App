import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- APP TITLE ----------
st.set_page_config(page_title="Disease Predictor", layout="centered")

st.title("🩺 Disease Category Prediction System")
st.write("Enter patient details and predict disease category")

# ---------- MODEL ----------
@st.cache_resource
def train_model():
    df = pd.read_csv("data.csv")

    df = df[['Disease Name', 'Treatment Type',
             'Age Group', 'Gender',
             'Disease Category']].dropna()

    df['text'] = (
        df['Disease Name'].astype(str) + " " +
        df['Treatment Type'].astype(str) + " " +
        df['Age Group'].astype(str) + " " +
        df['Gender'].astype(str)
    )

    X_text = df['text']
    y = df['Disease Category']

    cv = TfidfVectorizer()
    X = cv.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # 🔥 IMPORTANT FIX
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, cv, accuracy


# ---------- CALL MODEL ----------
model, cv, accuracy = train_model()

st.success(f"📊 Model Accuracy: {accuracy:.2f}")

st.markdown("---")
st.subheader("🧾 Patient Input Section")

# ---------- INPUTS ----------
col1, col2 = st.columns(2)

with col1:
    disease = st.selectbox(
        "Disease Name",
        ["COVID-19", "Diabetes", "Hypertension", "Asthma", "Malaria", "Tuberculosis"]
    )

with col2:
    treatment = st.selectbox(
        "Treatment Type",
        ["Vaccination", "Medication", "Therapy", "Surgery", "Lifestyle Change"]
    )

col3, col4 = st.columns(2)

with col3:
    age = st.selectbox("Age Group", ["0-18", "19-35", "36-60", "60+"])

with col4:
    gender = st.selectbox("Gender", ["Male", "Female"])

# ---------- PREDICTION ----------
if st.button("🔍 Predict Disease Category"):

    user_input = f"{disease} {treatment} {age} {gender}".lower()

    data = cv.transform([user_input])
    prediction = model.predict(data)

    st.success(f"🎯 Predicted Disease Category: {prediction[0]}")