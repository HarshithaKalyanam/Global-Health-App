import streamlit as st
import pandas as pd

st.markdown(
    """
    <style>

    /* App background */
    .stApp {
        background-color: #eaf2f8;
    }

    /* General text */
    h1, h2, h3, p, label {
        color: black !important;
    }

    /* Selectbox main area */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #f2f2f2 !important;
        color: black !important;
        border-radius: 8px;
    }

    /* Selected dropdown text */
    div[data-baseweb="select"] span {
        color: black !important;
    }

    /* Dropdown arrow */
    .stSelectbox svg {
        fill: black !important;
    }

    /* Dropdown popup */
    div[role="listbox"] {
        background-color: #2b2b2b !important;
    }

    /* Dropdown options */
    div[role="option"] {
        background-color: #2b2b2b !important;
    }

    /* Dropdown option text */
    div[role="option"] * {
        color: white !important;
    }

    /* Hover effect */
    div[role="option"]:hover {
        background-color: #444 !important;
    }

    /* Predict button */
    .stButton > button {
        background-color: #2E86C1 !important;
        color: white !important;
        border-radius: 10px;
        border: none;
        height: 45px;
        width: 100%;
        font-size: 16px;
    }

    .stButton > button:hover {
        background-color: #1B4F72 !important;
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

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