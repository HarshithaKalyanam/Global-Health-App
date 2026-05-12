import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Disease Predictor", layout="centered")

st.title("🩺 Disease Category Prediction System")
st.write("Enter details using dropdowns and predict disease category")

if st.checkbox("📂 Show Dataset"):
    df = pd.read_csv("data.csv")
    st.dataframe(df.head())

# ---------- MODEL ----------
@st.cache_resource
def train_model():
    df = pd.read_csv("data.csv")

    df = df[['Disease Name', 'Treatment Type',
             'Age Group', 'Gender',
             'Disease Category']]

    df['text'] = (
        df['Disease Name'].astype(str) + " " +
        df['Treatment Type'].astype(str) + " " +
        df['Age Group'].astype(str) + " " +
        df['Gender'].astype(str)
    )

    X = df['text']
    y = df['Disease Category']

    cv = CountVectorizer()
    X = cv.fit_transform(X)

    model = MultinomialNB()
    model.fit(X, y)

    accuracy = model.score(X, y)


    return model, cv, accuracy

model, cv , accuracy = train_model()
st.write(f"📊 Model Accuracy: {accuracy:.2f}")

st.markdown("---")
st.subheader("🧾 Patient Input Section")

# ---------- UI ----------
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    disease = st.selectbox(
    "Disease Name",
    [
        "COVID-19",
        "Diabetes",
        "Hypertension",
        "Asthma",
        "Malaria",
        "Tuberculosis"
    ]
)
with col2:  
    treatment = st.selectbox(
    "Treatment Type",
    [
        "Vaccination",
        "Medication",
        "Therapy",
        "Surgery",
        "Lifestyle Change"
    ]
)


col3, col4 = st.columns(2)

with col3:
    age = st.selectbox("Age Group", ["0-18", "19-35", "36-60", "60+"])

with col4:
    gender = st.selectbox("Gender", ["Male", "Female"])

# ---------- PREDICTION ----------
if st.button("🔍 Predict Category"):
    user_input = disease + " " + treatment + " " + age + " " + gender

    data = cv.transform([user_input])
    prediction = model.predict(data)

    st.success(f"🎯 Predicted Disease Category: {prediction[0]}")
