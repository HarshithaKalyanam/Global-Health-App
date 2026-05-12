import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("Disease Category Prediction using Naive Bayes")

# Load Dataset
df = pd.read_csv("data.csv")

# Select columns
df = df[['Disease Name', 'Treatment Type',
         'Age Group', 'Gender',
         'Disease Category']]

# Create text column
df['text'] = (
    df['Disease Name'].astype(str) + " " +
    df['Treatment Type'].astype(str) + " " +
    df['Age Group'].astype(str) + " " +
    df['Gender'].astype(str)
)

# Input and output
X = df['text']
y = df['Disease Category']

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# User Input
user_input = st.text_input(
    "Enter Disease Details",
    "COVID-19 Vaccination 36-60 Male"
)

# Prediction Button
if st.button("Predict"):
    
    data = cv.transform([user_input])
    
    prediction = model.predict(data)
    
    st.success(
        f"Predicted Disease Category: {prediction[0]}"
    )