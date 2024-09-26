import streamlit as st
import joblib
import pandas as pd

# Load the model and vectorizer from the current directory
model = joblib.load('lr_model.pkl')  # No need for 'C:/Users/HP/Documents/...'
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
st.title("Sentiment Analysis App")

# Input text
user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    # Transform the input using the vectorizer
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)

    # Display the result
    if prediction[0] == 1:
        st.success("This review is Positive!")
    else:
        st.error("This review is Negative!")
