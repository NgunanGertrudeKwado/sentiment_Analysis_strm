import streamlit as st
import joblib
import pandas as pd

# Load the model and vectorizer
model = joblib.load('C:/Users/HP/Documents/Streamlit_Senti_Analysis_Project/lr_model.pkl')
vectorizer = joblib.load('C:/Users/HP/Documents/Streamlit_Senti_Analysis_Project/tfidf_vectorizer.pkl')

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
