import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load pre-trained model
@st.cache_data
def load_model():
    return pickle.load(open("spam_ham_model.pkl", "rb"))

model = load_model()

# Streamlit UI
st.title("Spam/Ham Email Classifier")

# User input: Email message
email_message = st.text_area("Enter your email message:")

# Classify the email
if email_message:
    # Initialize and fit the vectorizer on the input text only
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    email_features = vectorizer.fit_transform([email_message])  # Vectorize the input email message

    # Predict whether the email is spam or ham
    prediction = model.predict(email_features)
    prediction_proba = model.predict_proba(email_features)

    # Display the result
    if prediction == 0:
        st.write("Prediction: Spam")
    else:
        st.write("Prediction: Ham")

    st.write(f"Confidence: {prediction_proba.max() * 100:.2f}%")
