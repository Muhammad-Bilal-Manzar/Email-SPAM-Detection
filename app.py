import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load pre-trained model and vectorizer
@st.cache_data
def load_model():
    model = pickle.load(open("spam_ham_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.title("Spam/Ham Email Classifier")

# User input: Email message
email_message = st.text_area("Enter your email message:")

# Classify the email
if email_message:
    # Vectorize the input email message
    email_features = vectorizer.transform([email_message])

    # Predict whether the email is spam or ham
    prediction = model.predict(email_features)
    prediction_proba = model.predict_proba(email_features)

    # Display the result
    if prediction == 0:
        st.write("Prediction: Spam")
    else:
        st.write("Prediction: Ham")

    st.write(f"Confidence: {prediction_proba.max() * 100:.2f}%")

