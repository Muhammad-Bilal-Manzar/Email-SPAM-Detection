import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

@st.cache_data
def load_model_and_vectorizer():
    model = pickle.load(open("spam_ham_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Streamlit UI
st.title("Spam/Ham Email Classifier")
email_message = st.text_area("Enter your email message:")

if email_message:
    email_features = vectorizer.transform([email_message])  # Vectorize using loaded vectorizer
    prediction = model.predict(email_features)
    prediction_proba = model.predict_proba(email_features)

    if prediction == 0:
        st.write("Prediction: Spam")
    else:
        st.write("Prediction: Ham")

    st.write(f"Confidence: {prediction_proba.max() * 100:.2f}%")
