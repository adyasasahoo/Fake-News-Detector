import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("📰 Fake News Detector")

# User input
headline = st.text_input("Enter a news headline to check if it's fake or real:")

if st.button("Check"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        # Transform input and predict
        input_vec = vectorizer.transform([headline])
        prediction = model.predict(input_vec)[0]

        if prediction == 0:
            st.error("🚫 This is likely *FAKE* news.")
        else:
            st.success("✅ This is likely *REAL* news.")