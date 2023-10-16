!pip install streamlit

import streamlit as st
import joblib

# Load your machine learning model
model = joblib.load('your_model.joblib')

# Define the Streamlit app
st.title('News Classification App')

user_input = st.text_area('Enter a news article:')
if st.button('Classify'):
    prediction = model.predict([user_input])[0]
    st.write('Prediction:', 'Fake' if prediction == 0 else 'Real')

# streamlit run fake_news_classification (2).py
streamlit run fake_news_classification (2).py --server.port 8502
