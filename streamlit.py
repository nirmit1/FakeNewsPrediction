import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your machine learning model
model = joblib.load('your_model.joblib')

# Preprocessing function
def text_preprocessing_user(data):
    lm = WordNetLemmatizer()
    preprocess_data = []
    review = re.sub(r'[^a-zA-Z0-9]', ' ', data)
    review = review.lower()
    review = nltk.word_tokenize(review)
    review = [lm.lemmatize(x) for x in review if x not in stopwords.words('english')]
    review = " ".join(review)
    preprocess_data.append(review)
    return preprocess_data

# Streamlit app
st.title("News Classification App")
st.write("This app can classify news articles into 'Real' or 'Fake'.")

# Input text area for the user
user_input = st.text_area("Enter a news article", "")

if st.button("Classify"):
    if user_input:
        # Perform preprocessing
        processed_input = text_preprocessing_user(user_input)

        # Vectorize the input
        tf = TfidfVectorizer()
        data = tf.transform(processed_input)

        # Make a prediction
        prediction = model.predict(data)

        # Display the result
        if prediction[0] == 0:
            st.write("The News Is Fake")
        else:
            st.write("The News Is Real")

st.sidebar.text("About")
st.sidebar.info("This is a simple News Classification app using Streamlit.")

streamlit run News Classification App.py --server.port 8502

