import os
import pickle

import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_resources():
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    model = pickle.load(open(os.path.join(BASE_DIR, "sentiment_review_trained_model.sav"), "rb"))
    tfidf = pickle.load(open(os.path.join(BASE_DIR, "sentiment_tfidf_trained"), "rb"))
    return model, tfidf, set(stopwords.words("english")), WordNetLemmatizer()


def clean_text(text, stop_words, lemmatizer):
    text = (
        pd.Series([text])
        .str.lower()
        .str.replace(r"<.*?", " ", regex=True)
        .str.replace(r"http\s+|www\.\S+", " ", regex=True)
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .iloc[0]
    )
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)


st.title("E-Commerce Review Sentiment Predictor")
# st.caption("App version: v2 (running " + os.path.abspath(__file__) + ")")
st.markdown("Predict review star rating and analyze customer feedback instantly.")

review_text = st.text_area("Type your review here:")

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review first!")
    else:
        try:
            model, tfidf, stop_words, lemmatizer = load_resources()
            cleaned = clean_text(review_text, stop_words, lemmatizer)
            if not cleaned:
                st.warning("Could not find any meaningful words in that review.")
            else:
                score = int(model.predict(tfidf.transform([cleaned]))[0])
                if score >= 4:
                    st.success(f"Sentiment: Positive Aspect (predicted score: {score} out of 5)")
                elif score == 3:
                    st.info(f"Sentiment: Neutral Aspect (predicted score: {score} out of 5)")
                else:
                    st.error(f"Sentiment: Negative Aspect (predicted score: {score} out of 5)")
        except Exception as e:
            st.exception(e)
            