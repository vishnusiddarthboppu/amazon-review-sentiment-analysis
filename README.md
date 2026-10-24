# Amazon Review Sentiment Predictor

A machine learning web app that predicts the star rating (1 to 5) of a product review and labels its sentiment as **Positive**, **Neutral** or **Negative**. It is built with scikit-learn and Streamlit.

**Live demo:**https://review-sentiment-predictor.streamlit.app/

## How it works

1. **Preprocessing:** the review is lowercased, punctuation and digits are removed, stopwords are removed, and the remaining words are lemmatized (NLTK).
2. **Vectorization:** the cleaned text is converted to numbers with a TF-IDF vectorizer.
3. **Prediction:** a Linear SVC (Support Vector Classifier) predicts a score from 1 to 5.
4. **Sentiment label:**
   - 4 to 5 stars: Positive
   - 3 stars: Neutral
   - 1 to 2 stars: Negative

The model was trained on the Amazon Fine Food Reviews dataset (`Reviews.csv`). The full workflow, including data cleaning, EDA, training and evaluation, is in the notebook `ML_amazon_reviews_project.ipynb`.

## Project structure

```
.
├── sentiment_review_web.py              # Streamlit app
├── sentiment_review_trained_model.sav   # trained Linear SVC model
├── sentiment_tfidf_trained              # fitted TF-IDF vectorizer
├── requirements.txt                     # Python dependencies
├── ML_amazon_reviews_project.ipynb      # data analysis and model training
└── README.md
```

## Run it locally

1. Clone the repository and go into the folder.
2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Start the app:
   ```
   streamlit run sentiment_review_web.py
   ```
5. Open `http://localhost:8501` in your browser.

On first run the app downloads the NLTK data it needs (stopwords and wordnet), so an internet connection is required once.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, select this repository and branch, and set the main file path to `sentiment_review_web.py`.
4. Click **Deploy**.

## Tech stack

Python, pandas, NLTK, scikit-learn, Streamlit

## Notes and limitations

- Predicting exact star ratings is harder than predicting positive or negative, especially for 3-star reviews.
- The dataset has far more 5-star reviews than other ratings, which can bias predictions.
- The model can struggle with negation (for example, "not good"), sarcasm and very short reviews.
- The saved model must be loaded with the same scikit-learn version it was trained with.

## Author

Vishnu Siddarth Boppu
