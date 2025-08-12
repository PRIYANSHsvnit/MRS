# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

def should_preprocess():
    """Check if preprocessing is needed"""
    return not os.path.exists('df_cleaned.pkl') or not os.path.exists('cosine_sim.pkl')

def main():
    if not should_preprocess():
        logging.info("✅ Data files already exist")
        return

    logging.info("🚀 Starting preprocessing...")
    nltk.download('punkt')
    nltk.download('stopwords')

    # Load and clean data
    try:
        df = pd.read_csv("movies.csv")
        logging.info("✅ Dataset loaded successfully. Rows: %d", len(df))
    except Exception as e:
        logging.error("❌ Failed to load dataset: %s", str(e))
        raise

    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = re.sub(r"[^a-zA-Z\s]", "", str(text))
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)

    required_columns = ["genres", "keywords", "overview", "title"]
    df = df[required_columns].dropna().reset_index(drop=True)
    df['combined'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']
    df['cleaned_text'] = df['combined'].apply(preprocess_text)

    # Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

    # Calculate similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save files
    joblib.dump(df, 'df_cleaned.pkl')
    joblib.dump(cosine_sim, 'cosine_sim.pkl')
    logging.info("💾 Data saved to disk")

if __name__ == "__main__":
    main()