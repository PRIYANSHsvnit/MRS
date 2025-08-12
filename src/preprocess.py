# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
import os
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("preprocess.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

def load_data():
    data_path = Path("movies.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing required file: {data_path}")
    
    df = pd.read_csv(data_path)
    required_columns = ["genres", "keywords", "overview", "title"]
    return df[required_columns].dropna().reset_index(drop=True)

def preprocess_text(text, stop_words):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join(word for word in tokens if word not in stop_words)

def main():
    setup_logging()
    logging.info("🚀 Starting preprocessing...")
    
    try:
        download_nltk_resources()
        stop_words = set(stopwords.words('english'))
        
        df = load_data()
        df['combined'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']
        df['cleaned_text'] = df['combined'].apply(lambda x: preprocess_text(x, stop_words))

        tfidf = TfidfVectorizer(max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        joblib.dump(df, 'df_cleaned.pkl')
        joblib.dump(cosine_sim, 'cosine_sim.pkl')
        logging.info("✅ Preprocessing completed successfully")
        
    except Exception as e:
        logging.error(f"❌ Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()