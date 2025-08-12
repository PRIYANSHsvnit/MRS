# recommend.py
import joblib
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Load data
try:
    if not os.path.exists('df_cleaned.pkl'):
        raise FileNotFoundError("Data files missing - run preprocess.py first")
        
    logging.info("🔁 Loading data...")
    df = joblib.load('df_cleaned.pkl')
    cosine_sim = joblib.load('cosine_sim.pkl')
    logging.info("✅ Data loaded successfully. Movies: %d", len(df))
except Exception as e:
    logging.error("❌ Data loading error: %s", str(e))
    raise

def recommend_movies(movie_name, top_n=5):
    logging.info("🎬 Recommendations for: '%s'", movie_name)
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Movie not found")
        return None
        
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    
    result_df = df.iloc[movie_indices].copy()
    result_df['similarity'] = [i[1] for i in sim_scores]
    result_df = result_df[['title', 'similarity']].reset_index(drop=True)
    result_df.index = result_df.index + 1
    result_df.index.name = "Rank"
    
    logging.info("✅ Generated %d recommendations", len(result_df))
    return result_df