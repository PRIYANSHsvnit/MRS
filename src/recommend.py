import os
import joblib
import logging
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🔁 Checking for preprocessed files...")

# Files we need
required_files = ['df_cleaned.pkl', 'cosine_sim.pkl']

# If missing, run preprocessing
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    logging.warning("⚠️ Missing files: %s", ", ".join(missing_files))
    logging.info("🏗 Running preprocessing script...")
    try:
        subprocess.run(['python', 'preprocess.py'], check=True)
        logging.info("✅ Preprocessing complete.")
    except subprocess.CalledProcessError as e:
        logging.error("❌ Failed to run preprocessing: %s", str(e))
        raise e
else:
    logging.info("✅ All required files found.")

# Load data
try:
    df = joblib.load('df_cleaned.pkl')
    cosine_sim = joblib.load('cosine_sim.pkl')
    logging.info("✅ Data loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load required files: %s", str(e))
    raise e


def recommend_movies(movie_name, top_n=5):
    logging.info("🎬 Recommending movies for: '%s'", movie_name)
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        logging.warning("⚠️ Movie not found in dataset.")
        return None
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    logging.info("✅ Top %d recommendations ready.", top_n)
    # Create DataFrame with clean serial numbers starting from 1
    result_df = df[['title']].iloc[movie_indices].reset_index(drop=True)
    result_df.index = result_df.index + 1  # Start from 1 instead of 0
    result_df.index.name = "S.No."

    return result_df
