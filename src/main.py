# app.py
import json
import os
import streamlit as st
import subprocess
import sys
from pathlib import Path

# Try to import dependencies, if fails install them
try:
    from recommend import df, recommend_movies
    from omdb_utils import get_movie_details
except ImportError:
    st.warning("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    st.experimental_rerun()

# Check and generate data files if missing
DATA_FILES = ['df_cleaned.pkl', 'cosine_sim.pkl']
if not all(Path(f).exists() for f in DATA_FILES):
    with st.spinner("⚙️ First-time setup: Preparing data..."):
        try:
            subprocess.run([sys.executable, "preprocess.py"], check=True)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Setup failed: {str(e)}. Please check if 'movies.csv' exists.")
            st.stop()

# Load CSS
css_path = Path("style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Load config
try:
    config = json.load(open("config.json"))
    OMDB_API_KEY = config["OMDB_API_KEY"]
except Exception as e:
    st.error(f"Config error: {str(e)}")
    st.stop()

# App UI
st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation App")

try:
    from recommend import df, recommend_movies
    # Rest of your existing UI code...
    movie_list = sorted(df['title'].dropna().unique())
    selected_movie = st.selectbox("🎬 Select a movie:", movie_list)

    if st.button("🚀 Recommend Similar Movies"):
        with st.spinner("Finding similar movies..."):
            recommendations = recommend_movies(selected_movie)
            if recommendations is None or recommendations.empty:
                st.warning("Sorry, no recommendations found.")
            else:
                st.success("Top similar movies:")
                for _, row in recommendations.iterrows():
                    movie_title = row['title']
                    plot, poster, imdb_link = get_movie_details(movie_title, OMDB_API_KEY)

                    with st.container():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if poster != "N/A":
                                st.image(poster, width=100)
                            else:
                                st.write("❌ No Poster Found")
                        with col2:
                            st.markdown(f"### [{movie_title}]({imdb_link})")
                            st.markdown(f"*{plot}*" if plot != "N/A" else "_Plot not available_")
except Exception as e:
    st.error(f"Application error: {str(e)}")