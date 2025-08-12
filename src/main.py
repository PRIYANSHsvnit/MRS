# app.py
import json
import os
import streamlit as st
from recommend import df, recommend_movies
from omdb_utils import get_movie_details
import subprocess

# Check and generate data files if missing
if not os.path.exists('df_cleaned.pkl') or not os.path.exists('cosine_sim.pkl'):
    with st.spinner("⚙️ First-time setup: Preparing data (this may take a few minutes)..."):
        try:
            subprocess.run(["python", "preprocess.py"], check=True)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Setup failed: {str(e)}")
            st.stop()

# Load CSS
st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)

# Load config
config = json.load(open("config.json"))
OMDB_API_KEY = config["OMDB_API_KEY"]

# App UI
st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation App")

# Movie selection
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