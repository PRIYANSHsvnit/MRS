# app.py
import json
import streamlit as st
from recommend import df, recommend_movies
from omdb_utils import get_movie_details
st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)


config = json.load(open("config.json"))

# OMDB api key
OMDB_API_KEY = config["OMDB_API_KEY"]

st.set_page_config(
    page_title="Movie Recommendation App",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation App")

# Using 'title'
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