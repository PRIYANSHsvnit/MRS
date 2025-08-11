# omdb_utils.py
import requests

def get_movie_details(title, api_key):
    url = f"http://www.omdbapi.com/?t={title}&plot=full&apikey={api_key}"
    res = requests.get(url).json()
    if res.get("Response") == "True":
        plot = res.get("Plot", "N/A")
        poster = res.get("Poster", "N/A")
        imdb_id = res.get("imdbID", None)
        imdb_link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
        return plot, poster, imdb_link

    return "N/A", "N/A", None