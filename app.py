import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def hybrid_recommendation(song, user_id=None):
    cb_recommendations, cb_posters = recommend(song)
    
    cf_recommendations = []
    if user_id:
        all_songs = music['extracted_song_id'].unique()
        predictions = [
            (song_id, final_algorithm.predict(user_id, song_id).est) for song_id in all_songs
        ]
        cf_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

    hybrid_recommendations = []
    seen_songs = set()
    for i in range(5):

        if i < len(cb_recommendations):
            song_cb = cb_recommendations[i]
            if song_cb not in seen_songs:
                hybrid_recommendations.append((song_cb, cb_posters[i]))
                seen_songs.add(song_cb)
                print("in cb "+song_cb+"\n")
        if i < len(cf_recommendations):
            song_cf_id = cf_recommendations[i][0]  
            if song_cf_id not in seen_songs:
                song_cf_name = music[music['extracted_song_id'] == song_cf_id]['song'].values[0] \
                if not music[music['extracted_song_id'] == song_cf_id].empty else "Unknown"
                
                artist_cf = music[music['extracted_song_id'] == song_cf_id]['artist'].values[0] \
                if not music[music['extracted_song_id'] == song_cf_id].empty else "Unknown"        
                album_cover_cf = get_song_album_cover_url(song_cf_name, artist_cf)
                hybrid_recommendations.append((song_cf_name, album_cover_cf))
                seen_songs.add(song_cf_id)
                print("in cf " + song_cf_name + "\n")

    return hybrid_recommendations[:5] 


def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track: {song_name} artist: {artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        print(album_cover_url)
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(song):
    index = music[music['song'] == song].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_music_names = []
    recommended_music_posters = []
    for i in distances[1:6]:
        artist = music.iloc[i[0]].artist
        recommended_music_posters.append(get_song_album_cover_url(music.iloc[i[0]].song, artist))
        recommended_music_names.append(music.iloc[i[0]].song)

    return recommended_music_names, recommended_music_posters

st.header('Music Recommendation System')
music = pickle.load(open('required_fd', 'rb'))
similarity = pickle.load(open('similar_needed', 'rb'))
music_list = music['song'].values
user_list = music['user_id'].values
selected_song = st.selectbox("Type or select a song from the dropdown", music_list)
selected_user = st.selectbox("Type or select a User from the dropdown", user_list)

with open('final_colaborative_music_recommender_model.pkl', 'rb') as file:
    final_algorithm = pickle.load(file)

if st.button('Show Hybrid Recommendation'):
    user_id = selected_user  
    recommendations = hybrid_recommendation(selected_song, user_id)

    if recommendations:
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, (song_name, poster_url) in enumerate(recommendations):
            with [col1, col2, col3, col4, col5][i]:
                st.text(song_name)
                st.image(poster_url)
    else:
        st.warning("No recommendations available.")
