import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
import random

# ------------------------------------------------------------------------------
# Custom CSS for Spinner and Song Titles
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Style for song title: fixed height, 2 lines clamp, and centered text.
       The title attribute will show full text on hover. */
    .song-title {
        height: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 3;  /* limit to 3 lines */
        -webkit-box-orient: vertical;
        text-align: center;
        word-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Setup: Environment and API Initialization
# ------------------------------------------------------------------------------
# Load environment variables (to get Spotify API credentials)
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# Initialize the Spotify client using client credentials
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# ------------------------------------------------------------------------------
# Data and Model Loading
# ------------------------------------------------------------------------------
# Load the pre-processed music data and similarity matrix (content-based info)
@st.cache_data(show_spinner=False)
def load_data():
    music = pickle.load(open('fd_dataframe', 'rb'))
    similarity = pickle.load(open('similarity_Vector', 'rb'))
    with open('final_colaborative_music_recommender_model.pkl', 'rb') as file:
        final_algorithm = pickle.load(file)
    return music, similarity, final_algorithm

music, similarity, final_algorithm = load_data()

# ------------------------------------------------------------------------------
# Caching for Expensive API Calls
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_album_cover(song_name, artist_name):
    """
    Cached version of the function to get the album cover URL.
    """
    search_query = f"track: {song_name} artist: {artist_name}"
    results = sp.search(q=search_query, type="track")
    if results and results["tracks"]["items"]:
        return results["tracks"]["items"][0]["album"]["images"][0]["url"]
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def get_song_album_cover_url(song_name, artist_name):
    # Call the cached function
    return get_album_cover(song_name, artist_name)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def recommend(selected_songs):
    """
    Content-Based Filtering (CBF) recommendation function.
    Given a list of selected songs (implicit user feedback), this function:
      - Finds the index of each selected song in the dataset.
      - Retrieves the top 5 most similar songs (excluding the song itself).
      - Aggregates recommendations from each selected song, ensuring no duplicates.
    Returns two lists: recommended song names and their corresponding album cover URLs.
    """
    recommended_songs = []
    recommended_posters = []
    seen = set()
    
    # Loop through each song the user selected
    for song in selected_songs:
        try:
            index = music[music['song'] == song].index[0]
        except IndexError:
            continue  # Skip if the song is not found
        
        # Get similarity scores for the song, sorted descending
        distances = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)
        # Skip the first result (the song itself) and take next top 5 recommendations
        for dist_tuple in distances[1:6]:
            rec_song = music.iloc[dist_tuple[0]].song
            artist = music.iloc[dist_tuple[0]].artist
            if rec_song not in seen:
                seen.add(rec_song)
                recommended_songs.append(rec_song)
                recommended_posters.append(get_song_album_cover_url(rec_song, artist))
    
    print("CBF Recommendations:", recommended_songs)
    return recommended_songs, recommended_posters

def find_similar_user(selected_songs, threshold=1):
    """
    Improved method to find the best-matching existing user:
    - Counts how many selected songs each user has interacted with.
    - Uses listen_count as a weight for ranking users.
    - Picks the user with the highest overlap (instead of random ties).
    """
    # Filter dataset for rows where the song is in selected_songs
    subset = music[music['song'].isin(selected_songs)]

    if subset.empty:
        print("No users found.")
        return None  # No matching users found

    # Group by user_id and sum the listen_count for weight
    user_scores = subset.groupby('user_id')['listen_count'].sum()

    # Sort users by weighted listen count (descending)
    user_scores = user_scores.sort_values(ascending=False)

    # Return the user with the highest score if above threshold
    if not user_scores.empty and user_scores.iloc[0] >= threshold:
        print("Best matching user:", user_scores.idxmax())
        return user_scores.idxmax()

    print("No strong user match found.")
    return None


def get_cf_recommendations(user_id, top_n=5):
    """
    Get Collaborative Filtering (CF) recommendations for an existing user.
    This function iterates over all unique song IDs in the dataset, predicts ratings for each song,
    and returns the top_n recommendations as a list of (song_name, album_cover_url) tuples.
    """
    cf_recs = []
    all_song_ids = music['extracted_song_id'].unique()
    predictions = []
    
    # Predict ratings for every song using the CF model
    for song_id in all_song_ids:
        pred = final_algorithm.predict(user_id, song_id)
        predictions.append((song_id, pred.est))
    
    # Sort the predictions descending by estimated rating and select top_n
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    
    # Retrieve song details for each recommended song_id
    for song_id, _ in predictions:
        subset = music[music['extracted_song_id'] == song_id]
        if not subset.empty:
            song_name = subset['song'].values[0]
            artist = subset['artist'].values[0]
            poster = get_song_album_cover_url(song_name, artist)
            cf_recs.append((song_name, poster))
    
    print("CF Recommendations:", cf_recs)
    
    return cf_recs

def merge_results(cb_recs, cb_posters, cf_recs, max_results=5):
    """
    Merge Content-Based Filtering (CBF) and Collaborative Filtering (CF) results fairly.
    - Interleaves CBF and CF recommendations to ensure diversity.
    - Ensures no duplicates in the final recommendation list.
    """
    merged = []
    seen = set()
    
    # Pair CBF recommendations together (song, poster)
    cb_pairs = list(zip(cb_recs, cb_posters))
    cf_pairs = cf_recs  # CF recommendations already in (song, poster) format

    # Interleave recommendations
    while (cb_pairs or cf_pairs) and len(merged) < max_results:
        if cb_pairs:
            song, poster = cb_pairs.pop(0)  # Take one from CBF
            if song not in seen:
                merged.append((song, poster))
                seen.add(song)
        if cf_pairs and len(merged) < max_results:
            song, poster = cf_pairs.pop(0)  # Take one from CF
            if song not in seen:
                merged.append((song, poster))
                seen.add(song)

    return merged


def hybrid_recommendation(selected_songs):
    """
    Generate hybrid recommendations using the following strategy:
      1. Use content-based filtering (CBF) on the selected songs.
      2. Attempt to find an existing user whose interaction history overlaps with the selected songs.
         - This existing user represents the new user's taste.
      3. If a similar user is found, fetch CF recommendations for that user.
      4. Merge the two recommendation lists, giving higher priority to CBF results.
    Returns a list of (song_name, album_cover_url) tuples.
    """
    # Step 1: Get recommendations from the content-based engine
    cb_recs, cb_posters = recommend(selected_songs)
    
    # Step 2: Find a similar existing user based on selected songs (using implicit feedback)
    matched_user = find_similar_user(selected_songs, threshold=1)
    
    # Step 3: If a similar user is found, get CF recommendations for that user
    cf_recs = []
    if matched_user:
        cf_recs = get_cf_recommendations(matched_user, top_n=5)
    
    # Step 4: Merge the two recommendation lists, with higher weight for CBF
    hybrid_recs = merge_results(cb_recs, cb_posters, cf_recs, max_results=5)
    return hybrid_recs

# ------------------------------------------------------------------------------
# Streamlit Frontend - Music Recommendation System
# ------------------------------------------------------------------------------
st.header('Music Recommendation System')
st.subheader("Select the songs you like:")

if st.button("Shuffle Songs"):
    st.session_state.sample_songs = random.sample(list(music['song'].values), 10)
    # Clear current selections when shuffling.
    st.session_state.selected_songs = set()
    st.rerun()

# Ensure initial song set remains constant across reruns.
if 'sample_songs' not in st.session_state:
    st.session_state.sample_songs = random.sample(list(music['song'].values), 10)
sample_songs = st.session_state.sample_songs

# Persist selected songs in session_state.
if 'selected_songs' not in st.session_state:
    st.session_state.selected_songs = set()

# Display songs in a grid (2 rows x 5 columns).
cols = st.columns(5)
for idx, song in enumerate(sample_songs):
    col = cols[idx % 5]
    artist = music[music['song'] == song]['artist'].values[0] if song in music['song'].values else "Unknown"
    poster = get_song_album_cover_url(song, artist)
    with col:
        st.image(poster, width=120)
        # Wrap the song title in a div with our custom CSS class for even height.
        st.markdown(f"<div class='song-title' title='{song}'>{song}</div>", unsafe_allow_html=True)
        if st.checkbox("Select", key=song, value=song in st.session_state.selected_songs, on_change=lambda: None):
            st.session_state.selected_songs.add(song)
        else:
            st.session_state.selected_songs.discard(song)

# The "Show Hybrid Recommendation" button.
if st.button('Show Hybrid Recommendation'):
    if st.session_state.selected_songs:
        # Create a container below the button to display recommendations.
        rec_container = st.empty()
        with st.spinner("Generating recommendations..."):
            recommendations = hybrid_recommendation(list(st.session_state.selected_songs))
        with rec_container:
            st.subheader("Recommended Songs:")
            if recommendations:
                rec_cols = st.columns(5)
                for i, (rec_song, rec_poster) in enumerate(recommendations):
                    with rec_cols[i]:
                        st.image(rec_poster, width=120)
                        st.markdown(f"<div class='song-title' title='{rec_song}'>{rec_song}</div>", unsafe_allow_html=True)
            else:
                st.warning("No recommendations available.")
    else:
        st.warning("Please select at least one song from the sample above.")