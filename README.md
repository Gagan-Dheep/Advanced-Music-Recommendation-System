# Hybrid Music Recommendation System

## Overview
The Hybrid Music Recommendation System delivers personalized song suggestions by integrating collaborative filtering (SVD) and content-based filtering (cosine similarity). Enriched with metadata from the Spotify API, the interactive Streamlit web app captures implicit user preferences via song selection. Performance is enhanced with caching, and the application is deployed on Streamlit Cloud for real-time accessibility.

## Features
- **Hybrid Recommendation Engine:** Combines collaborative filtering (SVD) and content-based filtering (cosine similarity) for balanced and personalized recommendations.
- **Implicit User Feedback:** Users select songs from a curated grid, and their choices are used as implicit signals of their taste.
- **Spotify API Integration:** Retrieves enriched metadata (e.g., album covers) to enhance recommendation quality.
- **Interactive Web App:** Built with Streamlit, enabling a smooth, real-time user experience.
- **Performance Optimization:** Implements caching to reduce load times and improve responsiveness.
- **Shuffle Functionality:** Allows users to refresh the sample song grid with a single click.

## Architecture
The system employs a hybrid approach:
- **Content-Based Filtering (CBF):** Uses cosine similarity on song features to find similar tracks.
- **Collaborative Filtering (CF):** Leverages Singular Value Decomposition (SVD) on user-song interaction data to predict preferences.
- **Hybrid Model:** Merges the outputs of CBF and CF to produce a final recommendation list.
- **Caching:** Optimizes performance by caching data loading and API calls.

## Installation

### Prerequisites
- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- [Spotipy](https://spotipy.readthedocs.io/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [scikit-learn](https://scikit-learn.org/)
- Other standard Python libraries: pickle, os, random, numpy

### Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/hybrid-music-recommendation.git
   cd hybrid-music-recommendation
Install Dependencies:

bash
Copy
Edit
pip install -r requirements.txt
(Ensure your requirements.txt lists all required packages.)

Configure Environment Variables: Create a .env file in the project root and add your Spotify API credentials:

ini
Copy
Edit
CLIENT_ID=your_spotify_client_id
CLIENT_SECRET=your_spotify_client_secret
Usage
To run the web app locally:

bash
Copy
Edit
streamlit run app.py
Select Songs: Choose your favorite songs from the grid.
Shuffle Songs: Click "Shuffle Songs" to refresh the grid with a new set of sample songs.
Get Recommendations: Press "Show Hybrid Recommendation" to view personalized song suggestions.
Deployment
The application is deployed on Streamlit Cloud, providing real-time access to users. Explore the live application via the link provided in the repository description.

File Structure
bash
Copy
Edit
├── app.py                                  # Main Streamlit application script
├── required_fd                             # Pickled music dataset file
├── similar_needed                          # Pickled similarity matrix file
├── final_colaborative_music_recommender_model.pkl  # Pre-trained collaborative filtering model
├── .env                                    # Environment variables file (not committed)
├── requirements.txt                        # List of project dependencies
└── README.md                               # This file
Contributing
Contributions, bug fixes, and enhancements are welcome! Please open an issue or submit a pull request with your suggestions.

License
This project is licensed under the MIT License.
