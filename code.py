from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

auth_manager = SpotifyClientCredentials(client_id="c389e6ff9ac847578d6ef13170394d33", client_secret="d96fc96fb1b3462b85d06aa63bac2c1c")
sp = Spotify(auth_manager=auth_manager)
genius = lyricsgenius.Genius("uX8DuZ_OeCJqTtz1f_zuW_Q9UNNORmcRxiq9nCBe-A64dK92Da6jmbLqM-aJcXs5", timeout=15, retries=3)
genius.verbose = False
genius.remove_section_headers = True

def fetch_lyrics(track_name, artist_name):
    try:
        song = genius.search_song(track_name, artist_name)
        if song and song.lyrics:
            lyrics = re.sub(r'\[.*?\]', '', song.lyrics)
            return lyrics.strip()
    except Exception as e:
        print(f"Error fetching lyrics for {track_name}: {e}")
    return "Lyrics not found."

def get_related_info(input_track_name, input_artist_name):
    track_results = sp.search(q=f"track:{input_track_name} artist:{input_artist_name}", type='track', limit=1)
    related_artists_info = []

    with open('track_lyrics.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Track Name', 'Artist Names', 'Lyrics'])

        if track_results['tracks']['items']:
            track = track_results['tracks']['items'][0]
            track_id = track['id']
            track_name = track['name']
            artist_ids = [artist['id'] for artist in track['artists']]
            artist_name = ', '.join(artist['name'] for artist in track['artists'])
            lyrics = fetch_lyrics(track_name, artist_name)
            writer.writerow([track_name, artist_name, lyrics])

            related_tracks = sp.recommendations(seed_tracks=[track_id], limit=50)['tracks']
            for related_track in related_tracks:
                related_track_name = related_track['name']
                related_artist_name = ', '.join(artist['name'] for artist in related_track['artists'])
                related_lyrics = fetch_lyrics(related_track_name, related_artist_name)
                writer.writerow([related_track_name, related_artist_name, related_lyrics])

            for artist_id in artist_ids:
                related_artists = sp.artist_related_artists(artist_id)['artists']
                top_related_artists = related_artists[:10]  # Limit to top 10 related artists
                for artist in top_related_artists:
                    related_artists_info.append((artist['name'], artist['popularity']))

    print_related_artists(related_artists_info)

def print_related_artists(related_artists_info):
    print("\nRelated Artists:")
    for artist, popularity in related_artists_info:
        print(f"{artist} - Popularity: {popularity}")

def print_top_similar_songs():
    df = pd.read_csv('track_lyrics.csv')
    df.dropna(subset=['Lyrics'], inplace=True)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Lyrics'])
    cosine_sim = cosine_similarity(X)

    input_song_index = 0
    similarities = cosine_sim[input_song_index]
    sorted_indices = similarities.argsort()[::-1][1:]

    print(f"\nTop 10 songs similar to '{df.iloc[input_song_index]['Track Name']}' by {df.iloc[input_song_index]['Artist Names']} based on lyrics:")
    for i in range(10):
        idx = sorted_indices[i]
        print(f"{i+1}. {df.iloc[idx]['Track Name']} by {df.iloc[idx]['Artist Names']} - Similarity: {similarities[idx]:.4f}")

def main():
    input_track_name = input("Enter the name of a track: ")
    input_artist_name = input("Enter the name of the artist: ")
    get_related_info(input_track_name, input_artist_name)
    print_top_similar_songs()

if __name__ == "__main__":
    main()
