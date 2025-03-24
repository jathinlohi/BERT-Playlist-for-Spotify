import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import random

# Load credentials from env or fallback
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

auth_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

def search_tracks_by_mood(mood, limit=10):
    # Only search by mood now — removed genre keywords
    results = sp.search(q=mood, type='track', limit=50)
    tracks = []

    for item in results['tracks']['items']:
        track_name = item['name']
        artist_name = item['artists'][0]['name']
        url = item['external_urls']['spotify']
        track_id = item['id']
        popularity = item['popularity']

        tracks.append({
            'track': track_name,
            'artist': artist_name,
            'url': url,
            'id': track_id,
            'popularity': popularity
        })

    # Shuffle for diversity, sort by popularity, and pick top N
    random.shuffle(tracks)
    tracks = sorted(tracks, key=lambda x: x['popularity'], reverse=True)
    return tracks[:limit]

def generate_playlist(moods, total_songs=15):
    seen_ids = set()
    playlist = []

    per_mood_quota = total_songs // len(moods) + 2

    for mood in moods:
        tracks = search_tracks_by_mood(mood, limit=per_mood_quota)
        for track in tracks:
            if track['id'] not in seen_ids:
                playlist.append(track)
                seen_ids.add(track['id'])
                if len(playlist) >= total_songs:
                    break
        if len(playlist) >= total_songs:
            break

    random.shuffle(playlist)
    return playlist[:total_songs]
