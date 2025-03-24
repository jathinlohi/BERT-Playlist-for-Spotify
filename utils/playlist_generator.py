import json
import random
import os

# Load the song database
def load_song_database(filepath="data/songs.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Generate playlist from mood list
def generate_playlist_from_moods(moods, num_songs=15):
    song_database = load_song_database()
    playlist = []

    for mood in moods:
        songs = song_database.get(mood, [])
        selected = random.sample(songs, min(5, len(songs)))  # max 5 per mood
        playlist.extend(selected)

    random.shuffle(playlist)
    return playlist[:num_songs]
