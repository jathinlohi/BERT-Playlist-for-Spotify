import torch
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from utils.label_map import emotion_labels
from utils.inference import predict_emotions
from utils.emotion_mapping import map_emotions_to_moods
from utils.spotify_utils import generate_playlist

# Constants
MODEL_PATH = "emotion_model/emotion_classifier.pt"
TOKENIZER_PATH = "bert-base-uncased"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
config = BertConfig.from_pretrained(TOKENIZER_PATH, num_labels=len(emotion_labels))
model = BertForSequenceClassification.from_pretrained(TOKENIZER_PATH, config=config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Take user input
user_input = input("Describe how you're feeling: ")

# Get top 3 predicted emotions
top_emotions = predict_emotions(user_input, model, tokenizer, top_k=3)
print(f"\nTop Emotions Detected:")
for emotion, prob in top_emotions:
    print(f"- {emotion}: {prob:.2f}")

# Map emotions to moods
moods = map_emotions_to_moods(top_emotions)
print(f"\nMapped Moods: {moods}")

# Generate playlist
playlist = generate_playlist(moods, total_songs=15)

# Display playlist
print("\nYour Curated Playlist:")
for song in playlist:
    print(f"{song['track']} by {song['artist']} 🎧 → {song['url']}")
