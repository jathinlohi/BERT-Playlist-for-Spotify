import torch
import numpy as np
from utils.label_map import emotion_labels

def predict_emotions(text, model, tokenizer, top_k=3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().numpy()

    top_indices = probs.argsort()[-top_k:][::-1]
    top_emotions = [emotion_labels[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]

    return list(zip(top_emotions, top_probs))
