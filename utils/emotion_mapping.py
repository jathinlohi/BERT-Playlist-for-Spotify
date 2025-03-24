def map_emotions_to_moods(predicted_emotions):
    from utils.emotions_mood import emotion_to_mood
    moods = []
    for emotion, _ in predicted_emotions:
        mood = emotion_to_mood.get(emotion, "indie")
        moods.append(mood)
    return list(set(moods))