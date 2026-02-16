
# Common “basic” emotions set:
CANONICAL = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Map common model labels
LABEL_MAP = {
    # Text model
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",

    # SpeechBrain
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    "neu": "neutral",
    "exc": "happy",   
    "fru": "angry",  
}

def to_canonical(label: str) -> str:
    if not label:
        return "unknown"
    key = label.strip().lower()
    return LABEL_MAP.get(key, key)
