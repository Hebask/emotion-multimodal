import re

_CANONICAL = {
    "angry": "anger",
    "anger": "anger",
    "mad": "anger",

    "happy": "joy",
    "happiness": "joy",
    "joy": "joy",

    "sad": "sadness",
    "sadness": "sadness",

    "fear": "fear",
    "fearful": "fear",

    "disgust": "disgust",
    "disgusted": "disgust",

    "surprise": "surprise",
    "surprised": "surprise",

    "neutral": "neutral",
    "calm": "neutral",
}

def to_canonical(label: str) -> str:
    if label is None:
        return "unknown"
    s = str(label).strip().lower()
    s = re.sub(r"[^a-z0-9_\- ]+", "", s)
    s = s.replace("_", " ").replace("-", " ").strip()

    if s.startswith("label"):
        return "unknown"

    if s in _CANONICAL:
        return _CANONICAL[s]

    token = s.split()[0] if s else ""
    return _CANONICAL.get(token, s if s else "unknown")
