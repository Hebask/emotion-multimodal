from transformers import pipeline
import librosa

from .config import settings
from .utils.io import ensure_file
from .utils.labels import to_canonical

_audio_pipe = None

def _get_pipe():
    global _audio_pipe
    if _audio_pipe is None:
        _audio_pipe = pipeline(
            task="audio-classification",
            model=settings.AUDIO_MODEL,
            top_k=5,
        )
    return _audio_pipe

def predict_audio_emotion(audio_path: str) -> dict:
    p = ensure_file(audio_path)

    y, sr = librosa.load(str(p), sr=16000, mono=True)

    pipe = _get_pipe()
    outputs = pipe({"array": y, "sampling_rate": 16000})

    best = max(outputs, key=lambda x: float(x.get("score", 0.0)))
    label = to_canonical(str(best.get("label", "unknown")))
    score = float(best.get("score", 0.0))

    top3 = sorted(outputs, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:3]
    top3 = [{"label": to_canonical(str(x.get("label"))), "score": float(x.get("score", 0.0))} for x in top3]

    return {
        "modality": "audio",
        "emotion": label,
        "score": score,
        "sample_rate": 16000,
        "top3": top3,
    }
