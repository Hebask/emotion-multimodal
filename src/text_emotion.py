from transformers import pipeline
from .config import settings
from .utils.labels import to_canonical

_text_pipe = None

def _get_pipe():
    global _text_pipe
    if _text_pipe is None:
        _text_pipe = pipeline(
            task="text-classification",
            model=settings.TEXT_MODEL,
            top_k=None,             
            return_all_scores=True,  
        )
    return _text_pipe

def _normalize_outputs(raw):
    if isinstance(raw, list):
        if len(raw) == 0:
            return []
        if isinstance(raw[0], list):
            return raw[0]
        if isinstance(raw[0], dict):
            return raw
    if isinstance(raw, dict):
        return [raw]
    return []

def predict_text_emotion(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("Text is empty")

    pipe = _get_pipe()
    raw = pipe(text)
    outputs = _normalize_outputs(raw)

    if not outputs:
        raise RuntimeError(f"Unexpected model output format: {type(raw)} -> {raw}")

    best = max(outputs, key=lambda x: float(x.get("score", 0.0)))
    label = to_canonical(str(best.get("label", "unknown")))
    score = float(best.get("score", 0.0))

    top3 = sorted(outputs, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:3]
    top3 = [{"label": to_canonical(str(x.get("label"))), "score": float(x.get("score", 0.0))} for x in top3]

    return {"modality": "text", "emotion": label, "score": score, "top3": top3}
