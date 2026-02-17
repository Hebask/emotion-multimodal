import cv2
import numpy as np
from PIL import Image

from .utils.io import ensure_file
from .utils.labels import to_canonical

try:
    from fer import FER
except ImportError:
    from fer.fer import FER

_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        _detector = FER(mtcnn=False)
    return _detector

def _read_image_any(path: str):
    img_bgr = cv2.imread(path)
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = Image.open(path).convert("RGB")
    return np.array(img)

def predict_image_emotion(image_path: str) -> dict:
    p = ensure_file(image_path)
    img_rgb = _read_image_any(str(p))

    detector = _get_detector()
    results = detector.detect_emotions(img_rgb)

    if not results:
        return {
            "modality": "image",
            "emotion": "no-face-detected",
            "score": 0.0,
            "faces": 0,
        }

    best_face = None
    best_score = -1.0
    best_label = None

    for r in results:
        emotions = r.get("emotions", {})
        if not emotions:
            continue
        label, score = max(emotions.items(), key=lambda x: x[1])
        if float(score) > best_score:
            best_score = float(score)
            best_label = label
            best_face = r

    return {
        "modality": "image",
        "emotion": to_canonical(best_label),
        "score": float(best_score),
        "faces": len(results),
        "bbox": best_face.get("box") if best_face else None,
    }
