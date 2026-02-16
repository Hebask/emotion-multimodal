import os
from pathlib import Path

def ensure_file(path: str) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return p

def pretty_result(modality: str, label: str, score: float | None = None) -> str:
    if score is None:
        return f"[{modality}] emotion={label}"
    return f"[{modality}] emotion={label}  score={score:.4f}"
