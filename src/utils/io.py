from pathlib import Path

def ensure_file(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    return p

def pretty_result(modality: str, emotion: str, score: float) -> str:
    return f"[{modality.upper()}] {emotion} (score={score:.3f})"
