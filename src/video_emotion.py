import subprocess
import tempfile
from pathlib import Path
import os

import cv2

from .image_emotion import predict_image_emotion
from .audio_emotion import predict_audio_emotion
from .utils.io import ensure_file

def _run(cmd: list[str]):
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _extract_audio_wav(video_path: str, out_wav: str):
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        out_wav
    ])

def _sample_frames(video_path: str, every_n_frames: int = 30, max_frames: int = 20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            frames.append(frame)
            if len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    return frames

def predict_video_emotion(video_path: str) -> dict:
    p = ensure_file(video_path)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav_path = td / "audio.wav"

        _extract_audio_wav(str(p), str(wav_path))
        audio_res = predict_audio_emotion(str(wav_path))

        frames = _sample_frames(str(p), every_n_frames=30, max_frames=20)

        frame_results = []
        for i, frame in enumerate(frames):
            frame_path = td / f"frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            r = predict_image_emotion(str(frame_path))
            if r.get("emotion") not in ["no-face-detected"]:
                frame_results.append(r)

        score_map = {}
        for r in frame_results:
            e = r["emotion"]
            score_map[e] = score_map.get(e, 0.0) + float(r.get("score", 0.0))

        best_frame_emotion = max(score_map.items(), key=lambda x: x[1])[0] if score_map else "no-face-detected"

        return {
            "modality": "video",
            "audio": audio_res,
            "frames_used": len(frames),
            "faces_used": len(frame_results),
            "frame_emotion": best_frame_emotion,
            "frame_scores": score_map,
        }
