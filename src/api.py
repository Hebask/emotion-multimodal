from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

import tempfile
from pathlib import Path

from .text_emotion import predict_text_emotion
from .image_emotion import predict_image_emotion
from .audio_emotion import predict_audio_emotion
from .video_emotion import predict_video_emotion
from .fusion import fuse_predictions

app = FastAPI(title="Emotion Multimodal API")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "detail": exc.errors()},
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/text")
def predict_text(text: str = Form(...)):
    return predict_text_emotion(text)

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / file.filename
        path.write_bytes(await file.read())
        return predict_image_emotion(str(path))

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / file.filename
        path.write_bytes(await file.read())
        return predict_audio_emotion(str(path))

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / file.filename
        path.write_bytes(await file.read())
        return predict_video_emotion(str(path))

@app.post("/predict/multimodal")
async def predict_multimodal(
    text: str | None = Form(None),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
):
    preds = []
    details = {}

    if text:
        r = predict_text_emotion(text)
        details["text"] = r
        preds.append({"modality": "text", "emotion": r["emotion"], "score": r["score"]})

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        if image:
            ip = td / image.filename
            ip.write_bytes(await image.read())
            r = predict_image_emotion(str(ip))
            details["image"] = r
            preds.append({"modality": "image", "emotion": r["emotion"], "score": r["score"]})

        if audio:
            ap = td / audio.filename
            ap.write_bytes(await audio.read())
            r = predict_audio_emotion(str(ap))
            details["audio"] = r
            preds.append({"modality": "audio", "emotion": r["emotion"], "score": r["score"]})

        if video:
            vp = td / video.filename
            vp.write_bytes(await video.read())
            vr = predict_video_emotion(str(vp))
            details["video"] = vr
            video_emotion = vr["audio"]["emotion"]
            video_score = vr["audio"]["score"]
            preds.append({"modality": "video", "emotion": video_emotion, "score": video_score})

    if not preds:
        return JSONResponse(status_code=400, content={"error": "Provide at least one of text/image/audio/video"})

    fused = fuse_predictions(preds)
    return {"fused": fused, "predictions": preds, "details": details}
