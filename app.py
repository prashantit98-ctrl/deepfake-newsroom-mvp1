from fastapi import FastAPI, UploadFile, File
import os

from utils.video import extract_frames
from utils.metadata import get_metadata
from utils.transcription import transcribe_audio
from utils.report import generate_report

app = FastAPI()

os.makedirs("uploads", exist_ok=True)


@app.get("/")
async def home():
    return {"status": "working"}


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    metadata = get_metadata(path)
    transcript = transcribe_audio(path)
    frames = extract_frames(path)

    return generate_report(
        metadata,
        transcript,
        frames
    )
