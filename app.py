from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import uvicorn

from utils.video import extract_frames
from utils.metadata import get_metadata
from utils.report import generate_report

app = FastAPI()

templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    metadata = get_metadata(path)

    # Whisper temporarily disabled
    transcript = "Transcription disabled"

    frames = extract_frames(path)

    return generate_report(
        metadata,
        transcript,
        frames
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
