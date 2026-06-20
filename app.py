from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from utils.metadata import get_metadata
from utils.video import extract_frames
from utils.report import generate_report

app = FastAPI()

templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    metadata = get_metadata(path)

    frame_data = extract_frames(path)

    transcript = "Transcription disabled"

    report = generate_report(
        metadata,
        transcript,
        frame_data
    )

    return {
        "filename": file.filename,
        "report": report,
        "status": "analysis complete"
    }
