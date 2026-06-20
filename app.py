from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from utils.metadata import get_metadata
from utils.video import extract_frames

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

    try:
        frame_count = extract_frames(path)
    except Exception as e:
        frame_count = f"Error: {str(e)}"

    return {
        "filename": file.filename,
        "metadata_found": len(metadata) > 0,
        "frames_processed": frame_count,
        "status": "analysis complete"
    }
