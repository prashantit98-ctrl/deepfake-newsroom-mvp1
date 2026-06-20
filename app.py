import os
import uuid

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from utils.metadata import get_metadata
from utils.video import extract_frames
from utils.report import generate_report
from utils.ai_detection import analyze_frames_for_deepfake

app = FastAPI()

templates = Jinja2Templates(
    directory="templates"
)

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs/frames", exist_ok=True)

# Serve extracted frame thumbnails so the browser can load them as images.
# This makes everything under outputs/frames reachable at /frames/<filename>
app.mount(
    "/frames",
    StaticFiles(directory="outputs/frames"),
    name="frames"
)

ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    "video/x-matroska",
}


@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(
    request: Request
):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):
    # Reject obviously-wrong file types before doing any work.
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a video."
        )

    # One ID shared by this upload's video file and its frame folder,
    # so frames from different uploads never mix or collide.
    upload_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1]
    path = f"uploads/{upload_id}{ext}"

    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        metadata = get_metadata(path)
        frame_data = extract_frames(path, upload_id)

        # Run the AI classifier on the frames already saved as thumbnails
        # (no need to re-read the video — these JPGs are already on disk).
        sample_paths = [
            f"outputs/frames/{name}" for name in frame_data.get("samples", [])
        ]
        ai_result = analyze_frames_for_deepfake(sample_paths)

        report = generate_report(
            metadata,
            "Transcription disabled",
            frame_data,
            video_path=path,
            ai_result=ai_result
        )
    finally:
        # The original video isn't needed after frames are extracted —
        # remove it so uploads/ doesn't grow forever.
        if os.path.exists(path):
            os.remove(path)

    # samples already contain "<upload_id>/frame_30.jpg" style paths,
    # so this just needs the /frames mount prefix to become a real URL.
    samples = frame_data.get("samples", [])
    frame_data["sample_urls"] = [f"/frames/{name}" for name in samples]
    report["frames_analyzed"] = frame_data

    return {
        "filename": file.filename,
        "report": report,
        "status": "analysis complete"
    }
