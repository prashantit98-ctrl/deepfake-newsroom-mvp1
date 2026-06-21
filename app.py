import os
import uuid

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from utils.metadata import get_metadata
from utils.video import extract_frames
from utils.report import generate_report
from utils.ai_detection import analyze_frames_for_deepfake, analyze_frames_for_ai_generation
from utils.reality_defender_detection import analyze_frames_with_reality_defender
from utils.url_download import download_video_from_url

app = FastAPI()

# Basic abuse protection: caps how often any single IP can hit the
# expensive endpoints. In-memory by default (fine for a single
# Railway instance) — resets on restart, which is an acceptable
# tradeoff for a screening-tool MVP, not a high-security system.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# Caps how large an uploaded file can be before we reject it. This is
# a screening tool, not a media archive — 200MB comfortably covers a
# few minutes of normal phone/WhatsApp video while protecting the
# server (and the Hugging Face free-tier quota) from huge uploads.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB

# How many analyze requests a single IP can make per minute. Generous
# enough for normal use (nobody legitimately analyzes 10 videos a
# minute), tight enough to blunt accidental or deliberate abuse.
ANALYZE_RATE_LIMIT = "10/minute"


class UrlAnalyzeRequest(BaseModel):
    url: str


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


def _run_full_analysis(path, upload_id):
    """
    Shared analysis pipeline used by both the file-upload and URL
    endpoints, so a fix made to one path never silently misses the
    other. Does NOT delete the video file — callers are responsible
    for cleanup, since the two endpoints get the file onto disk in
    different ways.
    """
    metadata = get_metadata(path)
    frame_data = extract_frames(path, upload_id)

    # Run both AI checks on the frames already saved as thumbnails
    # (no need to re-read the video — these JPGs are already on disk).
    # - Face deepfake check: only meaningful on frames with a face
    # - AI-generation check: runs on every frame, no face needed,
    #   so it covers content the face check has to skip (animals,
    #   food, objects, landscapes, etc.)
    sample_paths = [
        f"outputs/frames/{name}" for name in frame_data.get("samples", [])
    ]
    ai_result = analyze_frames_for_deepfake(sample_paths)
    ai_generation_result = analyze_frames_for_ai_generation(sample_paths)
    reality_defender_result = analyze_frames_with_reality_defender(sample_paths)

    report = generate_report(
        metadata,
        "Transcription disabled",
        frame_data,
        video_path=path,
        ai_result=ai_result,
        ai_generation_result=ai_generation_result,
        reality_defender_result=reality_defender_result
    )

    samples = frame_data.get("samples", [])
    frame_data["sample_urls"] = [f"/frames/{name}" for name in samples]
    report["frames_analyzed"] = frame_data

    return report


@app.post("/analyze")
@limiter.limit(ANALYZE_RATE_LIMIT)
async def analyze(
    request: Request,
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

    # Stream the upload to disk in chunks, checking size as we go,
    # so an oversized file gets rejected mid-stream rather than fully
    # written to disk first and rejected only afterward.
    total_bytes = 0
    chunk_size = 1024 * 1024  # 1MB

    try:
        with open(path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)}MB."
                    )
                f.write(chunk)
    except HTTPException:
        if os.path.exists(path):
            os.remove(path)
        raise

    try:
        # Run the slow, blocking analysis (frame extraction, Hugging
        # Face calls, Reality Defender polling — all synchronous) in a
        # background thread instead of directly on the event loop.
        # Without this, a single slow analysis blocks EVERY other
        # request this server can handle — including someone just
        # trying to load the homepage — since there's only one worker
        # process. This keeps the server responsive to other requests
        # while a long analysis runs.
        report = await run_in_threadpool(_run_full_analysis, path, upload_id)
    finally:
        # The original video isn't needed after frames are extracted —
        # remove it so uploads/ doesn't grow forever.
        if os.path.exists(path):
            os.remove(path)

    return {
        "filename": file.filename,
        "report": report,
        "status": "analysis complete"
    }


@app.post("/analyze-url")
@limiter.limit(ANALYZE_RATE_LIMIT)
async def analyze_url(
    request: Request,
    payload: UrlAnalyzeRequest
):
    # Downloads the video via yt-dlp, then runs it through the exact
    # same pipeline as a direct file upload — this is what keeps
    # behavior consistent between "upload a file" and "paste a link".
    try:
        path, display_name, upload_id = download_video_from_url(payload.url)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Downloaded videos get the same size cap as direct uploads —
    # yt-dlp's format_sort already favors 720p, but a long video at
    # that resolution can still be large, so this is a real backstop.
    if os.path.exists(path) and os.path.getsize(path) > MAX_UPLOAD_BYTES:
        os.remove(path)
        raise HTTPException(
            status_code=413,
            detail=f"Downloaded video is too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)}MB."
        )

    try:
        report = await run_in_threadpool(_run_full_analysis, path, upload_id)
    finally:
        if os.path.exists(path):
            os.remove(path)

    return {
        "filename": display_name,
        "report": report,
        "status": "analysis complete"
    }
