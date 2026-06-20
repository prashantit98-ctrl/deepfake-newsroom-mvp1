import os
import time
import requests

HF_API_URL = "https://api-inference.huggingface.co/models/prithivMLmods/Deep-Fake-Detector-v2-Model"
HF_TOKEN = os.environ.get("HF_API_TOKEN")


def _query_model(image_bytes, retries=3):
    """
    Sends one image to the Hugging Face Inference API and returns the
    raw classification result, e.g.:
    [{"label": "Deepfake", "score": 0.91}, {"label": "Realism", "score": 0.09}]

    Free-tier hosted models "cold start" if they haven't been called
    recently — the first request can take 10-20s while it loads, and
    the API responds with a 503 + estimated_time while that happens.
    We retry a few times rather than failing immediately.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    for attempt in range(retries):
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=image_bytes,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code == 503:
            # Model is loading — wait roughly as long as it says, then retry.
            wait_time = response.json().get("estimated_time", 5)
            time.sleep(min(wait_time, 15))
            continue

        # Any other error: stop retrying, surface it.
        response.raise_for_status()

    raise RuntimeError("Hugging Face model did not respond after retries (cold start took too long).")


def analyze_frames_for_deepfake(frame_paths):
    """
    Runs each given frame image (file paths on disk) through the
    deepfake detection model and returns a per-frame result plus an
    aggregated video-level verdict.

    frame_paths: list of local file paths to JPGs already saved by
    extract_frames(), e.g. ["outputs/frames/<id>/frame_0.jpg", ...]
    """

    if not HF_TOKEN:
        return {
            "available": False,
            "error": "HF_API_TOKEN is not set. AI detection skipped.",
            "frame_results": [],
            "fake_probability": None
        }

    frame_results = []
    fake_scores = []

    for path in frame_paths:
        if not os.path.exists(path):
            continue

        with open(path, "rb") as f:
            image_bytes = f.read()

        try:
            result = _query_model(image_bytes)
        except Exception as e:
            frame_results.append({
                "frame": os.path.basename(path),
                "error": str(e)
            })
            continue

        # result looks like [{"label": "Deepfake", "score": 0.91}, {"label": "Realism", "score": 0.09}]
        fake_entry = next((r for r in result if r["label"].lower() == "deepfake"), None)
        fake_score = fake_entry["score"] if fake_entry else None

        frame_results.append({
            "frame": os.path.basename(path),
            "label": result[0]["label"] if result else "Unknown",
            "fake_score": fake_score
        })

        if fake_score is not None:
            fake_scores.append(fake_score)

    if not fake_scores:
        return {
            "available": True,
            "error": "No frames could be analyzed.",
            "frame_results": frame_results,
            "fake_probability": None
        }

    # Aggregate: average fake-probability across analyzed frames.
    # A single suspicious frame can be noise; a consistently high
    # average across many frames is a stronger signal.
    avg_fake_probability = sum(fake_scores) / len(fake_scores)
    max_fake_probability = max(fake_scores)

    return {
        "available": True,
        "error": None,
        "frame_results": frame_results,
        "fake_probability": round(avg_fake_probability, 4),
        "max_frame_fake_probability": round(max_fake_probability, 4),
        "frames_analyzed": len(fake_scores)
    }
