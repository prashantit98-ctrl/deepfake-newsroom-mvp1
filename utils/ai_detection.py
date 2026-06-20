import os
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
HF_TOKEN = os.environ.get("HF_API_TOKEN")


def _get_client():
    return InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN
    )


def _query_model(image_bytes):
    """
    Sends one image to the Hugging Face model via the current
    huggingface_hub InferenceClient (the old raw api-inference.huggingface.co
    REST endpoint was deprecated in favor of this client/router setup).

    The client already retries automatically while a free-tier model is
    "cold starting" (loading after being idle), so we don't need to
    hand-roll that retry loop anymore.

    Returns a list like:
    [{"label": "Deepfake", "score": 0.91}, {"label": "Realism", "score": 0.09}]
    """
    client = _get_client()
    result = client.image_classification(image_bytes, model=MODEL_ID)

    # result is a list of ImageClassificationOutputElement objects,
    # each with .label and .score — normalize to plain dicts.
    return [{"label": r.label, "score": r.score} for r in result]


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
        except HfHubHTTPError as e:
            frame_results.append({
                "frame": os.path.basename(path),
                "error": f"Hugging Face API error: {e}"
            })
            continue
        except Exception as e:
            frame_results.append({
                "frame": os.path.basename(path),
                "error": str(e)
            })
            continue

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
        # Surface the first real error we hit, instead of a generic message,
        # so problems like this are easy to diagnose from the report alone.
        first_error = next(
            (r["error"] for r in frame_results if "error" in r),
            "No frames could be analyzed."
        )
        return {
            "available": True,
            "error": first_error,
            "frame_results": frame_results,
            "fake_probability": None
        }

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
