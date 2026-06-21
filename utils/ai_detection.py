import os
import cv2
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

FACE_DEEPFAKE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
AI_GENERATED_MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
HF_TOKEN = os.environ.get("HF_API_TOKEN")

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _get_client():
    return InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN
    )


def _contains_face(image_path):
    """
    Checks whether at least one human face is detectable in the frame.

    The face-deepfake model was trained specifically on human face
    images — feeding it frames with no face (animals, landscapes,
    objects) produces meaningless scores, since the input is outside
    what it was ever trained to classify. Gating on face presence
    avoids reporting a confident-looking number for content the model
    was never built to judge.
    """
    image = cv2.imread(image_path)
    if image is None:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    return len(faces) > 0


def _query_model(image_path, model_id):
    """
    Sends one image to the given Hugging Face model via the current
    huggingface_hub InferenceClient.

    The client wants binary bytes, a local file path, or a URL —
    not a PIL Image object — so we pass the path string directly
    and let the client handle reading and content-type detection.

    Returns a list like:
    [{"label": "Deepfake", "score": 0.91}, {"label": "Realism", "score": 0.09}]
    """
    client = _get_client()
    result = client.image_classification(image_path, model=model_id)

    return [{"label": r.label, "score": r.score} for r in result]


def _run_classifier(frame_paths, model_id, positive_labels, require_face=False):
    """
    Shared logic for running any single-model image classifier across
    a list of frames and aggregating a "positive" probability (e.g.
    "this frame is fake" or "this frame is AI-generated").

    positive_labels: lowercase label strings that count as the
    "positive" class for this model (different models use different
    wording — "Deepfake" vs "ai", etc.)

    require_face: if True, frames with no detected face are skipped
    (used for the face-specific deepfake model). If False, every
    frame is sent regardless of face presence (used for the general
    AI-generation check, which isn't face-specific).
    """

    if not HF_TOKEN:
        return {
            "available": False,
            "error": "HF_API_TOKEN is not set. AI detection skipped.",
            "frame_results": [],
            "positive_probability": None
        }

    frame_results = []
    positive_scores = []
    no_face_count = 0

    for path in frame_paths:
        if not os.path.exists(path):
            continue

        if require_face and not _contains_face(path):
            no_face_count += 1
            frame_results.append({
                "frame": os.path.basename(path),
                "skipped": True,
                "reason": "No face detected — frame skipped from this check"
            })
            continue

        try:
            result = _query_model(path, model_id)
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

        positive_entry = next(
            (r for r in result if r["label"].lower() in positive_labels), None
        )
        positive_score = positive_entry["score"] if positive_entry else None

        frame_results.append({
            "frame": os.path.basename(path),
            "label": result[0]["label"] if result else "Unknown",
            "positive_score": positive_score
        })

        if positive_score is not None:
            positive_scores.append(positive_score)

    total_frames_checked = len(frame_paths)

    if require_face and no_face_count == total_frames_checked and total_frames_checked > 0:
        return {
            "available": True,
            "error": None,
            "no_face_detected": True,
            "frame_results": frame_results,
            "positive_probability": None
        }

    if not positive_scores:
        first_error = next(
            (r["error"] for r in frame_results if "error" in r),
            "No frames could be analyzed."
        )
        return {
            "available": True,
            "error": first_error,
            "frame_results": frame_results,
            "positive_probability": None
        }

    avg_score = sum(positive_scores) / len(positive_scores)
    max_score = max(positive_scores)

    return {
        "available": True,
        "error": None,
        "frame_results": frame_results,
        "positive_probability": round(avg_score, 4),
        "max_frame_probability": round(max_score, 4),
        "frames_analyzed": len(positive_scores),
        "frames_skipped_no_face": no_face_count
    }


def analyze_frames_for_deepfake(frame_paths):
    """
    Face-specific deepfake check. Only runs on frames where a human
    face is detected — this model was trained on faces only, so
    faceless frames are skipped rather than given a meaningless score.
    """
    return _run_classifier(
        frame_paths,
        model_id=FACE_DEEPFAKE_MODEL_ID,
        positive_labels={"deepfake"},
        require_face=True
    )


def analyze_frames_for_ai_generation(frame_paths):
    """
    General AI-generation check. Runs on every sampled frame
    regardless of whether a face is present — this model was trained
    broadly on AI-generated vs. real images (art, photos, objects,
    scenes), not specifically on faces, so it's the right tool for
    content the face-deepfake model has to skip (animals, food,
    landscapes, objects, etc.).

    NOTE: the model card for this classifier notes some users have
    reported overfitting during evaluation — treat its output with
    the same "screening aid, not verdict" skepticism as the face model.
    """
    return _run_classifier(
        frame_paths,
        model_id=AI_GENERATED_MODEL_ID,
        positive_labels={"ai"},
        require_face=False
    )
