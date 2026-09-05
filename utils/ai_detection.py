import os
import cv2
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# --- Model config ---
# Labels confirmed directly from each model's config.json on Hugging Face.
# Do NOT change these model IDs without re-checking their id2label mapping —
# mismatched label strings are what caused every score to read as 100 before.

FACE_DEEPFAKE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-Model"
FACE_DEEPFAKE_POSITIVE_LABELS = {"fake"}  # this model's labels are "Real" / "Fake"

# Ateeqq/ai-vs-human-image-detector is NOT hosted on HF Inference Providers
# (confirmed via an open community request asking for it to be supported),
# so InferenceClient calls to it will always fail regardless of provider
# setting. Swapped to Organika/sdxl-detector, which IS hosted and actively
# used. Its labels are "artificial" / "human".
AI_GENERATED_MODEL_ID = "Organika/sdxl-detector"
AI_GENERATED_POSITIVE_LABELS = {"artificial"}

HF_TOKEN = os.environ.get("HF_API_TOKEN")

try:
    _FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception:
    _FACE_CASCADE = None


def _get_client():
    return InferenceClient(
        provider="auto",
        api_key=HF_TOKEN
    )


def _contains_face(image_path):
    if _FACE_CASCADE is None:
        return True
    image = cv2.imread(image_path)
    if image is None:
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    min_dim = int(min(height, width) * 0.12)
    min_dim = max(min_dim, 40)
    faces = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(min_dim, min_dim)
    )
    return len(faces) > 0


def _query_model(image_path, model_id):
    client = _get_client()
    result = client.image_classification(image_path, model=model_id)
    return [{"label": r.label, "score": r.score} for r in result]


def _run_classifier(frame_paths, model_id, positive_labels, require_face=False):
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

        # Raw labels/scores kept in the response so mismatched label
        # strings are visible immediately in the report instead of
        # silently producing garbage probabilities.
        positive_entry = next(
            (r for r in result if r["label"].lower() in positive_labels), None
        )
        positive_score = positive_entry["score"] if positive_entry else None

        frame_results.append({
            "frame": os.path.basename(path),
            "raw_labels": result,
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
            "No frames could be analyzed. Check that positive_labels matches "
            "this model's actual id2label output (see raw_labels in "
            "frame_results above for what the model actually returned)."
        )
        return {
            "available": True,
            "error": first_error,
            "frame_results": frame_results,
            "positive_probability": None
        }

    sorted_scores = sorted(positive_scores)
    n = len(sorted_scores)
    if n % 2 == 1:
        median_score = sorted_scores[n // 2]
    else:
        median_score = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2

    avg_score = sum(positive_scores) / len(positive_scores)
    max_score = max(positive_scores)

    return {
        "available": True,
        "error": None,
        "frame_results": frame_results,
        "frames_analyzed": len(positive_scores),
        "positive_probability": median_score,
        "mean_frame_probability": avg_score,
        "max_frame_probability": max_score
    }


def analyze_frames_for_deepfake(frame_paths):
    """
    Runs the face-specific deepfake classifier on a list of frame image
    paths. Requires a detected face per frame (skips frames without one).
    Returns a dict matching the shape generate_report() expects for
    ai_result.
    """
    return _run_classifier(
        frame_paths,
        FACE_DEEPFAKE_MODEL_ID,
        FACE_DEEPFAKE_POSITIVE_LABELS,
        require_face=True
    )


def analyze_frames_for_ai_generation(frame_paths):
    """
    Runs the general AI-vs-real image classifier on a list of frame image
    paths. Not face-specific, so no face requirement. Returns a dict
    matching the shape generate_report() expects for ai_generation_result.
    """
    return _run_classifier(
        frame_paths,
        AI_GENERATED_MODEL_ID,
        AI_GENERATED_POSITIVE_LABELS,
        require_face=False
    )
