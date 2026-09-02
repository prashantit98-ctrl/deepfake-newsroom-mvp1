import os
import cv2
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

FACE_DEEPFAKE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
AI_GENERATED_MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
HF_TOKEN = os.environ.get("HF_API_TOKEN")

try:
    _FACE_CASCADE = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
except Exception:
    _FACE_CASCADE = None


def _get_client():
    return InferenceClient(
        provider="hf-inference",
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
        "positive_probability": round(median_score, 4),
        "mean_frame_probability": round(avg_score, 4),
        "max_frame_probability": round(max_score, 4),
        "frames_analyzed": len(positive_scores),
        "frames_skipped_no_face": no_face_count
    }


def analyze_frames_for_deepfake(frame_paths):
    return _run_classifier(
        frame_paths,
        model_id=FACE_DEEPFAKE_MODEL_ID,
        positive_labels={"deepfake"},
        require_face=True
    )


def analyze_frames_for_ai_generation(frame_paths):
    return _run_classifier(
        frame_paths,
        model_id=AI_GENERATED_MODEL_ID,
        positive_labels={"ai"},
        require_face=False
    )
