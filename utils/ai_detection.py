import os
import cv2
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# --- Face-deepfake model: run LOCALLY, not via hosted Inference API ---
# Reason: essentially no community-trained deepfake classifier (this one
# included) is deployed on Hugging Face's Inference Providers. Confirmed
# directly on the model's own page: "This model isn't deployed by any
# Inference Provider." Calling it through InferenceClient will always
# 400 with "Model not supported by provider hf-inference", regardless of
# provider="auto" -- there's no provider hosting it to route to.
# Loading it locally with transformers sidesteps that entirely.
FACE_DEEPFAKE_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-Model"
FACE_DEEPFAKE_POSITIVE_LABELS = {"fake"}  # confirmed labels: "Real" / "Fake"

# --- AI-generation model: stays on the hosted Inference API ---
# Organika/sdxl-detector IS confirmed deployed on Inference Providers
# (it's a long-established, frequently-forked model), so this one is
# fine to call remotely.
AI_GENERATED_MODEL_ID = "Organika/sdxl-detector"
AI_GENERATED_POSITIVE_LABELS = {"artificial"}  # confirmed labels: "artificial" / "human"

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


# --- Local (in-process) model loading for the face-deepfake classifier ---
# Loaded once at import time and reused across requests, rather than
# reloading per-frame or per-request. First load downloads the weights
# (~350MB) and will be slow; subsequent calls are fast.
_local_pipeline = None
_local_pipeline_error = None


def _get_local_pipeline():
    global _local_pipeline, _local_pipeline_error
    if _local_pipeline is not None or _local_pipeline_error is not None:
        return _local_pipeline, _local_pipeline_error
    try:
        from transformers import pipeline
        _local_pipeline = pipeline(
            "image-classification",
            model=FACE_DEEPFAKE_MODEL_ID
        )
    except Exception as e:
        _local_pipeline_error = str(e)
    return _local_pipeline, _local_pipeline_error


def _query_local_model(image_path):
    pipe, err = _get_local_pipeline()
    if pipe is None:
        raise RuntimeError(err or "Local model failed to load.")
    result = pipe(image_path)
    return [{"label": r["label"], "score": r["score"]} for r in result]


def _query_hosted_model(image_path, model_id):
    client = _get_client()
    result = client.image_classification(image_path, model=model_id)
    return [{"label": r.label, "score": r.score} for r in result]


def _run_classifier(frame_paths, positive_labels, require_face=False, use_local=False, model_id=None):
    if not use_local and not HF_TOKEN:
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
                "reason": "No face detected -- frame skipped from this check"
            })
            continue

        try:
            if use_local:
                result = _query_local_model(path)
            else:
                result = _query_hosted_model(path, model_id)
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
    Runs the face-specific deepfake classifier LOCALLY (in-process, not via
    the Hugging Face hosted Inference API) on a list of frame image paths.
    Requires a detected face per frame (skips frames without one).
    Returns a dict matching the shape generate_report() expects for
    ai_result.
    """
    return _run_classifier(
        frame_paths,
        FACE_DEEPFAKE_POSITIVE_LABELS,
        require_face=True,
        use_local=True
    )


def analyze_frames_for_ai_generation(frame_paths):
    """
    Runs the general AI-vs-real image classifier via the hosted Hugging
    Face Inference API on a list of frame image paths. Not face-specific,
    so no face requirement. Returns a dict matching the shape
    generate_report() expects for ai_generation_result.
    """
    return _run_classifier(
        frame_paths,
        AI_GENERATED_POSITIVE_LABELS,
        require_face=False,
        use_local=False,
        model_id=AI_GENERATED_MODEL_ID
    )
