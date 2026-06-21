import os
from realitydefender import RealityDefender
from realitydefender.errors import RealityDefenderError

API_KEY = os.environ.get("REALITY_DEFENDER_API_KEY")

# The free tier only supports image and audio analysis — video requires
# a paid plan. Since this pipeline already extracts sampled frames as
# JPGs for the other two classifiers, those same images are reused here
# rather than sending whole videos (which the free tier would reject).
SUPPORTED_FREE_TIER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def analyze_frames_with_reality_defender(frame_paths, max_frames=5):
    """
    Sends a subset of the already-extracted sample frames to Reality
    Defender's API for a third, independent opinion alongside the two
    Hugging Face classifiers already in use.

    Capped at max_frames (default 5) by design — the free tier allows
    50 scans/month total, so analyzing every sampled frame on every
    video upload would burn through that quickly. 5 frames per video
    still gives a meaningful cross-check without exhausting the quota
    after only ~10 video analyses in a month.

    Returns aggregated results in the same shape as the other two
    classifiers (median/mean/max across analyzed frames) so the report
    layer can treat all three consistently.
    """

    if not API_KEY:
        return {
            "available": False,
            "error": "REALITY_DEFENDER_API_KEY is not set. This check was skipped.",
            "frame_results": [],
            "positive_probability": None
        }

    rd = RealityDefender(api_key=API_KEY)

    frames_to_check = [
        p for p in frame_paths
        if os.path.exists(p) and os.path.splitext(p)[1].lower() in SUPPORTED_FREE_TIER_EXTENSIONS
    ][:max_frames]

    if not frames_to_check:
        return {
            "available": True,
            "error": "No analyzable frames found.",
            "frame_results": [],
            "positive_probability": None
        }

    frame_results = []
    scores = []

    for path in frames_to_check:
        try:
            upload_result = rd.upload_sync(file_path=path)
            request_id = upload_result["request_id"]

            # Default SDK polling is up to 30 attempts at 2s each (60s max)
            # PER FRAME. Across up to 5 frames sequentially, that's a
            # worst-case 5-minute wait, which would make the whole tool
            # feel broken even when it's just slow. Tightened to a more
            # reasonable per-frame ceiling — if Reality Defender hasn't
            # responded in ~20s, treat that frame as unavailable rather
            # than blocking the whole analysis.
            result = rd.get_result_sync(request_id, max_attempts=10, polling_interval=2000)

            score = result.get("score")
            status = result.get("status")

            frame_results.append({
                "frame": os.path.basename(path),
                "status": status,
                "score": score
            })

            if score is not None:
                scores.append(score)

        except RealityDefenderError as e:
            frame_results.append({
                "frame": os.path.basename(path),
                "error": f"Reality Defender API error: {e.message} ({e.code})"
            })
        except Exception as e:
            frame_results.append({
                "frame": os.path.basename(path),
                "error": str(e)
            })

    rd.cleanup_sync()

    if not scores:
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

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    median_score = (
        sorted_scores[n // 2] if n % 2 == 1
        else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    )
    mean_score = sum(scores) / len(scores)
    max_score = max(scores)

    return {
        "available": True,
        "error": None,
        "frame_results": frame_results,
        "positive_probability": round(median_score, 4),
        "mean_frame_probability": round(mean_score, 4),
        "max_frame_probability": round(max_score, 4),
        "frames_analyzed": len(scores)
    }
