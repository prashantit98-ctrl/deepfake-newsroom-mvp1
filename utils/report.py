import cv2
import numpy as np


def _read_sample_frames(video_path, max_samples=12):
    """
    Reads up to max_samples frames spread evenly across the video,
    for use in the pixel-level checks below. Separate from the
    frames already saved as thumbnails — this just needs raw arrays.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total <= 0:
        cap.release()
        return []

    step = max(total // max_samples, 1)
    frames = []

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        if len(frames) >= max_samples:
            break

    cap.release()
    return frames


def _check_compression_noise(frames):
    """
    Heavily re-encoded/re-compressed video tends to have unusually
    smooth or unusually blocky noise patterns compared to a single
    clean camera encode. This computes the variance of the Laplacian
    (a common blur/noise proxy) across sampled frames and flags
    extremes in either direction.
    """
    if not frames:
        return None

    variances = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variances.append(lap.var())

    avg_variance = float(np.mean(variances))
    return avg_variance


def _check_frame_consistency(frames):
    """
    Computes the average histogram difference between consecutive
    sampled frames. Real continuous footage tends to change gradually;
    spliced or heavily edited video can show abrupt jumps. This is a
    coarse signal, not proof of editing on its own.
    """
    if len(frames) < 2:
        return None

    diffs = []
    for i in range(len(frames) - 1):
        hist_a = cv2.calcHist([frames[i]], [0], None, [64], [0, 256])
        hist_b = cv2.calcHist([frames[i + 1]], [0], None, [64], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        diff = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
        diffs.append(diff)

    return float(np.mean(diffs))


def _check_edge_density_variance(frames):
    """
    Looks at how much edge density (sharpness/detail) varies across
    sampled frames. Face-swap and blending artifacts can locally
    smooth or sharpen regions in a way that creates more variance
    than a normally encoded video. Coarse, frame-level signal only —
    this does not localize or confirm a specific manipulated region.
    """
    if not frames:
        return None

    densities = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = float(np.count_nonzero(edges)) / edges.size
        densities.append(density)

    return float(np.std(densities))


def generate_report(metadata, transcript, frames, video_path=None):
    """
    Produces a heuristic screening report.

    IMPORTANT: this performs signal-based heuristic screening only.
    It does NOT run any AI/ML deepfake detection model and cannot
    reliably confirm whether a video has been manipulated. It looks
    for secondary signals (metadata, compression, frame consistency,
    edge variance) that *can* correlate with editing or re-encoding,
    but plenty of real videos will trigger these too, and a well-made
    fake may trigger none of them. Treat results as a starting point
    for manual review, not a verdict.
    """

    risk_score = 0
    findings = []

    # --- Metadata check ---
    metadata_present = bool(metadata)
    if metadata_present:
        findings.append("Metadata present")
    else:
        findings.append("Metadata missing or stripped (common for messaging apps — weak signal alone)")
        risk_score += 10

    # --- Frame count / duration check ---
    total_frames = 0
    saved_frames = 0
    if isinstance(frames, dict):
        total_frames = frames.get("total_frames", 0)
        saved_frames = frames.get("saved_frames", 0)

    if total_frames < 100:
        findings.append("Video unusually short for reliable screening")
        risk_score += 5
    elif total_frames > 500:
        findings.append("Video length sufficient for review")

    if saved_frames < 3:
        findings.append("Very few key frames extracted")
        risk_score += 5

    # --- Pixel-level heuristic checks (need the actual video file) ---
    if video_path:
        sample_frames = _read_sample_frames(video_path)

        noise_variance = _check_compression_noise(sample_frames)
        if noise_variance is not None:
            if noise_variance < 50:
                findings.append(
                    f"Unusually smooth/low-detail frames (Laplacian variance {noise_variance:.1f}) "
                    "— can indicate heavy compression or re-encoding"
                )
                risk_score += 15
            elif noise_variance > 4000:
                findings.append(
                    f"Unusually high noise/blockiness (Laplacian variance {noise_variance:.1f}) "
                    "— can indicate aggressive re-compression"
                )
                risk_score += 15
            else:
                findings.append("Compression noise levels within typical range")

        consistency_score = _check_frame_consistency(sample_frames)
        if consistency_score is not None:
            if consistency_score > 0.35:
                findings.append(
                    f"Abrupt visual jumps between frames (score {consistency_score:.2f}) "
                    "— can indicate splicing or scene cuts"
                )
                risk_score += 15
            else:
                findings.append("Frame-to-frame visual consistency normal")

        edge_variance = _check_edge_density_variance(sample_frames)
        if edge_variance is not None:
            if edge_variance > 0.08:
                findings.append(
                    f"High variance in edge sharpness across frames ({edge_variance:.3f}) "
                    "— can indicate localized blending or smoothing artifacts"
                )
                risk_score += 15
            else:
                findings.append("Edge sharpness consistent across frames")
    else:
        findings.append("Pixel-level checks skipped (video file unavailable)")

    risk_score = min(risk_score, 100)

    if risk_score <= 20:
        risk_level = "LOW"
    elif risk_score <= 50:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    summary = (
        f"{total_frames} frames analyzed. {saved_frames} key frames extracted. "
        f"{len(findings)} heuristic checks completed. "
        "This is signal-based screening, not AI-based deepfake detection — "
        "treat results as a starting point for manual review."
    )

    recommendation = (
        "No strong heuristic red flags found. Manual review still advised for any "
        "publication decision — this tool cannot confirm authenticity."
        if risk_level == "LOW"
        else
        "One or more heuristic checks flagged unusual signals. Manual review strongly recommended "
        "before drawing any conclusions."
    )

    return {
        "status": "completed",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "metadata_present": metadata_present,
        "frames_analyzed": frames,
        "findings": findings,
        "summary": summary,
        "recommendation": recommendation,
        "disclaimer": "Heuristic screening only — not an AI deepfake detector. Results are not a verdict."
    }
