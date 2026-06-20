def generate_report(metadata, transcript, frames):

    risk_score = 10

    if not metadata:
        risk_score += 20

    if isinstance(frames, dict):
        total_frames = frames.get("total_frames", 0)
        saved_frames = frames.get("saved_frames", 0)
    else:
        total_frames = 0
        saved_frames = 0

    if total_frames < 100:
        risk_score += 10

    if risk_score <= 20:
        risk_level = "LOW"

    elif risk_score <= 50:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    return {
        "status": "completed",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "frames_analyzed": frames,
        "transcript": transcript,
        "metadata_present": bool(metadata),
        "summary":
            f"{total_frames} frames analyzed. "
            f"{saved_frames} key frames extracted. "
            f"Risk level assessed as {risk_level}."
    }
