def generate_report(metadata, transcript, frames):

    risk_score = 0

    findings = []

    metadata_present = bool(metadata)

    if metadata_present:
        findings.append("Metadata detected")
    else:
        findings.append("Metadata missing")
        risk_score += 25

    total_frames = 0
    saved_frames = 0

    if isinstance(frames, dict):

        total_frames = frames.get(
            "total_frames",
            0
        )

        saved_frames = frames.get(
            "saved_frames",
            0
        )

    if total_frames < 100:

        findings.append(
            "Video unusually short"
        )

        risk_score += 10

    elif total_frames > 500:

        findings.append(
            "Video length sufficient for review"
        )

    if saved_frames < 3:

        findings.append(
            "Very few key frames extracted"
        )

        risk_score += 10

    if risk_score <= 20:
        risk_level = "LOW"

    elif risk_score <= 50:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    summary = (
        f"{total_frames} frames analyzed. "
        f"{saved_frames} key frames extracted. "
        f"{len(findings)} verification checks completed."
    )

    recommendation = (
        "Appears authentic. Manual review optional."
        if risk_level == "LOW"
        else
        "Manual verification recommended."
    )

    return {

        "status": "completed",

        "risk_score": risk_score,

        "risk_level": risk_level,

        "metadata_present": metadata_present,

        "frames_analyzed": frames,

        "findings": findings,

        "summary": summary,

        "recommendation": recommendation
    }
