import cv2
import os


def extract_frames(video_path, output_id):
    """
    Extracts every 30th frame from the video and saves it as a JPG.

    output_id: a unique identifier (e.g. the upload's UUID) so each
    video's frames live in their own folder and never collide with
    another upload's frames.
    """
    out = f"outputs/frames/{output_id}"
    os.makedirs(out, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count % 30 == 0:
            filename = f"frame_{count}.jpg"
            filepath = os.path.join(out, filename)
            cv2.imwrite(filepath, frame)
            # Store the relative path (output_id/filename) so the
            # frontend can build a working URL to fetch this exact frame.
            saved.append(f"{output_id}/{filename}")
        count += 1

    cap.release()

    return {
        "total_frames": count,
        "saved_frames": len(saved),
        "samples": saved[:10]
    }
