import cv2
import os

def extract_frames(video_path):
    out = "outputs/frames"
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
            filepath = f"{out}/{filename}"

            cv2.imwrite(filepath, frame)

            saved.append(filename)

        count += 1

    cap.release()

    return {
        "total_frames": count,
        "saved_frames": len(saved),
        "samples": saved[:10]
    }
