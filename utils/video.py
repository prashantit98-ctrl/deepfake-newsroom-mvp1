import cv2, os
def extract_frames(video_path):
    out='outputs/frames'; os.makedirs(out, exist_ok=True)
    cap=cv2.VideoCapture(video_path)
    count=0
    while True:
        ok, frame=cap.read()
        if not ok: break
        if count % 30 == 0:
            cv2.imwrite(f"{out}/frame_{count}.jpg", frame)
        count += 1
    cap.release()
    return count
