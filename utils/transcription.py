import whisper
import subprocess
import os

model = None


def get_model():
    global model

    if model is None:
        model = whisper.load_model("tiny")

    return model


def transcribe_audio(video_path):

    os.makedirs("outputs", exist_ok=True)

    audio = "outputs/audio.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            video_path,
            audio,
            "-y"
        ],
        capture_output=True
    )

    model = get_model()

    result = model.transcribe(audio)

    return result["text"]
