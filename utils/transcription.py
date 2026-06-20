import whisper, subprocess, os
model=whisper.load_model('tiny')
def transcribe_audio(video_path):
    os.makedirs('outputs', exist_ok=True)
    audio='outputs/audio.wav'
    subprocess.run(['ffmpeg','-i',video_path,audio,'-y'],capture_output=True)
    return model.transcribe(audio)['text']
