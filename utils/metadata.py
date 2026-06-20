import subprocess
def get_metadata(video_path):
    r=subprocess.run(['ffprobe','-v','quiet','-print_format','json','-show_format',video_path],capture_output=True,text=True)
    return r.stdout
