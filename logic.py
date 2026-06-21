import yt_dlp
from faster_whisper import WhisperModel
import os
import glob

def extract_transcript(video_url: str) -> str:
    try:
        # 1. Configuration for yt-dlp to download only the audio
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_audio.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True # Reduce console spam
        }
        
        # Download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
            
        # 2. Find the downloaded file (it might be temp_audio.mp3)
        audio_file = None
        for file in glob.glob("temp_audio.*"):
            audio_file = file
            break
            
        if not audio_file:
            raise Exception("Failed to download audio.")
            
        # 3. Load faster-whisper model
        # "base" model is small and fast, perfect for local MVP
        model = WhisperModel("base", device="cpu", compute_type="int8")
        
        # Transcribe the audio file
        segments, info = model.transcribe(audio_file, beam_size=5)
        
        # Combine all spoken segments into a single paragraph
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
            
        # 4. Clean up the temporary file so it doesn't take up space
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        return transcript.strip()
        
    except Exception as e:
        # If there's an error, try to clean up any temp files anyway
        for file in glob.glob("temp_audio.*"):
            if os.path.exists(file):
                os.remove(file)
                
        # Raise the exception to be handled by the route
        raise Exception(f"An error occurred: {str(e)}. Please make sure FFMPEG is installed.")
