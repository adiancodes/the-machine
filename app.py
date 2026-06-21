from flask import Flask, render_template, request, jsonify
import yt_dlp
from faster_whisper import WhisperModel
import os
import glob

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the HTML template from the "templates" folder
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
        
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
            ydl.download([url])
            
        # 2. Find the downloaded file (it might be temp_audio.mp3)
        audio_file = None
        for file in glob.glob("temp_audio.*"):
            audio_file = file
            break
            
        if not audio_file:
            return jsonify({'error': 'Failed to download audio.'}), 500
            
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
            
        return jsonify({'text': transcript.strip()})
        
    except Exception as e:
        # If there's an error, try to clean up any temp files anyway
        for file in glob.glob("temp_audio.*"):
            if os.path.exists(file):
                os.remove(file)
                
        # Send the exact error message back to the frontend
        return jsonify({'error': f"An error occurred: {str(e)}. Please make sure FFMPEG is installed."}), 500

if __name__ == '__main__':
    # Run the server on http://127.0.0.1:5000/
    app.run(debug=True)
