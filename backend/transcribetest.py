from pydub import AudioSegment
import io
import requests
import os 
def transcribe_audio(file_path, server_url="http://localhost:8080"):
    """
    Convert audio to WAV format and send to whisper.cpp server
    """
    
    # Convert to WAV format (required by whisper.cpp)
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Mono, 16kHz
    
    # Save to in-memory WAV buffer
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    buffer.seek(0)
    
    # Send to whisper.cpp server - CORRECTED VERSION
    files = {'file': ('audio.wav', buffer, 'audio/wav')}  # Field name must be 'file'
    data = {
        'response_format': 'json',  # Required for JSON response
    }
    
    response = requests.post(
        f"{server_url}/inference", 
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        return response.json()['text']
    else:
        raise Exception(f"Transcription failed: {response.text}")

# Usage example
if __name__ == "__main__":
    text = transcribe_audio( "./backend/testaudio/alexa1.wav")
    print(text)