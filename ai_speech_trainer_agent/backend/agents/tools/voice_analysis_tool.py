import os
import json
import tempfile
import numpy as np
import librosa
from moviepy import VideoFileClip
from faster_whisper import WhisperModel
from agno.tools import tool
from dotenv import load_dotenv

load_dotenv()


# =========================
# Utility: Extract audio
# =========================
def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)
    audio_clip.close()
    video_clip.close()
    return output_audio_path


# =========================
# Whisper model loader
# =========================
def load_whisper_model():
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        return model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None


# =========================
# Core transcription logic
# =========================
def transcribe_audio(audio_file):
    if not audio_file or not os.path.exists(audio_file):
        return "No audio file exists at the specified path."

    model = load_whisper_model()
    if not model:
        return "Model failed to load."

    try:
        print("Model loaded successfully. Transcribing audio...")
        segments, _ = model.transcribe(audio_file)
        full_text = " ".join(segment.text for segment in segments)
        return full_text.strip() if full_text else "I couldn't understand the audio."

    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return "Transcription failed."


# =========================
# 🔥 NEW: TRANSCRIBE TOOL
# =========================
@tool(
    name="transcribe",
    description="Extracts audio from video and returns the raw transcript text.",
    show_result=True,
    stop_after_tool_call=True,
    cache_results=False
)
def transcribe(file_path: str) -> str:
    """
    Extract transcript text from a video or audio file.
    """

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # If video → extract audio
    if ext in [".mp4"]:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            audio_path = extract_audio_from_video(file_path, temp_audio_file.name)
    else:
        audio_path = file_path

    transcript = transcribe_audio(audio_path)

    # Cleanup temp file
    if ext in [".mp4"] and os.path.exists(audio_path):
        os.remove(audio_path)

    return transcript


# =========================
# Voice analysis tool
# =========================
@tool(
    name="analyze_voice_attributes",
    description="Analyzes vocal attributes like clarity, intonation, and pace.",
    show_result=True,
    stop_after_tool_call=True,
    cache_results=False
)
def analyze_voice_attributes(file_path: str) -> dict:

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in ['.mp4']:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio_file:
            audio_path = extract_audio_from_video(file_path, temp_audio_file.name)
    else:
        audio_path = file_path

    transcription = transcribe_audio(audio_path)

    y, sr = librosa.load(audio_path, sr=16000)

    words = transcription.split()
    duration = librosa.get_duration(y=y, sr=sr)
    speech_rate = len(words) / (duration / 60.0) if duration > 0 else 0

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_variation = np.std(pitch_values) if pitch_values.size > 0 else 0

    rms = librosa.feature.rms(y=y)[0]
    volume_consistency = np.std(rms)

    if ext in ['.mp4'] and os.path.exists(audio_path):
        os.remove(audio_path)

    return json.dumps({
        "transcription": transcription,
        "speech_rate_wpm": str(round(speech_rate, 2)),
        "pitch_variation": str(round(pitch_variation, 2)),
        "volume_consistency": str(round(volume_consistency, 4))
    })
