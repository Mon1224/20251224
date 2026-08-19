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

# Video formats from which the audio track is extracted before analysis.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv", ".mpeg", ".mpg"}


def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    """
    Extracts audio from a video file and saves it as an audio file.

    Args:
        video_path: Path to the input video file.
        output_audio_path: Path to save the extracted audio file.

    Returns:
        Path to the extracted audio file.

    Raises:
        ValueError: If the video cannot be opened or has no audio track.
    """
    try:
        video_clip = VideoFileClip(video_path)
    except Exception as e:
        raise ValueError(f"Could not open video file {video_path}: {e}") from e

    try:
        audio_clip = video_clip.audio
        if audio_clip is None:
            raise ValueError(f"Video file {video_path} has no audio track.")
        audio_clip.write_audiofile(output_audio_path)
        audio_clip.close()
    finally:
        video_clip.close()

    return output_audio_path


def load_whisper_model():
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        return model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None


def transcribe_audio(audio_file):
    """
    Transcribe the audio file using faster-whisper.

    Returns:
        str: Transcribed text, or None if transcription failed (missing file,
        model load failure, exception, or empty/silent audio).
    """
    if not audio_file or not os.path.exists(audio_file):
        print(f"No audio file exists at the specified path: {audio_file}")
        return None

    model = load_whisper_model()
    if not model:
        print("Model failed to load. Please check system resources or model path.")
        return None

    try:
        print("Model loaded successfully. Transcribing audio...")
        segments, _ = model.transcribe(audio_file)
        full_text = " ".join(segment.text for segment in segments).strip()
        return full_text if full_text else None

    except Exception as e:
        print(f"Error transcribing audio with faster-whisper: {e}")
        return None


def log_before_call(fc):
    """Pre-hook function that runs before the tool execution"""
    print(f"About to call function with arguments: {fc.arguments}")


def log_after_call(fc):
    """Post-hook function that runs after the tool execution"""
    print(f"Function call completed with result: {fc.result}")


def error_response(message: str) -> str:
    """Build a JSON error response so failures are explicit, not treated as speech."""
    return json.dumps({
        "error": message,
        "transcription": "",
        "speech_rate_wpm": "N/A",
        "pitch_variation": "N/A",
        "volume_consistency": "N/A",
    })


@tool(
    name="analyze_voice_attributes",  # Custom name for the tool (otherwise the function name is used)
    description="Analyzes vocal attributes like clarity, intonation, and pace.",
    # Custom description (otherwise the function docstring is used)
    show_result=True,  # Show result after function call
    stop_after_tool_call=True,  # Return the result immediately after the tool call and stop the agent
    pre_hook=log_before_call,  # Hook to run before execution
    post_hook=log_after_call,  # Hook to run after execution
    cache_results=False,  # Enable caching of results
    cache_dir="/tmp/agno_cache",  # Custom cache directory
    cache_ttl=3600  # Cache TTL in seconds (1 hour)
)
def analyze_voice_attributes(file_path: str) -> dict:
    """
    Analyzes vocal attributes in an audio or video file.

    Args:
        file_path: The path to the audio file (or a video file whose audio track
            will be extracted first).

    Returns:
        A JSON string with the transcription and voice metrics, or a JSON error
        response when the file cannot be processed.
    """
    if not file_path or not os.path.exists(file_path):
        return error_response("Audio file does not exist at the specified path.")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    temp_audio_path = None
    try:
        # If the file is a video, extract its audio track first.
        if ext in VIDEO_EXTENSIONS:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
                temp_audio_path = temp_audio_file.name
            # The handle is closed above on purpose: keeping it open would lock
            # the file on Windows while moviepy writes to the same path.
            try:
                audio_path = extract_audio_from_video(file_path, temp_audio_path)
            except Exception as e:
                print(f"Audio extraction failed: {e}")
                return error_response(str(e))
        else:
            audio_path = file_path

        # Transcribe first. On failure return an explicit error: no WPM/pitch/
        # volume is computed and nothing is passed on to the content agent.
        transcription = transcribe_audio(audio_path)
        if transcription is None:
            return error_response("Transcription failed. Check the audio quality and try again.")

        # Load the audio for metric computation; a corrupt file must not crash the tool.
        try:
            y, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Failed to load audio with librosa: {e}")
            return error_response(f"Failed to load audio file: {e}")

        duration = librosa.get_duration(y=y, sr=sr)
        if duration <= 0:
            return error_response("Audio duration is zero, cannot compute speech metrics.")

        words = transcription.split()
        speech_rate = len(words) / (duration / 60.0)  # words per minute

        # Pitch variation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_variation = np.std(pitch_values) if pitch_values.size > 0 else 0

        # Volume consistency
        rms = librosa.feature.rms(y=y)[0]
        volume_consistency = np.std(rms)

        return json.dumps({
            "transcription": transcription,
            "speech_rate_wpm": str(round(speech_rate, 2)),
            "pitch_variation": str(round(pitch_variation, 2)),
            "volume_consistency": str(round(volume_consistency, 4)),
        })
    finally:
        # Clean up the temporary audio file even if something failed above.
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
