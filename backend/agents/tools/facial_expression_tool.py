import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from agno.tools import tool

# Face Landmarker model used by the current MediaPipe Tasks API. The legacy
# `mp.solutions.face_mesh` API was removed in newer MediaPipe releases.
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent / "models" / "face_landmarker.task"
)


def log_before_call(fc):
    """Pre-hook function that runs before the tool execution"""
    print(f"About to call function with arguments: {fc.arguments}")


def log_after_call(fc):
    """Post-hook function that runs after the tool execution"""
    print(f"Function call completed with result: {fc.result}")


def _error_response(message: str) -> str:
    """Build a JSON error response so failures are explicit, not empty results."""
    return json.dumps({
        "error": message,
        "emotion_timeline": [],
        "engagement_metrics": {
            "eye_contact_frequency": 0,
            "smile_frequency": 0,
        },
    })


def _load_face_landmarker():
    """Load the FaceLandmarker, downloading the model on first use if needed."""
    if not FACE_LANDMARKER_MODEL_PATH.exists():
        print(f"Face landmark model not found; downloading from {FACE_LANDMARKER_MODEL_URL} ...")
        FACE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download the face landmark model to {FACE_LANDMARKER_MODEL_PATH}: {e}"
            ) from e

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


@tool(
    name="analyze_facial_expressions",              # Custom name for the tool (otherwise the function name is used)
    description="Analyzes facial expressions to detect emotions and engagement.",  # Custom description (otherwise the function docstring is used)
    show_result=True,                               # Show result after function call
    stop_after_tool_call=True,                      # Return the result immediately after the tool call and stop the agent
    pre_hook=log_before_call,                       # Hook to run before execution
    post_hook=log_after_call,                       # Hook to run after execution
    cache_results=False,                            # Enable caching of results
    cache_dir="/tmp/agno_cache",                    # Custom cache directory
    cache_ttl=3600                                  # Cache TTL in seconds (1 hour)
)
def analyze_facial_expressions(video_path: str) -> dict:
    """
    Analyzes facial expressions in a video to detect emotions and engagement.

    Args:
        video_path: The path to the video file.

    Returns:
        A JSON string containing the emotion timeline and engagement metrics,
        or an error response if the video/model cannot be processed.
    """
    try:
        landmarker = _load_face_landmarker()
    except Exception as e:
        print(f"Failed to load face landmarker: {e}")
        return _error_response(str(e))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        landmarker.close()
        return _error_response(f"Could not open video file: {video_path}")

    emotion_timeline = []
    eye_contact_count = 0
    smile_count = 0
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample roughly once per second: one emotion/engagement reading per second
    # is plenty for a speech timeline, and it cuts DeepFace calls ~30x on a
    # 30fps video (and more on higher frame rates).
    frame_interval = max(int(round(fps)) if fps and fps > 0 else 5, 1)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Timestamps must be monotonically increasing for VIDEO mode.
            timestamp_ms = int(frame_count * 1000 / fps) if fps else frame_count
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    landmarks = face_landmarks.landmark

                    # Convert landmarks to pixel coordinates
                    h, w, _ = frame.shape
                    landmark_coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

                    # Emotion detection. DeepFace is only reached when the
                    # landmarker has already found a face; if its own detector
                    # disagrees on a frame, that frame's emotion is skipped
                    # without interrupting the loop or the engagement metrics
                    # below. enforce_detection=True makes DeepFace raise
                    # (caught here) instead of guessing an emotion for a frame
                    # it considers faceless.
                    try:
                        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True)
                        emotion = analysis[0]['dominant_emotion']
                        if emotion == "happy":
                            smile_count += 1

                        timestamp = (frame_count / fps) if fps else frame_count
                        emotion_timeline.append({"timestamp": round(timestamp, 2), "emotion": emotion})
                    except Exception as e:
                        print(f"Emotion detection skipped for frame {frame_count}: {e}")

                    # Engagement Metric: Eye contact estimation
                    # Using eye landmarks: 159 (left eye upper lid), 145 (left eye lower lid),
                    # 386 (right eye upper lid), 374 (right eye lower lid)
                    left_eye_upper_lid = landmark_coords[159]
                    left_eye_lower_lid = landmark_coords[145]
                    right_eye_upper_lid = landmark_coords[386]
                    right_eye_lower_lid = landmark_coords[374]

                    left_eye_opening = np.linalg.norm(np.array(left_eye_upper_lid) - np.array(left_eye_lower_lid))
                    right_eye_opening = np.linalg.norm(np.array(right_eye_upper_lid) - np.array(right_eye_lower_lid))

                    eye_opening_avg = (left_eye_opening + right_eye_opening) / 2

                    # Simple heuristic: if eyes are wide open, assume eye contact
                    if eye_opening_avg > 5:  # Threshold adjustment through experimentation
                        eye_contact_count += 1
    finally:
        cap.release()
        landmarker.close()

    total_processed_frames = frame_count // frame_interval
    if total_processed_frames == 0:
        total_processed_frames = 1  # Avoid division by zero

    return json.dumps({
        "emotion_timeline": emotion_timeline,
        "engagement_metrics": {
            "eye_contact_frequency": eye_contact_count / total_processed_frames,
            "smile_frequency": smile_count / total_processed_frames
        }
    })
