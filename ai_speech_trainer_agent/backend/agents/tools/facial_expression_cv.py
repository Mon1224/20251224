import cv2
import numpy as np
import mediapipe as mp
import gc

from .facial_expression_base import FacialExpressionAnalyzer


# ===============================
# 🔥 全局单例 FaceMesh
# ===============================
mp_face_mesh = mp.solutions.face_mesh
global_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


class CVFacialExpressionAnalyzer(FacialExpressionAnalyzer):

    def analyze(self, video_path: str) -> dict:

        cap = None

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")

            emotion_timeline = []
            eye_contact_count = 0
            smile_count = 0
            frame_count = 0

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = 5

            LEFT_MOUTH = 61
            RIGHT_MOUTH = 291
            UPPER_LIP = 13
            LOWER_LIP = 14

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue

                frame = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 🔥 使用全局 face_mesh
                results = global_face_mesh.process(rgb_frame)

                if not results.multi_face_landmarks:
                    continue

                h, w, _ = frame.shape

                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark
                    landmark_coords = np.array([
                        (int(lm.x * w), int(lm.y * h)) for lm in landmarks
                    ])

                    mouth_width = np.linalg.norm(
                        landmark_coords[LEFT_MOUTH] - landmark_coords[RIGHT_MOUTH]
                    )
                    mouth_height = np.linalg.norm(
                        landmark_coords[UPPER_LIP] - landmark_coords[LOWER_LIP]
                    )

                    smile_ratio = mouth_width / (mouth_height + 1e-6)

                    if smile_ratio > 1.8:
                        smile_count += 1
                        emotion = "happy"
                    else:
                        emotion = "neutral"

                    timestamp = round(frame_count / fps, 2)
                    emotion_timeline.append({
                        "timestamp": timestamp,
                        "emotion": emotion
                    })

                    left_eye_opening = np.linalg.norm(
                        landmark_coords[159] - landmark_coords[145]
                    )
                    right_eye_opening = np.linalg.norm(
                        landmark_coords[386] - landmark_coords[374]
                    )

                    if (left_eye_opening + right_eye_opening) / 2 > 5:
                        eye_contact_count += 1

            total_frames = max(frame_count // frame_interval, 1)

            return {
                "emotion_timeline": emotion_timeline,
                "engagement_metrics": {
                    "eye_contact_frequency": round(eye_contact_count / total_frames, 3),
                    "smile_frequency": round(smile_count / total_frames, 3)
                }
            }

        finally:
            if cap is not None:
                cap.release()

            gc.collect()
