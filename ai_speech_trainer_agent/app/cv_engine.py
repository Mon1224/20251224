import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List


class FacialExpressionCVEngine:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        # landmark index
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291
        self.UPPER_LIP = 13
        self.LOWER_LIP = 14
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374

    def analyze(self, video_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)

        emotion_timeline: List[Dict] = []
        frame_count = 0
        smile_count = 0
        eye_contact_count = 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 5

        print("[CV] Start analyzing video:", video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            h, w, _ = frame.shape

            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([
                    (int(lm.x * w), int(lm.y * h))
                    for lm in face_landmarks.landmark
                ])

                # 😊 Smile heuristic
                mouth_width = np.linalg.norm(
                    landmarks[self.LEFT_MOUTH] - landmarks[self.RIGHT_MOUTH]
                )
                mouth_height = np.linalg.norm(
                    landmarks[self.UPPER_LIP] - landmarks[self.LOWER_LIP]
                )

                smile_ratio = mouth_width / (mouth_height + 1e-6)
                is_smiling = smile_ratio > 1.8

                if is_smiling:
                    smile_count += 1
                    emotion = "happy"
                else:
                    emotion = "neutral"

                timestamp = round(frame_count / fps, 2)
                emotion_timeline.append({
                    "timestamp": timestamp,
                    "emotion": emotion
                })

                # 👀 Eye contact heuristic
                left_eye = np.linalg.norm(
                    landmarks[self.LEFT_EYE_TOP] - landmarks[self.LEFT_EYE_BOTTOM]
                )
                right_eye = np.linalg.norm(
                    landmarks[self.RIGHT_EYE_TOP] - landmarks[self.RIGHT_EYE_BOTTOM]
                )

                if (left_eye + right_eye) / 2 > 5:
                    eye_contact_count += 1

        cap.release()
        self.face_mesh.close()

        total_frames = max(frame_count // frame_interval, 1)

        print("[CV] Analysis finished")

        return {
            "emotion_timeline": emotion_timeline,
            "engagement_metrics": {
                "smile_frequency": round(smile_count / total_frames, 3),
                "eye_contact_frequency": round(eye_contact_count / total_frames, 3)
            }
        }