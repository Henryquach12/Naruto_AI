import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time


class HandDetector:
    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    _MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self._ensure_model()

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def _ensure_model(self):
        if not os.path.exists(self._MODEL_PATH):
            print("Downloading hand landmark model (~2 MB)...")
            urllib.request.urlretrieve(self._MODEL_URL, self._MODEL_PATH)
            print("Model ready.")

    def find_hands(self, frame):
        """Process frame and return frame + list of hand landmark lists."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        results = self._detector.detect_for_video(mp_image, timestamp_ms)

        hand_list = []
        h, w = frame.shape[:2]
        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                hand_list.append(landmarks)

        return frame, hand_list

    def is_hand_raised(self, landmarks, frame_height, thresh=0.80):
        """True when wrist is above `thresh` fraction down the frame."""
        return landmarks[0][1] < frame_height * thresh

    def get_palm_center(self, landmarks):
        """Pixel (x, y) of the palm center."""
        palm_indices = [0, 5, 9, 13, 17]
        xs = [landmarks[i][0] for i in palm_indices]
        ys = [landmarks[i][1] for i in palm_indices]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def get_hand_size(self, landmarks):
        """Distance from wrist to middle-finger MCP as a size proxy."""
        wrist   = np.array(landmarks[0])
        mid_mcp = np.array(landmarks[9])
        return float(np.linalg.norm(mid_mcp - wrist))

    def draw_landmarks(self, frame, hand_list):
        for landmarks in hand_list:
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 3, (0, 255, 180), -1)
        return frame
