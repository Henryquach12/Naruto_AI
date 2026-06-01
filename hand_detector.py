import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os


class HandDetector:
    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    _MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

    def __init__(self, max_hands=4, detection_confidence=0.5, tracking_confidence=0.5):
        self._ensure_model()

        # IMAGE mode re-detects every frame — most reliable for multi-hand
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _ensure_model(self):
        if not os.path.exists(self._MODEL_PATH):
            print("Downloading hand landmark model (~2 MB)...")
            urllib.request.urlretrieve(self._MODEL_URL, self._MODEL_PATH)
            print("Model ready.")

    def find_hands(self, frame):
        """Detect hands in frame. Returns (frame, list_of_landmark_lists).

        Detects on a half-resolution copy for speed; landmarks are normalised
        (0-1) so they map back to the full-resolution frame correctly.
        """
        h, w  = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = self._detector.detect(mp_image)

        hand_list = []
        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                hand_list.append(landmarks)

        return frame, hand_list

    def get_palm_center(self, landmarks):
        xs = [landmarks[i][0] for i in [0, 5, 9, 13, 17]]
        ys = [landmarks[i][1] for i in [0, 5, 9, 13, 17]]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def is_palm_facing_camera(self, landmarks):
        """
        True when the palm side (not the dorsal side) faces the camera.

        Uses the 2D cross product of:
          v1 = index MCP (5)  - wrist (0)
          v2 = pinky MCP (17) - wrist (0)

        In the mirrored frame the cross product sign is opposite for left vs
        right hands, so we multiply by sign(index.x - pinky.x) to get a
        handedness-independent value:
          - negative → palm facing camera   (for BOTH hands)
          - positive → dorsal facing camera
        """
        wx, wy = landmarks[0]
        ix, iy = landmarks[5]    # index MCP
        px, py = landmarks[17]   # pinky MCP
        v1 = (ix - wx, iy - wy)
        v2 = (px - wx, py - wy)
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        mag = ((v1[0]**2 + v1[1]**2) * (v2[0]**2 + v2[1]**2)) ** 0.5
        if mag < 1e-6:
            return False
        norm_cross = cross_z / mag          # sin of angle, in [-1, 1]
        lateral    = 1 if ix > px else -1   # +1 right-hand, -1 left-hand appearance
        return norm_cross * lateral < -0.1

    def get_hand_size(self, landmarks):
        wrist   = np.array(landmarks[0])
        mid_mcp = np.array(landmarks[9])
        return float(np.linalg.norm(mid_mcp - wrist))

    def draw_landmarks(self, frame, hand_list):
        colors = [(0, 255, 180), (0, 180, 255)]
        for i, landmarks in enumerate(hand_list):
            col = colors[i % len(colors)]
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 4, col, -1)
            # draw numbered circle at palm center
            cx, cy = self.get_palm_center(landmarks)
            cv2.circle(frame, (cx, cy), 18, col, 3, cv2.LINE_AA)
            cv2.putText(frame, str(i + 1), (cx - 6, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        return frame
