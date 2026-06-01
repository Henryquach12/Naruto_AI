import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        # Landmark indices for fingertips and MCP joints
        self.FINGERTIPS = [4, 8, 12, 16, 20]
        self.MCP_JOINTS = [2, 5, 9, 13, 17]

    def find_hands(self, frame):
        """Process frame and return annotated frame + list of hand data."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hand_list = []

        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
                hand_list.append(landmarks)

        return frame, hand_list

    def is_hand_raised(self, landmarks, frame_height):
        """Return True when the wrist is in the upper 60% of the frame."""
        wrist_y = landmarks[0][1]
        return wrist_y < frame_height * 0.6

    def get_palm_center(self, landmarks):
        """Return the pixel (x, y) of the palm center."""
        palm_indices = [0, 5, 9, 13, 17]
        xs = [landmarks[i][0] for i in palm_indices]
        ys = [landmarks[i][1] for i in palm_indices]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def get_hand_size(self, landmarks):
        """Estimate hand size as distance from wrist to middle finger MCP."""
        wrist = np.array(landmarks[0])
        mid_mcp = np.array(landmarks[9])
        return float(np.linalg.norm(mid_mcp - wrist))

    def draw_landmarks(self, frame, hand_list):
        """Draw subtle landmark dots on the frame."""
        for landmarks in hand_list:
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 3, (0, 255, 180), -1)
        return frame
