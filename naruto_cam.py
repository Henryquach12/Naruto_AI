# naruto_cam.py — Real-time Naruto transformation webcam app
#
# Install:
#   pip install opencv-python mediapipe numpy
#
# Run:
#   python naruto_cam.py
#
# Raise your hand (wrist above 40% of frame height) to transform into
# Naruto: spiky hair + headband + orange jacket overlays, plus a glowing
# Rasengan rendered on your palm. Press Q to quit.
#
# NOTE: MediaPipe 0.10.30+ removed the legacy `mp.solutions` API, so this
# uses the equivalent lightweight Tasks API models (BlazePalm lite +
# BlazeFace short-range) — the same models `model_complexity=0` selected.

import os
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp


# ── tunables ─────────────────────────────────────────────────────────────────
DISPLAY_W, DISPLAY_H = 640, 480     # max processing/render resolution
INFER_W,   INFER_H   = 320, 240     # downscaled copy fed to MediaPipe
RAISE_THRESHOLD      = 0.40         # wrist y < 40% of frame height = raised
MAX_HANDS            = 2
FACE_SKIP            = 2            # run face detection every Nth frame

WINDOW_TITLE = "Naruto Cam"
ASSET_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))

ORANGE = (0, 110, 255)              # BGR


# ── model files (auto-downloaded once, ~2 MB total) ──────────────────────────
HAND_MODEL = {
    "path": os.path.join(MODEL_DIR, "hand_landmarker.task"),
    "url": ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"),
}
FACE_MODEL = {
    "path": os.path.join(MODEL_DIR, "blaze_face_short_range.tflite"),
    "url": ("https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"),
}


def ensure_model(model):
    """Download a MediaPipe model file on first run."""
    if not os.path.exists(model["path"]):
        print(f"Downloading {os.path.basename(model['path'])} ...")
        urllib.request.urlretrieve(model["url"], model["path"])
        print("Model ready.")
    return model["path"]
