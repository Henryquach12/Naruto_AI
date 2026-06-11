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


# ── detector setup ───────────────────────────────────────────────────────────
def create_hand_detector():
    """Lightweight hand landmarker (lite model, IMAGE mode)."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model(HAND_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def create_face_detector():
    """BlazeFace short-range — the lightest face model MediaPipe ships."""
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model(FACE_MODEL)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return mp.tasks.vision.FaceDetector.create_from_options(options)


def detect_hands(detector, infer_rgb, disp_w, disp_h):
    """Run hand inference on the downscaled RGB copy; return landmark lists
    scaled up to display coordinates."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)
    results = detector.detect(mp_image)
    hands = []
    if results.hand_landmarks:
        for lms in results.hand_landmarks:
            hands.append([(int(lm.x * disp_w), int(lm.y * disp_h)) for lm in lms])
    return hands


def detect_face(detector, infer_rgb, disp_w, disp_h):
    """Run face inference on the downscaled RGB copy; return the largest face
    bounding box (x, y, w, h) in display coordinates, or None."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=infer_rgb)
    results = detector.detect(mp_image)
    if not results.detections:
        return None
    sx = disp_w / infer_rgb.shape[1]
    sy = disp_h / infer_rgb.shape[0]
    best = max(results.detections,
               key=lambda d: d.bounding_box.width * d.bounding_box.height)
    bb = best.bounding_box
    return (int(bb.origin_x * sx), int(bb.origin_y * sy),
            int(bb.width * sx), int(bb.height * sy))


# ── PNG assets (optional — procedural fallback used when missing) ────────────
def load_assets():
    """Load RGBA overlay PNGs from ./assets if present.
    Expected files: hair.png, headband.png, jacket.png."""
    assets = {}
    for name in ("hair", "headband", "jacket"):
        path = os.path.join(ASSET_DIR, f"{name}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                assets[name] = img
    return assets


def overlay_rgba(frame, rgba, x, y, width):
    """Alpha-blend an RGBA image onto frame, resized to `width` px wide and
    centred horizontally at x, with its top edge at y. Clips at frame edges."""
    fh, fw = frame.shape[:2]
    scale = width / rgba.shape[1]
    height = max(1, int(rgba.shape[0] * scale))
    rgba = cv2.resize(rgba, (width, height), interpolation=cv2.INTER_LINEAR)

    x0 = x - width // 2
    y0 = y
    x1, y1 = x0 + width, y0 + height
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, fw), min(y1, fh)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    crop = rgba[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    roi = frame[cy0:cy1, cx0:cx1]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (crop[:, :, :3].astype(np.float32) * alpha
              + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
