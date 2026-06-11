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


# ── procedural fallback art (used when no PNG assets exist) ──────────────────
BLONDE      = (60, 215, 255)    # Naruto's spiky hair
BLONDE_DARK = (20, 165, 230)
STEEL       = (150, 150, 150)   # headband plate
STEEL_DARK  = (90, 90, 90)
CLOTH_BLUE  = (110, 60, 20)     # headband cloth


def draw_spiky_hair(frame, face):
    """Yellow filled triangles fanning out above the face bounding box."""
    x, y, w, h = face
    base_y = y + int(h * 0.18)              # spikes root just above eyebrows
    n = 8
    for i in range(n):
        sx0 = x - int(w * 0.15) + int((w * 1.30) * i / n)
        sx1 = sx0 + int(w * 1.30 / n)
        tip_x = (sx0 + sx1) // 2 + int(w * 0.10 * math.sin(i * 2.1))
        tip_y = base_y - int(h * (0.55 + 0.25 * math.sin(i * 1.7 + 0.6)))
        pts = np.array([[sx0, base_y], [sx1, base_y], [tip_x, tip_y]], np.int32)
        cv2.fillPoly(frame, [pts], BLONDE, cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, BLONDE_DARK, 1, cv2.LINE_AA)
    # side spikes hugging the temples
    for side in (-1, 1):
        bx = x + (w if side > 0 else 0)
        pts = np.array([[bx, base_y - int(h * 0.05)],
                        [bx, base_y + int(h * 0.30)],
                        [bx + side * int(w * 0.28), base_y + int(h * 0.05)]], np.int32)
        cv2.fillPoly(frame, [pts], BLONDE, cv2.LINE_AA)


def draw_headband(frame, face):
    """Grey plate with the Konoha leaf symbol on a dark-blue cloth band."""
    x, y, w, h = face
    band_y0 = y + int(h * 0.16)
    band_y1 = y + int(h * 0.34)
    cv2.rectangle(frame, (x - int(w * 0.12), band_y0),
                  (x + w + int(w * 0.12), band_y1), CLOTH_BLUE, -1)
    # metal plate
    px0 = x + int(w * 0.26)
    px1 = x + int(w * 0.74)
    cv2.rectangle(frame, (px0, band_y0 + 2), (px1, band_y1 - 2), STEEL, -1)
    cv2.rectangle(frame, (px0, band_y0 + 2), (px1, band_y1 - 2), STEEL_DARK, 1)
    # leaf symbol: spiral ellipse + a swoosh line
    cx = (px0 + px1) // 2
    cy = (band_y0 + band_y1) // 2
    r = max(3, (band_y1 - band_y0) // 3)
    cv2.ellipse(frame, (cx, cy), (r, r), 0, -30, 300, STEEL_DARK, 2, cv2.LINE_AA)
    cv2.line(frame, (cx + r, cy - r // 2), (cx + int(r * 2.2), cy - r),
             STEEL_DARK, 2, cv2.LINE_AA)
    # plate rivets
    for rx in (px0 + 5, px1 - 5):
        cv2.circle(frame, (rx, cy), 2, STEEL_DARK, -1, cv2.LINE_AA)


def draw_outfit(frame, face):
    """Orange jacket hint over the torso, estimated from face geometry."""
    x, y, w, h = face
    fh, fw = frame.shape[:2]
    chin_y = y + int(h * 1.05)
    torso_w = int(w * 2.8)
    cx = x + w // 2
    tx0 = max(0, cx - torso_w // 2)
    tx1 = min(fw, cx + torso_w // 2)
    ty1 = min(fh, chin_y + int(h * 2.6))
    if chin_y >= fh or tx0 >= tx1:
        return
    cv2.rectangle(frame, (tx0, chin_y), (tx1, ty1), ORANGE, -1)
    # collar — two dark flaps meeting at a centre zipper
    collar_h = int(h * 0.45)
    for side in (-1, 1):
        pts = np.array([[cx, chin_y],
                        [cx + side * int(w * 0.85), chin_y],
                        [cx, min(fh, chin_y + collar_h)]], np.int32)
        cv2.fillPoly(frame, [pts], (20, 60, 200), cv2.LINE_AA)
    cv2.line(frame, (cx, chin_y), (cx, ty1), (30, 70, 190), 3, cv2.LINE_AA)
    # shoulder stripes
    cv2.line(frame, (tx0, chin_y + collar_h), (tx0 + int(w * 0.5), chin_y),
             (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (tx1, chin_y + collar_h), (tx1 - int(w * 0.5), chin_y),
             (255, 255, 255), 2, cv2.LINE_AA)


def draw_flame_icon(frame, x, y, size):
    """Tiny procedural flame (cv2.putText cannot render the fire emoji)."""
    pts = np.array([[x, y], [x - size // 2, y - size // 2], [x - size // 4, y - size],
                    [x, y - size // 2], [x + size // 4, y - int(size * 1.2)],
                    [x + size // 2, y - size // 3], [x + size // 3, y]], np.int32)
    cv2.fillPoly(frame, [pts], (0, 140, 255), cv2.LINE_AA)
    cv2.circle(frame, (x, y - size // 3), size // 4, (60, 230, 255), -1, cv2.LINE_AA)


# ── Rasengan ─────────────────────────────────────────────────────────────────
class Rasengan:
    """Glowing blue/white energy sphere with a spinning-line animation.

    The glow is built on a small ROI around the palm only — two GaussianBlur
    passes additively blended — so the cost stays constant regardless of
    frame size."""

    DEEP_BLUE  = (220,  90,  10)
    BLUE       = (255, 160,  40)
    LIGHT_BLUE = (255, 220, 160)
    WHITE      = (255, 255, 255)

    def __init__(self):
        self.tick = 0

    def draw(self, frame, center, radius):
        self.tick += 1
        cx, cy = int(center[0]), int(center[1])
        r = max(12, int(radius))
        fh, fw = frame.shape[:2]

        # ROI big enough to hold the glow halo
        pad = int(r * 2.2)
        x0, y0 = max(cx - pad, 0), max(cy - pad, 0)
        x1, y1 = min(cx + pad, fw), min(cy + pad, fh)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        lx, ly = cx - x0, cy - y0          # centre in ROI coords
        roi = frame[y0:y1, x0:x1]

        # paint the energy shapes on a black canvas, blur it, add it
        glow = np.zeros_like(roi)
        cv2.circle(glow, (lx, ly), int(r * 1.45), self.DEEP_BLUE, -1, cv2.LINE_AA)
        cv2.circle(glow, (lx, ly), int(r * 1.05), self.BLUE, -1, cv2.LINE_AA)
        cv2.circle(glow, (lx, ly), int(r * 0.65), self.LIGHT_BLUE, -1, cv2.LINE_AA)
        cv2.circle(glow, (lx, ly), int(r * 0.32), self.WHITE, -1, cv2.LINE_AA)

        # spinning lines — three chord families rotating at different speeds
        for k in range(6):
            a = self.tick * 0.18 + k * math.pi / 3
            x_a = lx + int(r * 0.95 * math.cos(a))
            y_a = ly + int(r * 0.95 * math.sin(a))
            x_b = lx + int(r * 0.95 * math.cos(a + 2.4))
            y_b = ly + int(r * 0.95 * math.sin(a + 2.4))
            cv2.line(glow, (x_a, y_a), (x_b, y_b), self.WHITE, 1, cv2.LINE_AA)
        for k in range(4):
            a = -self.tick * 0.11 + k * math.pi / 2
            cv2.ellipse(glow, (lx, ly), (int(r * 1.0), int(r * 0.35)),
                        math.degrees(a), 0, 360, self.LIGHT_BLUE, 1, cv2.LINE_AA)

        # layered glow: tight blur + wide blur, additively blended
        tight = cv2.GaussianBlur(glow, (0, 0), r * 0.18)
        wide = cv2.GaussianBlur(glow, (0, 0), r * 0.55)
        halo = cv2.addWeighted(tight, 0.9, wide, 0.7, 0)
        roi[:] = cv2.add(roi, halo)        # additive: light only brightens

        # crisp core on top of the glow
        cv2.circle(roi, (lx, ly), int(r * 0.30), self.WHITE, -1, cv2.LINE_AA)
        cv2.circle(roi, (lx, ly), r, self.LIGHT_BLUE, 2, cv2.LINE_AA)
        pulse = int(r * (1.10 + 0.06 * math.sin(self.tick * 0.25)))
        cv2.circle(roi, (lx, ly), pulse, self.BLUE, 1, cv2.LINE_AA)


# ── hand geometry ────────────────────────────────────────────────────────────
PALM_IDS = (0, 5, 9, 13, 17)        # wrist + finger MCP knuckles


def palm_center(landmarks):
    xs = sum(landmarks[i][0] for i in PALM_IDS) // len(PALM_IDS)
    ys = sum(landmarks[i][1] for i in PALM_IDS) // len(PALM_IDS)
    return xs, ys


def hand_size(landmarks):
    """Wrist-to-middle-knuckle distance — scales the Rasengan with depth."""
    dx = landmarks[9][0] - landmarks[0][0]
    dy = landmarks[9][1] - landmarks[0][1]
    return math.hypot(dx, dy)


def is_hand_raised(landmarks, frame_h):
    """Wrist landmark above RAISE_THRESHOLD of the frame height."""
    return landmarks[0][1] < frame_h * RAISE_THRESHOLD


# ── HUD ──────────────────────────────────────────────────────────────────────
def draw_idle_label(frame):
    cv2.putText(frame, "RAISE HAND TO TRANSFORM", (14, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def draw_naruto_label(frame):
    text = "NARUTO MODE"
    cv2.putText(frame, text, (14, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (14, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, ORANGE, 2, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    draw_flame_icon(frame, 14 + tw + 18, frame.shape[0] - 18, 16)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS {fps:5.1f}", (frame.shape[1] - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 180), 2, cv2.LINE_AA)


# ── main loop ────────────────────────────────────────────────────────────────
def apply_transformation(frame, face, assets):
    """Overlay jacket → hair → headband (back to front). PNG assets when
    available, procedural art otherwise."""
    x, y, w, h = face
    cx = x + w // 2
    if "jacket" in assets:
        overlay_rgba(frame, assets["jacket"], cx, y + int(h * 1.0), int(w * 3.0))
    else:
        draw_outfit(frame, face)
    if "hair" in assets:
        overlay_rgba(frame, assets["hair"], cx, y - int(h * 0.75), int(w * 1.6))
    else:
        draw_spiky_hair(frame, face)
    if "headband" in assets:
        overlay_rgba(frame, assets["headband"], cx, y + int(h * 0.14), int(w * 1.2))
    else:
        draw_headband(frame, face)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # never queue stale frames
    cap.set(cv2.CAP_PROP_FPS, 60)

    hand_detector = create_hand_detector()
    face_detector = create_face_detector()
    assets = load_assets()
    if assets:
        print(f"Loaded PNG assets: {', '.join(sorted(assets))}")
    else:
        print("No PNG assets found in ./assets — using procedural Naruto art.")

    rasengans = [Rasengan() for _ in range(MAX_HANDS)]
    cached_face = None
    frame_n = 0
    fps = 0.0
    tick_freq = cv2.getTickFrequency()
    last_tick = cv2.getTickCount()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, DISPLAY_W * 2, DISPLAY_H * 2)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_n += 1

        # clamp processing resolution to 640x480
        if frame.shape[1] > DISPLAY_W or frame.shape[0] > DISPLAY_H:
            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                               interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # one downscaled RGB copy feeds BOTH detectors
        infer_rgb = cv2.cvtColor(
            cv2.resize(frame, (INFER_W, INFER_H), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB)

        # hands every frame
        hands = detect_hands(hand_detector, infer_rgb, w, h)
        raised = [lm for lm in hands if is_hand_raised(lm, h)]

        # face every other frame, cached in between
        if frame_n % FACE_SKIP == 0 or cached_face is None:
            face = detect_face(face_detector, infer_rgb, w, h)
            if face is not None:
                cached_face = face

        transformed = bool(raised)
        if transformed:
            if cached_face is not None:
                apply_transformation(frame, cached_face, assets)
            for i, lm in enumerate(raised[:MAX_HANDS]):
                radius = max(28, int(hand_size(lm) * 0.60))
                rasengans[i].draw(frame, palm_center(lm), radius)
            draw_naruto_label(frame)
        else:
            draw_idle_label(frame)

        # FPS via tick counter (no sleeps anywhere in this loop)
        now = cv2.getTickCount()
        inst = tick_freq / max(now - last_tick, 1)
        last_tick = now
        fps = inst if fps == 0.0 else fps * 0.9 + inst * 0.1
        draw_fps(frame, fps)

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        window_open = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1
        if key in (ord("q"), ord("Q"), 27) or not window_open:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
