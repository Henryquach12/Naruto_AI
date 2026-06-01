import cv2
import numpy as np
import time
from hand_detector import HandDetector
from energy_ball import EnergyBall


# ── constants ─────────────────────────────────────────────────────────────────
WINDOW_TITLE   = "Naruto AI Vision — Rasengan"
BALL_BASE_SIZE = 0.55   # ball radius as fraction of hand size
RAISE_FRAMES   = 3      # consecutive raised frames before ball appears
FADE_FRAMES    = 20     # frames to fade ball out after hand is lowered
RAISE_THRESH   = 0.80   # wrist must be above this fraction of frame height
MAX_HANDS      = 2


def draw_hud(frame, fps, active_balls, hand_count=0, raise_cnts=None):
    """Render FPS counter and status overlay."""
    h, w = frame.shape[:2]
    # semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:>5.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 180), 2, cv2.LINE_AA)
    status = f"Rasengan x{active_balls}" if active_balls else "Raise your hand!"
    cv2.putText(frame, status, (w - 300, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 80), 2, cv2.LINE_AA)

    # debug line: hand count + raise counters
    if raise_cnts is not None:
        dbg = f"Hands:{hand_count}  Raise:{raise_cnts}"
        cv2.putText(frame, dbg, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # watermark
    cv2.putText(frame, "NARUTO AI", (w // 2 - 70, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 120), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    detector = HandDetector(max_hands=MAX_HANDS)
    # one EnergyBall instance per possible hand so animations are independent
    balls     = [EnergyBall() for _ in range(MAX_HANDS)]
    # per-hand counters for hysteresis
    raise_cnt = [0] * MAX_HANDS
    fade_cnt  = [0] * MAX_HANDS
    active    = [False] * MAX_HANDS

    prev_time = time.time()

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)      # mirror so it feels natural
        h, w  = frame.shape[:2]

        frame, hand_list = detector.find_hands(frame)
        detector.draw_landmarks(frame, hand_list)

        active_count = 0

        for i in range(MAX_HANDS):
            if i < len(hand_list):
                landmarks = hand_list[i]
                raised    = detector.is_hand_raised(landmarks, h, RAISE_THRESH)

                if raised:
                    raise_cnt[i] = min(raise_cnt[i] + 1, RAISE_FRAMES)
                    fade_cnt[i]  = 0
                else:
                    raise_cnt[i] = max(raise_cnt[i] - 1, 0)
                    if active[i]:
                        fade_cnt[i] += 1

                if raise_cnt[i] >= RAISE_FRAMES:
                    active[i] = True
                if fade_cnt[i] >= FADE_FRAMES:
                    active[i]    = False
                    fade_cnt[i]  = 0

                if active[i]:
                    center = detector.get_palm_center(landmarks)
                    size   = detector.get_hand_size(landmarks)
                    radius = max(30, int(size * BALL_BASE_SIZE))

                    # fade-out alpha when hand is lowering
                    if fade_cnt[i] > 0:
                        alpha = 1.0 - (fade_cnt[i] / FADE_FRAMES)
                        overlay = frame.copy()
                        balls[i].draw(overlay, center, radius)
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    else:
                        balls[i].draw(frame, center, radius)

                    active_count += 1
            else:
                # hand disappeared entirely
                raise_cnt[i] = 0
                fade_cnt[i]  = 0
                active[i]    = False

        # ── FPS ──
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_hud(frame, fps, active_count, len(hand_list), raise_cnt[:])

        cv2.imshow(WINDOW_TITLE, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # q or Esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
