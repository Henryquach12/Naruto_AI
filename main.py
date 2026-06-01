import cv2
import numpy as np
import time
from hand_detector import HandDetector
from energy_ball import EnergyBall


WINDOW_TITLE   = "Naruto AI Vision — Rasengan"
BALL_BASE_SIZE = 0.55
FADE_FRAMES    = 20
MAX_HANDS      = 2


def blend(frame, layer, alpha):
    """Safe in-place blend: frame = alpha*layer + (1-alpha)*frame."""
    frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


def draw_hud(frame, fps, hand_count, active_count):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 44), (20, 20, 20), -1)
    blend(frame, bar, 0.55)

    cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hands: {hand_count}  Ball: {active_count}", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 80), 2, cv2.LINE_AA)

    tip = "Raise your hand!" if active_count == 0 else "RASENGAN!"
    cv2.putText(frame, tip, (w - 320, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector  = HandDetector(max_hands=MAX_HANDS)
    balls     = [EnergyBall() for _ in range(MAX_HANDS)]
    fade_cnt  = [0] * MAX_HANDS
    prev_time = time.time()
    frame_n   = 0

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        h, w   = frame.shape[:2]
        frame_n += 1

        _, hand_list = detector.find_hands(frame)
        detector.draw_landmarks(frame, hand_list)

        # ── print debug every 30 frames ──────────────────────────────────────
        if frame_n % 30 == 0:
            if hand_list:
                for idx, lm in enumerate(hand_list):
                    wrist_y = lm[0][1]
                    palm    = detector.get_palm_center(lm)
                    size    = detector.get_hand_size(lm)
                    print(f"[frame {frame_n}] hand{idx}: wrist_y={wrist_y}/{h}  "
                          f"palm={palm}  size={size:.1f}  "
                          f"radius={max(30, int(size * BALL_BASE_SIZE))}")
            else:
                print(f"[frame {frame_n}] NO HANDS DETECTED")

        # ── ball logic: show ball whenever hand is present ───────────────────
        active_count = 0
        for i in range(MAX_HANDS):
            if i < len(hand_list):
                fade_cnt[i] = 0
                landmarks   = hand_list[i]
                center      = detector.get_palm_center(landmarks)
                size        = detector.get_hand_size(landmarks)
                radius      = max(30, int(size * BALL_BASE_SIZE))
                balls[i].draw(frame, center, radius)
                active_count += 1
            else:
                # hand gone — fade out
                if fade_cnt[i] < FADE_FRAMES:
                    fade_cnt[i] += 1
                # nothing to draw once fully faded

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_hud(frame, fps, len(hand_list), active_count)

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
