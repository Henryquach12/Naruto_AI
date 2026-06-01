import cv2
import numpy as np
import time
from hand_detector import HandDetector
from energy_ball import EnergyBall


WINDOW_TITLE        = "Naruto AI Vision — Rasengan"
BALL_BASE_SIZE      = 0.55
FADE_FRAMES         = 20
MAX_HANDS           = 2
COMBINE_DIST_FACTOR = 1.20   # combine when palm-dist < factor*(s0+s1)
COMBINE_GROWTH_RATE = 1.8    # px/frame growth
COMBINE_MAX_RADIUS  = 240


def blend(frame, layer, alpha):
    frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


def draw_hud(frame, fps, hand_count, msg):
    h, w = frame.shape[:2]
    bar  = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 48), (20, 20, 20), -1)
    blend(frame, bar, 0.60)
    cv2.putText(frame, f"FPS {fps:.1f}  |  Hands: {hand_count}  |  {msg}",
                (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 180), 2, cv2.LINE_AA)


def draw_combine_beam(frame, c0, c1, progress):
    """Glowing beam between palms; progress 0-1 controls brightness."""
    layer = frame.copy()
    cv2.line(layer, c0, c1, (255, 220, 80), 8, cv2.LINE_AA)
    cv2.line(layer, c0, c1, (255, 255, 255), 2, cv2.LINE_AA)
    blend(frame, layer, 0.5 + 0.4 * progress)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(max_hands=MAX_HANDS)

    hand_balls      = [EnergyBall() for _ in range(MAX_HANDS)]
    combined_ball   = EnergyBall()
    combined        = False
    combined_radius = 0.0

    prev_time = time.time()
    frame_n   = 0

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        frame_n += 1

        _, hand_list = detector.find_hands(frame)
        detector.draw_landmarks(frame, hand_list)

        n = len(hand_list)

        # ── console debug every 30 frames ─────────────────────────────────────
        if frame_n % 30 == 0:
            if hand_list:
                for idx, lm in enumerate(hand_list):
                    c  = detector.get_palm_center(lm)
                    sz = detector.get_hand_size(lm)
                    print(f"[f{frame_n}] hand{idx}  palm={c}  size={sz:.1f}  "
                          f"r={max(30, int(sz * BALL_BASE_SIZE))}")
            else:
                print(f"[f{frame_n}] NO hands detected")

        msg = ""

        # ── two hands ─────────────────────────────────────────────────────────
        if n >= 2:
            lm0, lm1  = hand_list[0], hand_list[1]
            c0        = detector.get_palm_center(lm0)
            c1        = detector.get_palm_center(lm1)
            s0        = detector.get_hand_size(lm0)
            s1        = detector.get_hand_size(lm1)
            dist      = float(np.linalg.norm(np.array(c0) - np.array(c1)))
            threshold = (s0 + s1) * COMBINE_DIST_FACTOR

            if dist < threshold:
                # ── COMBINED ──────────────────────────────────────────────────
                if not combined:
                    r0 = max(30, int(s0 * BALL_BASE_SIZE))
                    r1 = max(30, int(s1 * BALL_BASE_SIZE))
                    combined_radius = float((r0 + r1) / 2 * 1.10)
                    combined = True

                combined_radius = min(combined_radius + COMBINE_GROWTH_RATE,
                                      COMBINE_MAX_RADIUS)
                power  = 1.0 + (combined_radius - 60) / 80.0
                cx     = (c0[0] + c1[0]) // 2
                cy     = (c0[1] + c1[1]) // 2
                progress = min(1.0, (combined_radius - 60) / 120.0)

                draw_combine_beam(frame, c0, c1, progress)
                combined_ball.draw(frame, (cx, cy), int(combined_radius), power=power)
                msg = f"COMBINED!  r={int(combined_radius)}  pwr={power:.1f}"
            else:
                # ── SEPARATE — two independent balls ──────────────────────────
                combined        = False
                combined_radius = 0.0
                for i in range(2):
                    lm     = hand_list[i]
                    center = detector.get_palm_center(lm)
                    size   = detector.get_hand_size(lm)
                    radius = max(30, int(size * BALL_BASE_SIZE))
                    hand_balls[i].draw(frame, center, radius, power=1.0)
                msg = f"2 hands  dist={int(dist)}  threshold={int(threshold)}  (bring closer!)"

        # ── one hand ──────────────────────────────────────────────────────────
        elif n == 1:
            combined        = False
            combined_radius = 0.0
            lm     = hand_list[0]
            center = detector.get_palm_center(lm)
            size   = detector.get_hand_size(lm)
            radius = max(30, int(size * BALL_BASE_SIZE))
            hand_balls[0].draw(frame, center, radius, power=1.0)
            msg = "RASENGAN!"

        # ── no hands ──────────────────────────────────────────────────────────
        else:
            combined        = False
            combined_radius = 0.0
            msg = "Raise your hand!"

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_hud(frame, fps, n, msg)
        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
