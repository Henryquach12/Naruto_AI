import cv2
import numpy as np
import time
from hand_detector import HandDetector
from energy_ball import EnergyBall


WINDOW_TITLE        = "Naruto AI Vision — Rasengan"
BALL_BASE_SIZE      = 0.55   # individual ball radius = hand_size * this
FADE_FRAMES         = 20     # frames to fade out after hand disappears
MAX_HANDS           = 2

# ── combine settings ─────────────────────────────────────────────────────────
COMBINE_DIST_FACTOR = 0.80   # combine when palm-dist < factor*(s0+s1)
COMBINE_GROWTH_RATE = 1.8    # pixels per frame the combined ball grows
COMBINE_MAX_RADIUS  = 240    # hard cap on combined ball size
COMBINE_START_MULT  = 1.10   # combined ball starts at this * avg-individual-radius


def blend(frame, layer, alpha):
    frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


def draw_hud(frame, fps, hand_count, state_label):
    h, w = frame.shape[:2]
    bar  = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 44), (20, 20, 20), -1)
    blend(frame, bar, 0.55)

    cv2.putText(frame, f"FPS {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hands: {hand_count}",
                (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, state_label,
                (w - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2, cv2.LINE_AA)


def draw_combine_beam(frame, c0, c1):
    """Draw a glowing beam connecting the two palms during combination."""
    layer = frame.copy()
    cv2.line(layer, c0, c1, (255, 220, 80), 6, cv2.LINE_AA)
    cv2.line(layer, c0, c1, (255, 255, 255), 2, cv2.LINE_AA)
    blend(frame, layer, 0.60)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(max_hands=MAX_HANDS)

    # Per-hand balls + fade counters
    hand_balls  = [EnergyBall() for _ in range(MAX_HANDS)]
    fade_cnt    = [0] * MAX_HANDS

    # Combined ball state
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
        h, w    = frame.shape[:2]
        frame_n += 1

        _, hand_list = detector.find_hands(frame)
        detector.draw_landmarks(frame, hand_list)

        state_label = "Raise your hand!"

        # ── debug console every 30 frames ─────────────────────────────────────
        if frame_n % 30 == 0:
            if hand_list:
                for idx, lm in enumerate(hand_list):
                    wrist_y = lm[0][1]
                    palm    = detector.get_palm_center(lm)
                    size    = detector.get_hand_size(lm)
                    print(f"[f{frame_n}] hand{idx}: wrist_y={wrist_y}/{h} "
                          f"palm={palm}  size={size:.1f}  "
                          f"r={max(30, int(size * BALL_BASE_SIZE))}")
            else:
                print(f"[f{frame_n}] no hands detected")

        # ── two-hand logic ─────────────────────────────────────────────────────
        if len(hand_list) == 2:
            lm0, lm1  = hand_list[0], hand_list[1]
            c0        = detector.get_palm_center(lm0)
            c1        = detector.get_palm_center(lm1)
            s0        = detector.get_hand_size(lm0)
            s1        = detector.get_hand_size(lm1)
            dist      = float(np.linalg.norm(np.array(c0) - np.array(c1)))
            threshold = (s0 + s1) * COMBINE_DIST_FACTOR

            if dist < threshold:
                # ── COMBINED mode ──────────────────────────────────────────────
                if not combined:
                    r0 = max(30, int(s0 * BALL_BASE_SIZE))
                    r1 = max(30, int(s1 * BALL_BASE_SIZE))
                    combined_radius = ((r0 + r1) / 2) * COMBINE_START_MULT
                    combined        = True

                combined_radius = min(combined_radius + COMBINE_GROWTH_RATE,
                                      COMBINE_MAX_RADIUS)

                cx = (c0[0] + c1[0]) // 2
                cy = (c0[1] + c1[1]) // 2

                # power increases as ball grows
                power = 1.0 + (combined_radius - 60) / 80.0

                draw_combine_beam(frame, c0, c1)
                combined_ball.draw(frame, (cx, cy), int(combined_radius), power=power)

                fade_cnt = [0, 0]
                state_label = f"COMBINED! r={int(combined_radius)} pwr={power:.1f}"

            else:
                # ── SEPARATE — two individual balls ───────────────────────────
                combined        = False
                combined_radius = 0.0

                for i, lm in enumerate(hand_list[:MAX_HANDS]):
                    fade_cnt[i] = 0
                    center = detector.get_palm_center(lm)
                    size   = detector.get_hand_size(lm)
                    radius = max(30, int(size * BALL_BASE_SIZE))
                    hand_balls[i].draw(frame, center, radius, power=1.0)

                state_label = "Bring hands together!"

        elif len(hand_list) == 1:
            # ── single hand ───────────────────────────────────────────────────
            combined        = False
            combined_radius = 0.0
            fade_cnt[1]     = 0

            lm     = hand_list[0]
            center = detector.get_palm_center(lm)
            size   = detector.get_hand_size(lm)
            radius = max(30, int(size * BALL_BASE_SIZE))
            hand_balls[0].draw(frame, center, radius, power=1.0)
            fade_cnt[0] = 0
            state_label = "RASENGAN!"

        else:
            # ── no hands ──────────────────────────────────────────────────────
            combined        = False
            combined_radius = 0.0

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_hud(frame, fps, len(hand_list), state_label)
        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
