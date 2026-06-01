import cv2
import numpy as np
import time
from hand_detector import HandDetector
from energy_ball import EnergyBall


WINDOW_TITLE        = "Naruto AI Vision — Rasengan"
BALL_BASE_SIZE      = 0.55
MAX_HANDS           = 6      # MediaPipe handles up to ~6 reliably
COMBINE_DIST_FACTOR = 1.20   # combine when dist < factor * (s_i + s_j)
COMBINE_GROWTH_RATE = 1.8    # px/frame
COMBINE_MAX_RADIUS  = 240


def blend(frame, layer, alpha):
    frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


def greedy_pairs(centers, sizes):
    """
    Find non-overlapping pairs of hands that are close enough to combine.
    Returns (list of (i,j) pairs, list of unpaired indices).
    """
    n = len(centers)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            dist      = float(np.linalg.norm(np.array(centers[i]) - np.array(centers[j])))
            threshold = (sizes[i] + sizes[j]) * COMBINE_DIST_FACTOR
            if dist < threshold:
                candidates.append((dist, i, j))
    candidates.sort()

    used   = set()
    pairs  = []
    for _, i, j in candidates:
        if i not in used and j not in used:
            pairs.append((i, j))
            used.add(i)
            used.add(j)

    unmatched = [i for i in range(n) if i not in used]
    return pairs, unmatched


def draw_hud(frame, fps, hand_count, msg):
    w   = frame.shape[1]
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 48), (20, 20, 20), -1)
    blend(frame, bar, 0.60)
    cv2.putText(frame, f"FPS {fps:.1f}  |  Hands: {hand_count}  |  {msg}",
                (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 180), 2, cv2.LINE_AA)


def draw_combine_beam(frame, c0, c1, progress):
    layer = frame.copy()
    cv2.line(layer, c0, c1, (255, 220, 80), 8, cv2.LINE_AA)
    cv2.line(layer, c0, c1, (255, 255, 255), 2, cv2.LINE_AA)
    blend(frame, layer, 0.5 + 0.4 * progress)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,   960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  540)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,      1)   # minimise capture latency

    detector   = HandDetector(max_hands=MAX_HANDS)
    hand_balls = [EnergyBall() for _ in range(MAX_HANDS)]

    # Combined state: keyed by frozenset{i, j}
    # Each entry: {"ball": EnergyBall, "radius": float}
    combined_states: dict = {}

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

        # Sort left-to-right for stable per-frame indexing
        hand_list.sort(key=lambda lm: detector.get_palm_center(lm)[0])

        detector.draw_landmarks(frame, hand_list)
        n = len(hand_list)

        # ── console debug every 30 frames ─────────────────────────────────────
        if frame_n % 30 == 0:
            if hand_list:
                for idx, lm in enumerate(hand_list):
                    c  = detector.get_palm_center(lm)
                    sz = detector.get_hand_size(lm)
                    print(f"[f{frame_n}] hand{idx} palm={c} size={sz:.1f}")
            else:
                print(f"[f{frame_n}] NO hands detected")

        msg = "Raise your hand!"

        if n > 0:
            centers = [detector.get_palm_center(lm) for lm in hand_list]
            sizes   = [detector.get_hand_size(lm)   for lm in hand_list]

            pairs, unmatched = greedy_pairs(centers, sizes)

            # Expire combined states whose pair no longer exists
            active_keys = {frozenset(p) for p in pairs}
            for key in list(combined_states):
                if key not in active_keys:
                    del combined_states[key]

            # ── draw combined balls ────────────────────────────────────────────
            for i, j in pairs:
                key = frozenset({i, j})
                if key not in combined_states:
                    r0 = max(30, int(sizes[i] * BALL_BASE_SIZE))
                    r1 = max(30, int(sizes[j] * BALL_BASE_SIZE))
                    combined_states[key] = {
                        "ball":   EnergyBall(),
                        "radius": float((r0 + r1) / 2 * 1.10),
                    }

                state           = combined_states[key]
                state["radius"] = min(state["radius"] + COMBINE_GROWTH_RATE,
                                      COMBINE_MAX_RADIUS)

                power    = 1.0 + (state["radius"] - 60) / 80.0
                cx       = (centers[i][0] + centers[j][0]) // 2
                cy       = (centers[i][1] + centers[j][1]) // 2
                progress = min(1.0, (state["radius"] - 60) / 120.0)

                draw_combine_beam(frame, centers[i], centers[j], progress)
                state["ball"].draw(frame, (cx, cy), int(state["radius"]), power=power)

            # ── draw individual balls for unmatched hands ──────────────────────
            for i in unmatched:
                radius = max(30, int(sizes[i] * BALL_BASE_SIZE))
                hand_balls[i % MAX_HANDS].draw(frame, centers[i], radius, power=1.0)

            # ── status message ─────────────────────────────────────────────────
            parts = []
            if pairs:
                parts.append(f"COMBINED x{len(pairs)}")
            if unmatched:
                parts.append(f"RASENGAN x{len(unmatched)}")
            msg = "  +  ".join(parts) if parts else "RASENGAN!"

        else:
            combined_states.clear()

        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_hud(frame, fps, n, msg)
        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        window_open = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1
        if not window_open or key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
