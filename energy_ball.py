import cv2
import numpy as np
import math


class EnergyBall:
    """Renders a Rasengan-style blue energy ball on a frame."""

    # ── palette ───────────────────────────────────────────────────────────────
    CORE_COLOR    = (255, 255, 255)   # BGR white hot core
    INNER_COLOR   = (255, 220, 120)   # BGR cyan-white inner ring
    MID_COLOR     = (255, 160,  40)   # BGR blue-white mid
    OUTER_COLOR   = (200,  80,   0)   # BGR deep blue outer glow
    SPARK_COLOR   = (255, 230, 150)   # BGR spark streaks
    AURA_COLOR    = (160,  40,   0)   # BGR aura halo

    def __init__(self):
        self.tick = 0               # animation frame counter
        self.spiral_particles = self._init_spiral_particles(60)
        self.spark_particles   = self._init_spark_particles(30)

    # ── particle initialisation ───────────────────────────────────────────────
    def _init_spiral_particles(self, n):
        particles = []
        for i in range(n):
            particles.append({
                "angle":  (2 * math.pi / n) * i,
                "radius": np.random.uniform(0.55, 0.90),   # fraction of ball radius
                "speed":  np.random.uniform(0.04, 0.10),
                "layer":  np.random.choice([0, 1]),         # inner/outer spiral arm
                "size":   np.random.randint(2, 5),
            })
        return particles

    def _init_spark_particles(self, n):
        particles = []
        for i in range(n):
            angle = np.random.uniform(0, 2 * math.pi)
            particles.append({
                "angle":  angle,
                "length": np.random.uniform(0.3, 0.8),
                "speed":  np.random.uniform(0.06, 0.15),
                "offset": np.random.uniform(0, 2 * math.pi),
            })
        return particles

    # ── core drawing helpers ──────────────────────────────────────────────────
    def _draw_glow_circle(self, overlay, cx, cy, radius, color, alpha_center=0.9, alpha_edge=0.0, steps=8):
        """Draw a radial glow by stacking semi-transparent circles."""
        for s in range(steps, 0, -1):
            r = int(radius * s / steps)
            t = s / steps                          # 1 at edge → 0 at center
            a = alpha_center + (alpha_edge - alpha_center) * t
            a = max(0.0, min(1.0, a))
            layer = overlay.copy()
            cv2.circle(layer, (cx, cy), r, color, -1)
            cv2.addWeighted(layer, a / steps, overlay, 1 - a / steps, 0, overlay)

    def _draw_spiral_particles(self, frame, cx, cy, radius):
        for p in self.spiral_particles:
            p["angle"] += p["speed"]
            arm_offset = math.pi if p["layer"] == 1 else 0
            angle = p["angle"] + arm_offset

            # Whirlpool spiral inward offset
            wobble = 0.04 * math.sin(self.tick * 0.05 + p["angle"] * 3)
            r = int((p["radius"] + wobble) * radius)

            px = cx + int(r * math.cos(angle))
            py = cy + int(r * math.sin(angle))
            size = p["size"]

            # colour shifts with depth
            blue_val  = int(200 + 55 * math.sin(p["angle"]))
            green_val = int(100 + 80 * math.cos(p["angle"] * 0.5))
            color = (blue_val, green_val, 50)

            cv2.circle(frame, (px, py), size, color, -1)
            # bright highlight dot
            cv2.circle(frame, (px, py), max(1, size - 1), (255, 240, 200), -1)

    def _draw_spark_streaks(self, frame, cx, cy, radius):
        for p in self.spark_particles:
            p["angle"] += p["speed"]
            base_r = int(radius * 0.6)
            tip_r  = int(radius * (0.95 + 0.15 * math.sin(self.tick * 0.07 + p["offset"])))

            x1 = cx + int(base_r * math.cos(p["angle"]))
            y1 = cy + int(base_r * math.sin(p["angle"]))
            x2 = cx + int(tip_r  * math.cos(p["angle"]))
            y2 = cy + int(tip_r  * math.sin(p["angle"]))

            thickness = max(1, int(1 + p["length"]))
            cv2.line(frame, (x1, y1), (x2, y2), self.SPARK_COLOR, thickness, cv2.LINE_AA)

    def _draw_outer_ring(self, frame, cx, cy, radius):
        """Rotating segmented outer ring."""
        segments = 24
        rot = self.tick * 0.03
        for i in range(segments):
            a1 = rot + (2 * math.pi / segments) * i
            a2 = a1 + (2 * math.pi / segments) * 0.55
            r_inner = int(radius * 1.05)
            r_outer = int(radius * 1.25)
            pts = []
            for ang in np.linspace(a1, a2, 6):
                pts.append([cx + int(r_outer * math.cos(ang)),
                             cy + int(r_outer * math.sin(ang))])
            for ang in np.linspace(a2, a1, 6):
                pts.append([cx + int(r_inner * math.cos(ang)),
                             cy + int(r_inner * math.sin(ang))])
            pts = np.array(pts, dtype=np.int32)
            alpha = 0.5 + 0.3 * math.sin(self.tick * 0.04 + i)
            layer = frame.copy()
            cv2.fillPoly(layer, [pts], (255, 180, 60))
            cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, frame)

    # ── public API ────────────────────────────────────────────────────────────
    def draw(self, frame, center, radius):
        """
        Composite the energy ball onto `frame` at `center` with `radius` pixels.
        Modifies frame in place.
        """
        cx, cy = int(center[0]), int(center[1])
        r      = int(radius)
        self.tick += 1

        # ── 1. wide soft aura ──
        aura_layer = frame.copy()
        cv2.circle(aura_layer, (cx, cy), int(r * 2.2), self.AURA_COLOR, -1)
        cv2.addWeighted(aura_layer, 0.18, frame, 0.82, 0, frame)

        # ── 2. outer glow ──
        glow = frame.copy()
        for step, (col, alpha, scale) in enumerate([
            (self.OUTER_COLOR, 0.30, 1.9),
            (self.MID_COLOR,   0.35, 1.55),
            (self.INNER_COLOR, 0.40, 1.20),
        ]):
            cv2.circle(glow, (cx, cy), int(r * scale), col, -1)
            cv2.addWeighted(glow, alpha, frame, 1 - alpha, 0, frame)
            glow = frame.copy()

        # ── 3. rotating outer ring ──
        self._draw_outer_ring(frame, cx, cy, r)

        # ── 4. spiral particles ──
        self._draw_spiral_particles(frame, cx, cy, r)

        # ── 5. spark streaks ──
        self._draw_spark_streaks(frame, cx, cy, r)

        # ── 6. main filled ball ──
        ball_layer = frame.copy()
        cv2.circle(ball_layer, (cx, cy), r, self.MID_COLOR, -1)
        cv2.addWeighted(ball_layer, 0.65, frame, 0.35, 0, frame)

        # ── 7. inner bright ring ──
        cv2.circle(frame, (cx, cy), int(r * 0.7), self.INNER_COLOR, -1)

        # ── 8. hot white core ──
        cv2.circle(frame, (cx, cy), int(r * 0.35), self.CORE_COLOR, -1)

        # ── 9. specular highlight ──
        hx = cx - int(r * 0.22)
        hy = cy - int(r * 0.22)
        cv2.circle(frame, (hx, hy), int(r * 0.15), (255, 255, 255), -1)

        # ── 10. edge rim ──
        cv2.circle(frame, (cx, cy), r, self.INNER_COLOR, 2, cv2.LINE_AA)

        return frame
