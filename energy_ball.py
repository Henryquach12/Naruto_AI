import cv2
import numpy as np
import math


def _blend(frame, layer, alpha):
    """Safe in-place blend: frame = alpha*layer + (1-alpha)*frame."""
    frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


class EnergyBall:
    """Renders a Rasengan-style blue energy ball on a frame."""

    CORE_COLOR  = (255, 255, 255)
    INNER_COLOR = (255, 220,  80)
    MID_COLOR   = (255, 140,  20)
    OUTER_COLOR = (180,  60,   0)
    AURA_COLOR  = (120,  20,   0)
    SPARK_COLOR = (255, 200, 100)
    RING_COLOR  = (255, 180,  60)

    def __init__(self):
        self.tick = 0
        self.spiral_particles = self._init_spiral_particles(80)
        self.spark_particles  = self._init_spark_particles(40)

    # ── particle init ─────────────────────────────────────────────────────────
    def _init_spiral_particles(self, n):
        rng = np.random.default_rng(42)
        return [
            {
                "angle":  float(2 * math.pi / n * i),
                "radius": float(rng.uniform(0.50, 0.92)),
                "speed":  float(rng.uniform(0.04, 0.10)),
                "layer":  int(rng.integers(0, 2)),
                "size":   int(rng.integers(2, 5)),
            }
            for i in range(n)
        ]

    def _init_spark_particles(self, n):
        rng = np.random.default_rng(7)
        return [
            {
                "angle":  float(rng.uniform(0, 2 * math.pi)),
                "length": float(rng.uniform(0.3, 0.8)),
                "speed":  float(rng.uniform(0.06, 0.15)),
                "offset": float(rng.uniform(0, 2 * math.pi)),
            }
            for _ in range(n)
        ]

    # ── drawing helpers ───────────────────────────────────────────────────────
    def _draw_glow(self, frame, cx, cy, radius, color, alpha, scale):
        layer = frame.copy()
        cv2.circle(layer, (cx, cy), max(1, int(radius * scale)), color, -1)
        _blend(frame, layer, min(alpha, 0.95))

    def _draw_shockwave_rings(self, frame, cx, cy, radius, power):
        """Expanding energy rings — only visible when power > 1."""
        n_rings = min(int(power), 4)
        for i in range(n_rings):
            phase     = self.tick * 0.08 + i * 0.9
            ring_r    = int(radius * (1.6 + 0.35 * i + 0.12 * math.sin(phase)))
            thickness = max(1, 4 - i)
            alpha     = max(0.05, 0.45 - i * 0.10)
            ring_layer = frame.copy()
            cv2.circle(ring_layer, (cx, cy), ring_r,
                       self.INNER_COLOR, thickness, cv2.LINE_AA)
            _blend(frame, ring_layer, alpha)

    def _draw_spiral_particles(self, frame, cx, cy, radius, speed_mult):
        for p in self.spiral_particles:
            p["angle"] += p["speed"] * speed_mult
            arm_offset = math.pi if p["layer"] == 1 else 0.0
            angle  = p["angle"] + arm_offset
            wobble = 0.04 * math.sin(self.tick * 0.05 + p["angle"] * 3)
            r = int((p["radius"] + wobble) * radius)
            px = cx + int(r * math.cos(angle))
            py = cy + int(r * math.sin(angle))
            b = int(200 + 55 * math.sin(p["angle"]))
            g = int(100 + 80 * math.cos(p["angle"] * 0.5))
            cv2.circle(frame, (px, py), p["size"],          (b, g, 50), -1)
            cv2.circle(frame, (px, py), max(1, p["size"]-1), (255, 240, 200), -1)

    def _draw_spark_streaks(self, frame, cx, cy, radius, speed_mult):
        for p in self.spark_particles:
            p["angle"] += p["speed"] * speed_mult
            base_r = int(radius * 0.60)
            tip_r  = int(radius * (0.95 + 0.18 * math.sin(self.tick * 0.07 + p["offset"])))
            x1 = cx + int(base_r * math.cos(p["angle"]))
            y1 = cy + int(base_r * math.sin(p["angle"]))
            x2 = cx + int(tip_r  * math.cos(p["angle"]))
            y2 = cy + int(tip_r  * math.sin(p["angle"]))
            cv2.line(frame, (x1, y1), (x2, y2), self.SPARK_COLOR,
                     max(1, int(1 + p["length"])), cv2.LINE_AA)

    def _draw_smoke_haze(self, frame, cx, cy, radius):
        """Diffuse misty fog that surrounds the ball before the ribbons."""
        for scale, alpha in [(3.0, 0.06), (2.4, 0.09), (1.8, 0.13)]:
            layer = frame.copy()
            cv2.circle(layer, (cx, cy), max(1, int(radius * scale)),
                       (210, 230, 255), -1)
            _blend(frame, layer, alpha)

    def _draw_smoke_ribbons(self, frame, cx, cy, radius, speed_mult):
        """
        Rotating white wind/chakra ribbons that envelop the ball,
        matching the Rasengan's outer smoke layer.
        Each ribbon is a semi-transparent rotated ellipse crossing the ball center.
        """
        # ── primary ribbons (long, bright, forward rotation) ──────────────────
        n_primary = 6
        for i in range(n_primary):
            ang_deg = (self.tick * 2.8 * speed_mult + 60.0 * i) % 360
            a       = max(1, int(radius * 1.55))   # semi-major: extends well beyond ball
            b       = max(1, int(radius * 0.15))   # semi-minor: thin ribbon

            # soft halo layer behind each ribbon
            layer = frame.copy()
            cv2.ellipse(layer, (cx, cy),
                        (max(1, int(a * 1.18)), max(1, int(b * 2.4))),
                        ang_deg, 0, 360, (200, 225, 255), -1, cv2.LINE_AA)
            _blend(frame, layer, 0.10)

            # bright ribbon core
            layer = frame.copy()
            cv2.ellipse(layer, (cx, cy), (a, b),
                        ang_deg, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
            _blend(frame, layer, 0.32)

        # ── secondary counter-rotating thinner ribbons ─────────────────────────
        n_secondary = 4
        for i in range(n_secondary):
            ang_deg = (-self.tick * 2.0 * speed_mult + 45.0 * i) % 360
            a       = max(1, int(radius * 1.05))
            b       = max(1, int(radius * 0.08))
            layer   = frame.copy()
            cv2.ellipse(layer, (cx, cy), (a, b),
                        ang_deg, 0, 360, (235, 245, 255), -1, cv2.LINE_AA)
            _blend(frame, layer, 0.20)

    def _draw_outer_ring(self, frame, cx, cy, radius, speed_mult):
        segments = 24
        rot = self.tick * 0.03 * speed_mult
        ring_layer = frame.copy()
        for i in range(segments):
            a1 = rot + (2 * math.pi / segments) * i
            a2 = a1 + (2 * math.pi / segments) * 0.55
            r_in  = int(radius * 1.05)
            r_out = int(radius * 1.30)
            pts = (
                [[cx + int(r_out * math.cos(a)), cy + int(r_out * math.sin(a))]
                 for a in np.linspace(a1, a2, 6)]
                + [[cx + int(r_in  * math.cos(a)), cy + int(r_in  * math.sin(a))]
                   for a in np.linspace(a2, a1, 6)]
            )
            cv2.fillPoly(ring_layer, [np.array(pts, dtype=np.int32)], self.RING_COLOR)
        ring_alpha = 0.55 + 0.20 * math.sin(self.tick * 0.04)
        _blend(frame, ring_layer, ring_alpha)

    # ── public API ────────────────────────────────────────────────────────────
    def draw(self, frame, center, radius, power=1.0):
        """
        Draw the energy ball.
        power=1.0  → normal single-hand ball
        power>1.0  → combined/growing ball: brighter, faster, shockwave rings
        """
        cx, cy     = int(center[0]), int(center[1])
        r          = int(radius)
        self.tick += 1

        speed_mult = 1.0 + (power - 1.0) * 0.5   # particles spin faster when combined
        glow_boost = min(power * 0.25, 0.35)       # extra glow alpha at high power

        # 1. shockwave rings (only when combined)
        if power > 1.05:
            self._draw_shockwave_rings(frame, cx, cy, r, power)

        # 2. aura + glow layers
        self._draw_glow(frame, cx, cy, r, self.AURA_COLOR,  0.20 + glow_boost, 2.4)
        self._draw_glow(frame, cx, cy, r, self.OUTER_COLOR, 0.35 + glow_boost, 1.9)
        self._draw_glow(frame, cx, cy, r, self.MID_COLOR,   0.40 + glow_boost, 1.55)
        self._draw_glow(frame, cx, cy, r, self.INNER_COLOR, 0.45 + glow_boost, 1.20)

        # 3. smoke haze (misty atmosphere behind everything)
        self._draw_smoke_haze(frame, cx, cy, r)

        # 4. rotating outer ring
        self._draw_outer_ring(frame, cx, cy, r, speed_mult)

        # 5. spiral particles
        self._draw_spiral_particles(frame, cx, cy, r, speed_mult)

        # 6. spark streaks
        self._draw_spark_streaks(frame, cx, cy, r, speed_mult)

        # 7. main ball body
        ball_layer = frame.copy()
        cv2.circle(ball_layer, (cx, cy), r, self.MID_COLOR, -1)
        _blend(frame, ball_layer, 0.70)

        # 8. smoke ribbons — over the ball, enveloping it
        self._draw_smoke_ribbons(frame, cx, cy, r, speed_mult)

        # 9. inner ring + core (solid, punches through the smoke)
        cv2.circle(frame, (cx, cy), max(1, int(r * 0.70)), self.INNER_COLOR, -1)
        cv2.circle(frame, (cx, cy), max(1, int(r * 0.35)), self.CORE_COLOR,  -1)

        # 10. specular highlight
        cv2.circle(frame,
                   (cx - int(r * 0.22), cy - int(r * 0.22)),
                   max(1, int(r * 0.15)), (255, 255, 255), -1)

        # 11. edge rim
        cv2.circle(frame, (cx, cy), r, self.INNER_COLOR, 2, cv2.LINE_AA)

        return frame
