import cv2
import numpy as np
import math


def _blend(frame, layer, alpha):
    frame[:] = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)


class EnergyBall:
    CORE_COLOR  = (255, 255, 255)
    INNER_COLOR = (255, 220,  80)
    MID_COLOR   = (255, 140,  20)
    OUTER_COLOR = (180,  60,   0)
    AURA_COLOR  = (120,  20,   0)
    SPARK_COLOR = (255, 200, 100)
    RING_COLOR  = (255, 180,  60)

    def __init__(self):
        self.tick = 0
        self._buf  = None   # pre-allocated blend buffer (reused every frame)
        self.spiral_particles = self._init_spiral_particles(50)   # 80 → 50
        self.spark_particles  = self._init_spark_particles(25)    # 40 → 25

    # ── buffer helper ─────────────────────────────────────────────────────────
    def _fresh(self, frame):
        """Return a pre-allocated copy of frame, avoiding per-call malloc."""
        if self._buf is None or self._buf.shape != frame.shape:
            self._buf = np.empty_like(frame)
        np.copyto(self._buf, frame)
        return self._buf

    # ── particle init ─────────────────────────────────────────────────────────
    def _init_spiral_particles(self, n):
        rng = np.random.default_rng(42)
        return [{"angle":  float(2 * math.pi / n * i),
                 "radius": float(rng.uniform(0.50, 0.92)),
                 "speed":  float(rng.uniform(0.04, 0.10)),
                 "layer":  int(rng.integers(0, 2)),
                 "size":   int(rng.integers(2, 5))}
                for i in range(n)]

    def _init_spark_particles(self, n):
        rng = np.random.default_rng(7)
        return [{"angle":  float(rng.uniform(0, 2 * math.pi)),
                 "length": float(rng.uniform(0.3, 0.8)),
                 "speed":  float(rng.uniform(0.06, 0.15)),
                 "offset": float(rng.uniform(0, 2 * math.pi))}
                for _ in range(n)]

    # ── drawing helpers ───────────────────────────────────────────────────────
    def _draw_glow_all(self, frame, cx, cy, r, glow_boost):
        """4 glow circles → 2 blends (was 4 blends)."""
        layer = self._fresh(frame)
        cv2.circle(layer, (cx, cy), max(1, int(r * 2.4)), self.AURA_COLOR,  -1)
        cv2.circle(layer, (cx, cy), max(1, int(r * 1.9)), self.OUTER_COLOR, -1)
        _blend(frame, layer, min(0.28 + glow_boost, 0.65))

        layer = self._fresh(frame)
        cv2.circle(layer, (cx, cy), max(1, int(r * 1.55)), self.MID_COLOR,   -1)
        cv2.circle(layer, (cx, cy), max(1, int(r * 1.20)), self.INNER_COLOR, -1)
        _blend(frame, layer, min(0.42 + glow_boost, 0.85))

    def _draw_shockwave_rings(self, frame, cx, cy, r, power):
        """All rings on ONE layer → 1 blend (was n_rings blends)."""
        layer   = self._fresh(frame)
        n_rings = min(int(power), 4)
        for i in range(n_rings):
            rr = int(r * (1.6 + 0.35 * i + 0.12 * math.sin(self.tick * 0.08 + i * 0.9)))
            cv2.circle(layer, (cx, cy), rr, self.INNER_COLOR, max(1, 4 - i), cv2.LINE_AA)
        _blend(frame, layer, 0.45)

    def _draw_smoke_haze(self, frame, cx, cy, r):
        """3 haze circles → 1 blend (was 3 blends)."""
        layer = self._fresh(frame)
        cv2.circle(layer, (cx, cy), max(1, int(r * 3.0)), (180, 210, 255), -1)
        cv2.circle(layer, (cx, cy), max(1, int(r * 2.3)), (200, 225, 255), -1)
        cv2.circle(layer, (cx, cy), max(1, int(r * 1.7)), (215, 235, 255), -1)
        _blend(frame, layer, 0.12)

    def _draw_outer_ring(self, frame, cx, cy, r, speed_mult):
        """All ring segments on ONE layer → 1 blend. Segments 24→18."""
        segments   = 18
        rot        = self.tick * 0.03 * speed_mult
        ring_layer = self._fresh(frame)
        r_in  = int(r * 1.05)
        r_out = int(r * 1.30)
        for i in range(segments):
            a1 = rot + (2 * math.pi / segments) * i
            a2 = a1 + (2 * math.pi / segments) * 0.55
            angs_out = np.linspace(a1, a2, 5)
            angs_in  = np.linspace(a2, a1, 5)
            pts = np.array(
                [[cx + int(r_out * math.cos(a)), cy + int(r_out * math.sin(a))]
                 for a in angs_out]
                + [[cx + int(r_in  * math.cos(a)), cy + int(r_in  * math.sin(a))]
                   for a in angs_in],
                dtype=np.int32)
            cv2.fillPoly(ring_layer, [pts], self.RING_COLOR)
        _blend(frame, ring_layer, 0.55 + 0.20 * math.sin(self.tick * 0.04))

    def _draw_spiral_particles(self, frame, cx, cy, r, speed_mult):
        for p in self.spiral_particles:
            p["angle"] += p["speed"] * speed_mult
            angle  = p["angle"] + (math.pi if p["layer"] == 1 else 0.0)
            wobble = 0.04 * math.sin(self.tick * 0.05 + p["angle"] * 3)
            pr = int((p["radius"] + wobble) * r)
            px = cx + int(pr * math.cos(angle))
            py = cy + int(pr * math.sin(angle))
            b  = int(200 + 55 * math.sin(p["angle"]))
            g  = int(100 + 80 * math.cos(p["angle"] * 0.5))
            cv2.circle(frame, (px, py), p["size"],           (b, g, 50),      -1)
            cv2.circle(frame, (px, py), max(1, p["size"]-1), (255, 240, 200), -1)

    def _draw_spark_streaks(self, frame, cx, cy, r, speed_mult):
        for p in self.spark_particles:
            p["angle"] += p["speed"] * speed_mult
            base_r = int(r * 0.60)
            tip_r  = int(r * (0.95 + 0.18 * math.sin(self.tick * 0.07 + p["offset"])))
            x1 = cx + int(base_r * math.cos(p["angle"]))
            y1 = cy + int(base_r * math.sin(p["angle"]))
            x2 = cx + int(tip_r  * math.cos(p["angle"]))
            y2 = cy + int(tip_r  * math.sin(p["angle"]))
            cv2.line(frame, (x1, y1), (x2, y2), self.SPARK_COLOR,
                     max(1, int(1 + p["length"])), cv2.LINE_AA)

    def _draw_smoke_ribbons(self, frame, cx, cy, r, speed_mult):
        """
        10 ribbon ellipses → 2 blends (was 20 blends).
        Pass 1: soft halos for all ribbons on one layer.
        Pass 2: bright white cores for all ribbons on one layer.
        """
        n_prim = 5   # primary ribbons  (was 6)
        n_sec  = 3   # counter-ribbons  (was 4)

        # pass 1 — halo backing
        halo = self._fresh(frame)
        for i in range(n_prim):
            ang = (self.tick * 2.8 * speed_mult + 72.0 * i) % 360
            a   = max(1, int(r * 1.55));  b = max(1, int(r * 0.15))
            cv2.ellipse(halo, (cx, cy),
                        (max(1, int(a * 1.18)), max(1, int(b * 2.4))),
                        ang, 0, 360, (200, 225, 255), -1, cv2.LINE_AA)
        for i in range(n_sec):
            ang = (-self.tick * 2.0 * speed_mult + 60.0 * i) % 360
            cv2.ellipse(halo, (cx, cy),
                        (max(1, int(r * 1.05)), max(1, int(r * 0.10))),
                        ang, 0, 360, (215, 235, 255), -1, cv2.LINE_AA)
        _blend(frame, halo, 0.11)

        # pass 2 — bright white cores
        cores = self._fresh(frame)        # fresh copy of frame after pass-1 blend
        for i in range(n_prim):
            ang = (self.tick * 2.8 * speed_mult + 72.0 * i) % 360
            a   = max(1, int(r * 1.55));  b = max(1, int(r * 0.15))
            cv2.ellipse(cores, (cx, cy), (a, b),
                        ang, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
        for i in range(n_sec):
            ang = (-self.tick * 2.0 * speed_mult + 60.0 * i) % 360
            cv2.ellipse(cores, (cx, cy),
                        (max(1, int(r * 1.05)), max(1, int(r * 0.09))),
                        ang, 0, 360, (240, 248, 255), -1, cv2.LINE_AA)
        _blend(frame, cores, 0.30)

    # ── public API ────────────────────────────────────────────────────────────
    def draw(self, frame, center, radius, power=1.0):
        cx, cy     = int(center[0]), int(center[1])
        r          = int(radius)
        self.tick += 1
        speed_mult = 1.0 + (power - 1.0) * 0.5
        glow_boost = min(power * 0.25, 0.35)

        # Total blends per call:
        #   shockwave 0-1 | glow 2 | haze 1 | ring 1 | ball 1 | ribbons 2
        #   = 7-8 blends  (was ~29)

        if power > 1.05:
            self._draw_shockwave_rings(frame, cx, cy, r, power)
        self._draw_glow_all(frame, cx, cy, r, glow_boost)
        self._draw_smoke_haze(frame, cx, cy, r)
        self._draw_outer_ring(frame, cx, cy, r, speed_mult)
        self._draw_spiral_particles(frame, cx, cy, r, speed_mult)
        self._draw_spark_streaks(frame, cx, cy, r, speed_mult)

        # main ball body
        layer = self._fresh(frame)
        cv2.circle(layer, (cx, cy), r, self.MID_COLOR, -1)
        _blend(frame, layer, 0.70)

        # smoke ribbons over the ball
        self._draw_smoke_ribbons(frame, cx, cy, r, speed_mult)

        # solid core punches through smoke
        cv2.circle(frame, (cx, cy), max(1, int(r * 0.70)), self.INNER_COLOR, -1)
        cv2.circle(frame, (cx, cy), max(1, int(r * 0.35)), self.CORE_COLOR,  -1)
        cv2.circle(frame,
                   (cx - int(r * 0.22), cy - int(r * 0.22)),
                   max(1, int(r * 0.15)), (255, 255, 255), -1)
        cv2.circle(frame, (cx, cy), r, self.INNER_COLOR, 2, cv2.LINE_AA)

        return frame
