"""Generate a realistic chest-radiograph-like background.

Simulated structures:
  - Thoracic cage outline (soft tissue gradient)
  - Bilateral lung fields (darker, air-filled)
  - Heart silhouette (pear-shape, central + slightly left, soft tissue density)
  - Mediastinum (spine + central vessels — bright vertical column)
  - Diaphragm (curved bright lower boundary)
  - Posterior ribs (arched across thorax, brighter)
  - Anterior ribs (downward-angled overlap with posteriors)
  - Bilateral hilar regions (denser, with branching vasculature)
  - Vascular tree throughout each lung field (branching from hilum)
  - Soft tissue shoulders / breast shadows
  - Photon noise + low-frequency beam inhomogeneity
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from data.primitives.drawing import (
    add_low_freq_inhomogeneity, add_photon_noise, aa_line, bezier_curve,
    soft_ellipse, vascular_tree,
)


IMG_SIZE = 320


@dataclass
class LungGeometry:
    left_cx: float
    left_cy: float
    left_rx: float
    left_ry: float
    right_cx: float
    right_cy: float
    right_rx: float
    right_ry: float
    heart_cx: float
    heart_cy: float
    heart_rx: float
    heart_ry: float
    diaphragm_y: float

    def field(self, side: str) -> tuple[float, float, float, float]:
        if side == "left":
            return self.left_cx, self.left_cy, self.left_rx, self.left_ry
        return self.right_cx, self.right_cy, self.right_rx, self.right_ry

    def both(self) -> list[tuple[float, float, float, float]]:
        return [
            (self.left_cx, self.left_cy, self.left_rx, self.left_ry),
            (self.right_cx, self.right_cy, self.right_rx, self.right_ry),
        ]


def make_background(rng: np.random.Generator) -> tuple[np.ndarray, LungGeometry]:
    h = w = IMG_SIZE
    img = np.full((h, w), 65.0, dtype=np.float32)

    # 1. Body soft tissue: bright outer envelope (thoracic outline)
    body_cx = w / 2 + rng.uniform(-3, 3)
    body_cy = h / 2 + rng.uniform(-3, 3)
    body_rx = w * 0.48 + rng.uniform(-2, 2)
    body_ry = h * 0.48 + rng.uniform(-2, 2)
    # Two passes: a wide soft halo and an inner brighter tissue layer to
    # avoid the hard ring artifact a single ellipse + noise produces.
    soft_ellipse(img, body_cx, body_cy, body_rx + 8, body_ry + 8,
                 intensity=20.0, edge_softness=0.45)
    soft_ellipse(img, body_cx, body_cy, body_rx, body_ry,
                 intensity=30.0, edge_softness=0.30)

    # Shoulders: slight darker rounded soft tissue at top
    soft_ellipse(img, w * 0.16 + rng.uniform(-3, 3), h * 0.18,
                 w * 0.22, h * 0.20, intensity=18.0, edge_softness=0.40)
    soft_ellipse(img, w * 0.84 + rng.uniform(-3, 3), h * 0.18,
                 w * 0.22, h * 0.20, intensity=18.0, edge_softness=0.40)

    # 2. Lung fields (carve out — darker because air-filled)
    lung_off = rng.uniform(-2, 2)
    lung_rx = rng.uniform(54, 60)
    lung_ry = rng.uniform(85, 92)
    geom = LungGeometry(
        left_cx=w * 0.32 + lung_off, left_cy=h * 0.46,
        left_rx=lung_rx, left_ry=lung_ry,
        right_cx=w * 0.68 - lung_off, right_cy=h * 0.46,
        right_rx=lung_rx, right_ry=lung_ry,
        heart_cx=w * 0.51, heart_cy=h * 0.61,
        heart_rx=42.0, heart_ry=55.0,
        diaphragm_y=h * 0.78,
    )
    for cx, cy, rx, ry in geom.both():
        soft_ellipse(img, cx, cy, rx, ry, intensity=-50.0, edge_softness=0.16)

    # 3. Vascular tree from each hilum (brighter than surrounding lung)
    for side, cx, cy, rx, ry in [
        ("left", geom.left_cx, geom.left_cy, geom.left_rx, geom.left_ry),
        ("right", geom.right_cx, geom.right_cy, geom.right_rx, geom.right_ry),
    ]:
        # Hilar bright center
        hilum_x = cx + (8 if side == "left" else -8)
        hilum_y = cy
        soft_ellipse(img, hilum_x, hilum_y, 12.0, 10.0, intensity=22.0, edge_softness=0.4)

        # 4-6 main branches outward
        n_main = int(rng.integers(4, 7))
        for k in range(n_main):
            base_angle = (math.pi * (k + 0.5) / n_main) + rng.uniform(-0.2, 0.2)
            if side == "left":
                base_angle = math.pi - base_angle  # mirror
            length = rng.uniform(20, 38)
            thickness = rng.uniform(1.4, 2.2)
            intensity = rng.uniform(11.0, 18.0)
            vascular_tree(img, hilum_x, hilum_y, base_angle, length,
                          thickness, intensity, depth=4, rng=rng,
                          branch_prob=0.65, length_decay=0.62)

    # 4. Heart silhouette: distinctly brighter (soft tissue density blocks
    # the lung field translucency) — must be unmistakeable for cardiomegaly
    # to be visually meaningful.
    soft_ellipse(img, geom.heart_cx, geom.heart_cy, geom.heart_rx, geom.heart_ry,
                 intensity=48.0, edge_softness=0.16)
    # Cardiac contour bulges (left ventricle slightly inferior-lateral)
    soft_ellipse(img, geom.heart_cx + 14, geom.heart_cy + 16, 18, 22,
                 intensity=18.0, edge_softness=0.20)
    # Right atrial border — small inferior-right bulge
    soft_ellipse(img, geom.heart_cx - 18, geom.heart_cy + 8, 14, 18,
                 intensity=12.0, edge_softness=0.22)

    # 5. Mediastinum: bright central vertical column
    soft_ellipse(img, w / 2, h * 0.45, 14, 100,
                 intensity=24.0, edge_softness=0.28)
    # Aortic knob (bright bump upper left of mediastinum)
    soft_ellipse(img, w * 0.55, h * 0.32, 9, 7,
                 intensity=18.0, edge_softness=0.30)
    # Vertebral body markings (vertical bright dotted column)
    for i in range(11):
        ycoord = h * 0.18 + i * (h * 0.6 / 10)
        soft_ellipse(img, w / 2, ycoord, 6, 4, intensity=4.0, edge_softness=0.30)

    # 6. Diaphragm: curved bright lower boundary (right hemidiaphragm slightly higher)
    for side_off, side_factor in [(-1, 0.0), (1, -3.0)]:
        cx_d = w / 2 + side_off * w * 0.22
        cy_d = geom.diaphragm_y + side_factor
        n = 60
        xs = np.linspace(cx_d - 75, cx_d + 75, n)
        ys = cy_d - 12 * np.sin(np.linspace(0, math.pi, n))
        for i in range(n - 1):
            aa_line(img, xs[i], ys[i], xs[i + 1], ys[i + 1],
                    intensity=18.0, thickness=1.6, taper=False)
    # Subdiaphragmatic bright band (abdomen)
    soft_ellipse(img, w / 2, h - 20, w * 0.45, 30,
                 intensity=22.0, edge_softness=0.30)

    # 7. Posterior ribs: arching curves across thorax (clearly visible)
    n_ribs = int(rng.integers(8, 11))
    for k in range(n_ribs):
        t = (k + 0.5) / n_ribs
        y0 = h * 0.20 + t * (h * 0.58)
        amp = 18 + 8 * (1 - abs(t - 0.5) * 2)
        # Left rib (curve from outside-down-to-medial)
        x_outer_l = w * 0.05
        x_inner_l = w * 0.46
        x_mid_l = (x_outer_l + x_inner_l) / 2
        y_top_l = y0 - amp
        bezier_curve(img, (x_outer_l, y0 + 4), (x_mid_l, y_top_l),
                     (x_inner_l, y0 - 6), intensity=rng.uniform(15.0, 19.0),
                     thickness=1.6, taper=False)
        # Right rib (mirror)
        x_outer_r = w * 0.95
        x_inner_r = w * 0.54
        x_mid_r = (x_outer_r + x_inner_r) / 2
        bezier_curve(img, (x_outer_r, y0 + 4), (x_mid_r, y_top_l),
                     (x_inner_r, y0 - 6), intensity=rng.uniform(15.0, 19.0),
                     thickness=1.6, taper=False)

    # 8. Anterior ribs: downward-angled, fainter (overlap with posteriors)
    n_ant = int(rng.integers(6, 8))
    for k in range(n_ant):
        t = (k + 0.5) / n_ant
        y0 = h * 0.30 + t * (h * 0.42)
        bezier_curve(img, (w * 0.20, y0 - 6),
                     (w * 0.32, y0 + 4), (w * 0.46, y0 + 12),
                     intensity=rng.uniform(8.0, 11.0), thickness=1.1, taper=True)
        bezier_curve(img, (w * 0.80, y0 - 6),
                     (w * 0.68, y0 + 4), (w * 0.54, y0 + 12),
                     intensity=rng.uniform(8.0, 11.0), thickness=1.1, taper=True)

    # 9. Clavicles: bright sloped curves at top
    bezier_curve(img, (w * 0.10, h * 0.20), (w * 0.30, h * 0.13),
                 (w * 0.50, h * 0.17), intensity=22.0, thickness=1.8, taper=False)
    bezier_curve(img, (w * 0.50, h * 0.17), (w * 0.70, h * 0.13),
                 (w * 0.90, h * 0.20), intensity=22.0, thickness=1.8, taper=False)

    # 10. Low-frequency inhomogeneity (beam intensity variation)
    add_low_freq_inhomogeneity(img, amplitude=4.0, scale=80, rng=rng)

    # 11. Photon noise (Poisson-like)
    add_photon_noise(img, baseline=80.0, noise_scale=0.4, rng=rng)

    return img, geom
