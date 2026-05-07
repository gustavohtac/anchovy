"""v3 renderer: produces the 320×320 image from a FindingState.

Each finding has its own draw_* function; the pipeline draws all findings
on top of a realistic background canvas and captures a soft segmentation
mask per finding by snapshotting the canvas before/after each draw call.

Returned tuple from render_state:
    (image_uint8, annotations, masks)
where ``masks`` maps finding-label -> float32 array in [0, 1] of shape
(IMG_SIZE, IMG_SIZE). Multiple instances of the same finding type
(e.g. several nodules) are merged into a single per-type mask.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from data.primitives.background import IMG_SIZE, LungGeometry, make_background
from data.primitives.drawing import (
    aa_dot, aa_line, air_bronchogram, bezier_curve, cyst_cluster,
    irregular_calcific, lobulated_blob, random_point_in_ellipse,
    reticular_network, soft_ellipse, spiculated_lesion,
)
from data.findings import FindingState, NoduleSpec


# ---------- Cardiac / Mediastinum ----------


def draw_cardiomegaly(img: np.ndarray, geom: LungGeometry, severity: int,
                       rng: np.random.Generator) -> None:
    rx_factor = {1: 1.25, 2: 1.45, 3: 1.7}[severity]
    ry_factor = {1: 1.20, 2: 1.40, 3: 1.6}[severity]
    rx = geom.heart_rx * rx_factor + rng.uniform(-2, 2)
    ry = geom.heart_ry * ry_factor + rng.uniform(-2, 2)
    cx = geom.heart_cx + rng.uniform(-2, 2)
    cy = geom.heart_cy + rng.uniform(-2, 2)
    soft_ellipse(img, cx, cy, rx, ry, intensity=22.0, edge_softness=0.16)
    soft_ellipse(img, cx + rx * 0.55, cy + ry * 0.30, rx * 0.45, ry * 0.40,
                 intensity=12.0, edge_softness=0.20)
    soft_ellipse(img, cx - rx * 0.50, cy - ry * 0.10, rx * 0.40, ry * 0.35,
                 intensity=10.0, edge_softness=0.20)


def draw_pulm_venous_redistribution(img: np.ndarray, geom: LungGeometry,
                                     rng: np.random.Generator) -> None:
    for cx, cy, rx, ry in geom.both():
        for _ in range(int(rng.integers(5, 9))):
            x, y = random_point_in_ellipse(cx, cy - ry * 0.45,
                                            rx * 0.7, ry * 0.20, rng)
            r = rng.uniform(2.5, 4.5)
            soft_ellipse(img, x, y, r, r * rng.uniform(0.7, 1.3),
                         intensity=rng.uniform(7, 12), edge_softness=0.35)


def draw_hilar_adenopathy(img: np.ndarray, geom: LungGeometry,
                           rng: np.random.Generator) -> None:
    for cx, cy, rx, ry in geom.both():
        hilum_x = cx + (10 if cx < IMG_SIZE / 2 else -10)
        hilum_y = cy
        lobulated_blob(img, hilum_x, hilum_y, mean_radius=14, intensity=24,
                       n_lobes=4, lobe_amp=0.35, rng=rng, edge_softness=0.20,
                       inner_heterogeneity=0.10)


# ---------- Pleura ----------


def _draw_meniscus(img, cx, ry_lung, base_y, width, height, rng):
    h, w = img.shape
    y0 = max(0, int(base_y - height - 2)); y1 = min(h, int(base_y) + 1)
    x0 = max(0, int(cx - width / 2 - 2)); x1 = min(w, int(cx + width / 2 + 2) + 1)
    if y0 >= y1 or x0 >= x1: return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    nx = (xx - cx) / max(width / 2, 1e-6)
    inside_x = np.abs(nx) <= 1.0
    curve_factor = 1.0 - 0.55 * (1.0 - nx ** 2)
    upper_y = base_y - height * curve_factor
    in_fluid = inside_x & (yy >= upper_y) & (yy <= base_y)
    depth = (yy - upper_y) / np.maximum(base_y - upper_y, 1e-6)
    depth = np.clip(depth, 0, 1)
    weight = in_fluid.astype(np.float32) * (0.6 + 0.4 * depth)
    img[y0:y1, x0:x1] += 50.0 * weight
    n = 60
    xs = np.linspace(cx - width / 2 + 6, cx + width / 2 - 6, n)
    for i in range(n - 1):
        t0 = (xs[i] - cx) / (width / 2); t1 = (xs[i + 1] - cx) / (width / 2)
        y_a = base_y - height * (1.0 - 0.55 * (1 - t0 ** 2))
        y_b = base_y - height * (1.0 - 0.55 * (1 - t1 ** 2))
        aa_line(img, xs[i], y_a, xs[i + 1], y_b, intensity=28.0,
                thickness=1.1, taper=False)


def draw_effusion(img: np.ndarray, geom: LungGeometry, side: str,
                   severity: int, rng: np.random.Generator) -> None:
    cx, cy, rx, ry = geom.field(side)
    base_y = cy + ry - 6
    width = rx * 1.6
    height = {1: rng.uniform(12, 18), 2: rng.uniform(20, 32),
              3: rng.uniform(32, 48)}[severity]
    _draw_meniscus(img, cx, ry, base_y, width, height, rng)


def draw_pneumothorax(img: np.ndarray, geom: LungGeometry, side: str,
                       severity: int, rng: np.random.Generator) -> None:
    cx, cy, rx, ry = geom.field(side)
    apex_x = cx + (rx * 0.4 * (-1 if side == "left" else 1))
    apex_y = cy - ry * 0.55
    extent = {1: rng.uniform(20, 30), 2: rng.uniform(30, 42),
              3: rng.uniform(42, 56)}[severity]
    angle_dir = math.pi if side == "left" else 0.0
    h, w = img.shape
    pad = int(extent + 4)
    y0 = max(0, int(apex_y - pad)); y1 = min(h, int(apex_y + pad) + 1)
    x0 = max(0, int(apex_x - pad)); x1 = min(w, int(apex_x + pad) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    r_out = extent
    r_in = extent * 0.85
    icx = apex_x + r_out * 0.4 * math.cos(angle_dir)
    icy = apex_y + r_out * 0.4 * math.sin(angle_dir)
    in_outer = ((xx - apex_x) ** 2 + (yy - apex_y) ** 2) <= r_out ** 2
    in_inner = ((xx - icx) ** 2 + (yy - icy) ** 2) <= r_in ** 2
    crescent = in_outer & ~in_inner
    img[y0:y1, x0:x1] += -22.0 * crescent.astype(np.float32)
    n = 40
    if side == "right":
        a_start, a_end = math.pi * 0.65, math.pi * 1.35
    else:
        a_start, a_end = -math.pi * 0.35, math.pi * 0.35
    bx_prev = by_prev = None
    for k in range(n):
        a = a_start + (a_end - a_start) * (k / (n - 1))
        bx = icx + r_in * math.cos(a); by = icy + r_in * math.sin(a)
        if bx_prev is not None:
            aa_line(img, bx_prev, by_prev, bx, by, intensity=22.0,
                    thickness=0.8, taper=False)
        bx_prev, by_prev = bx, by


def draw_pleural_thickening(img: np.ndarray, geom: LungGeometry, side: str,
                             rng: np.random.Generator) -> None:
    cx, cy, rx, ry = geom.field(side)
    n = 25
    for k in range(n):
        t = k / (n - 1)
        ang = math.pi * 0.5 + (t - 0.5) * math.pi * 0.6
        if side == "left":
            ang = math.pi - ang
        x = cx + rx * 0.95 * math.cos(ang)
        y = cy + ry * math.sin(ang)
        thickness = rng.uniform(2.5, 5.0)
        intensity = rng.uniform(14, 22)
        soft_ellipse(img, x, y, thickness * 1.2, thickness * 0.6,
                     intensity=intensity, edge_softness=0.40,
                     rotation=ang + math.pi / 2)


def draw_volume_loss(img: np.ndarray, geom: LungGeometry, side: str,
                      rng: np.random.Generator) -> None:
    sign = -1 if side == "left" else 1
    cx_shift = IMG_SIZE / 2 + sign * 8
    soft_ellipse(img, cx_shift, IMG_SIZE * 0.45, 18, 90,
                 intensity=15.0, edge_softness=0.30)
    cx, cy, rx, ry = geom.field(side)
    soft_ellipse(img, cx + sign * rx * 0.8, cy, rx * 0.25, ry * 0.7,
                 intensity=-10.0, edge_softness=0.4)


# ---------- Airspace ----------


def draw_consolidation(img: np.ndarray, geom: LungGeometry, severity: int,
                        side: str, zone: str, has_air_bronchogram: bool,
                        rng: np.random.Generator) -> None:
    cx, cy, rx, ry = geom.field(side if side in ("left", "right") else rng.choice(["left", "right"]))
    zone_bias = {"upper": -ry * 0.4, "middle": 0, "lower": ry * 0.4,
                 "diffuse": 0}[zone]
    bx = cx + rng.uniform(-rx * 0.3, rx * 0.3)
    by = cy + zone_bias + rng.uniform(-ry * 0.15, ry * 0.15)
    radius = {1: rng.uniform(18, 26), 2: rng.uniform(28, 40),
              3: rng.uniform(40, 55)}[severity]
    inten = {1: 28.0, 2: 38.0, 3: 50.0}[severity]
    lobulated_blob(img, bx, by, radius, intensity=inten, n_lobes=4,
                   lobe_amp=0.20, rng=rng, edge_softness=0.20,
                   inner_heterogeneity=0.12)
    if has_air_bronchogram:
        air_bronchogram(img, bx, by, radius,
                        n_branches=int(rng.integers(2, 4)),
                        intensity=-14.0, rng=rng)


# ---------- Interstitium ----------


def draw_reticular_pattern(img: np.ndarray, geom: LungGeometry, severity: int,
                            zone: str, rng: np.random.Generator) -> None:
    density = {1: 0.6, 2: 1.2, 3: 2.0}[severity]
    inten = {1: 11.0, 2: 16.0, 3: 22.0}[severity]
    for cx, cy, rx, ry in geom.both():
        h, w = img.shape
        mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.mgrid[:h, :w]
        y_bias = {"upper": -ry * 0.3, "middle": 0, "lower": ry * 0.3,
                  "diffuse": 0}[zone]
        y_factor = 0.7 if zone in ("upper", "lower") else 1.0
        in_lung = ((xx - cx) ** 2) / (rx ** 2) + ((yy - cy + y_bias) ** 2) / (ry * y_factor) ** 2 <= 1.0
        mask[in_lung] = True
        reticular_network(img, mask, intensity=inten, density=density,
                          line_length_range=(4, 12), rng=rng)


def draw_honeycombing(img: np.ndarray, geom: LungGeometry, severity: int,
                       side: str, rng: np.random.Generator) -> int:
    cx, cy, rx, ry = geom.field(side if side in ("left", "right") else rng.choice(["left", "right"]))
    region_cx = cx + (rx * 0.5 * (-1 if side == "right" else 1))
    region_cy = cy + ry * 0.4
    region_radius = {1: 18, 2: 24, 3: 32}[severity]
    n_cysts = {1: rng.integers(5, 9), 2: rng.integers(9, 16),
               3: rng.integers(16, 25)}[severity]
    cyst_radius_range = (2.0, 4.0) if severity == 1 else (2.5, 5.0) if severity == 2 else (3.0, 6.5)
    wall_intensity = {1: 22, 2: 28, 3: 34}[severity]
    cyst_cluster(img, region_cx, region_cy, region_radius,
                 int(n_cysts), cyst_radius_range, wall_intensity, rng)
    return int(n_cysts)


# ---------- Edema marker ----------


def draw_kerley_b_lines(img: np.ndarray, geom: LungGeometry,
                         rng: np.random.Generator) -> None:
    for cx, cy, rx, ry in geom.both():
        n = int(rng.integers(8, 18))
        for _ in range(n):
            t = rng.uniform(0.55, 0.95)
            side_off = rng.choice([-1, 1])
            x = cx + side_off * rx * (0.78 + rng.uniform(-0.05, 0.05))
            y = cy - ry * 0.5 + ry * (t + 0.5)
            length = rng.uniform(5, 10)
            angle = rng.uniform(-0.2, 0.2)
            x_end = x - side_off * length * math.cos(angle)
            y_end = y - length * math.sin(angle)
            aa_line(img, x, y, x_end, y_end,
                    intensity=rng.uniform(10, 16), thickness=0.7, taper=False)


# ---------- Nodules / Micronodules ----------


def draw_nodule(img: np.ndarray, nd: NoduleSpec,
                 rng: np.random.Generator) -> None:
    x, y, sz = nd.x, nd.y, nd.size_px
    base_intensity = 36.0 + sz * 0.4

    if nd.has_halo:
        soft_ellipse(img, x, y, sz * 1.7, sz * 1.7,
                     intensity=base_intensity * 0.25, edge_softness=0.40)

    lobulated_blob(img, x, y, sz, intensity=base_intensity,
                   n_lobes=int(rng.integers(2, 5)),
                   lobe_amp=0.20, rng=rng, edge_softness=0.18,
                   inner_heterogeneity=0.08)

    if nd.cavitary:
        cav_r = sz * rng.uniform(0.35, 0.55)
        soft_ellipse(img, x + rng.uniform(-1, 1), y + rng.uniform(-1, 1),
                     cav_r, cav_r * rng.uniform(0.8, 1.2),
                     intensity=-base_intensity * 1.05, edge_softness=0.18)

    if nd.spiculated:
        n_spikes = int(rng.integers(7, 13))
        spike_length = sz * rng.uniform(0.7, 1.4)
        spiculated_lesion(img, x, y, sz * 0.4, n_spikes, spike_length,
                          spike_thickness=0.7, intensity=base_intensity * 0.8,
                          rng=rng)


def draw_micronodules(img: np.ndarray, geom: LungGeometry, count: int,
                       zone: str, rng: np.random.Generator) -> int:
    pts = []
    fields = geom.both()
    for _ in range(count):
        cx, cy, rx, ry = fields[rng.integers(0, 2)]
        u = rng.uniform(-1, 1)
        if zone == "lower":
            v = rng.uniform(-0.4, 1.0)
        elif zone == "upper":
            v = rng.uniform(-1.0, 0.4)
        else:
            v = rng.uniform(-1.0, 1.0)
        if u * u + v * v > 1.0: continue
        x = cx + u * rx * 0.85; y = cy + v * ry * 0.85
        pts.append((x, y))
    for x, y in pts:
        size_px = rng.uniform(0.7, 1.6)
        intensity = rng.uniform(28, 50)
        irregular_calcific(img, x, y, size_px, intensity, rng)
    return len(pts)


# ---------- Master pipeline ----------


def _capture(img: np.ndarray, raw_masks: dict[str, np.ndarray],
             label: str, draw_fn: Callable, *args, **kwargs):
    """Run draw_fn(img, *args) and accumulate the painted contribution under
    `raw_masks[label]` (max-blend across multiple instances)."""
    before = img.copy()
    result = draw_fn(img, *args, **kwargs)
    delta = np.abs(img - before)
    if delta.max() > 1e-6:
        if label in raw_masks:
            np.maximum(raw_masks[label], delta, out=raw_masks[label])
        else:
            raw_masks[label] = delta.astype(np.float32)
    return result


def render_state(state: FindingState, rng: np.random.Generator
                 ) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
    """Render an entire image from a FindingState.

    Returns:
        img        — uint8 array of shape (IMG_SIZE, IMG_SIZE)
        annotations — list of {finding, ...metadata, mask: <label>}
        masks      — dict label -> float32 array in [0, 1] of shape
                     (IMG_SIZE, IMG_SIZE), the soft segmentation per finding
    """
    img, geom = make_background(rng)
    annotations: list[dict] = []
    raw_masks: dict[str, np.ndarray] = {}

    if state.cardiomegaly > 0:
        _capture(img, raw_masks, "cardiomegaly",
                 draw_cardiomegaly, geom, state.cardiomegaly, rng)
        annotations.append({"finding": "cardiomegaly",
                             "severity": int(state.cardiomegaly),
                             "mask": "cardiomegaly"})

    if state.pulm_venous_redistribution:
        _capture(img, raw_masks, "pulm_venous_redistribution",
                 draw_pulm_venous_redistribution, geom, rng)
        annotations.append({"finding": "pulm_venous_redistribution",
                             "mask": "pulm_venous_redistribution"})

    if state.hilar_adenopathy:
        _capture(img, raw_masks, "hilar_adenopathy",
                 draw_hilar_adenopathy, geom, rng)
        annotations.append({"finding": "hilar_adenopathy",
                             "mask": "hilar_adenopathy"})

    if state.effusion_left > 0:
        _capture(img, raw_masks, "effusion_left",
                 draw_effusion, geom, "left", state.effusion_left, rng)
        annotations.append({"finding": "effusion_left",
                             "severity": int(state.effusion_left),
                             "mask": "effusion_left"})
    if state.effusion_right > 0:
        _capture(img, raw_masks, "effusion_right",
                 draw_effusion, geom, "right", state.effusion_right, rng)
        annotations.append({"finding": "effusion_right",
                             "severity": int(state.effusion_right),
                             "mask": "effusion_right"})

    if state.pneumothorax_left > 0:
        _capture(img, raw_masks, "pneumothorax_left",
                 draw_pneumothorax, geom, "left", state.pneumothorax_left, rng)
        annotations.append({"finding": "pneumothorax_left",
                             "severity": int(state.pneumothorax_left),
                             "mask": "pneumothorax_left"})
    if state.pneumothorax_right > 0:
        _capture(img, raw_masks, "pneumothorax_right",
                 draw_pneumothorax, geom, "right", state.pneumothorax_right, rng)
        annotations.append({"finding": "pneumothorax_right",
                             "severity": int(state.pneumothorax_right),
                             "mask": "pneumothorax_right"})

    if state.pleural_thickening_left:
        _capture(img, raw_masks, "pleural_thickening_left",
                 draw_pleural_thickening, geom, "left", rng)
        annotations.append({"finding": "pleural_thickening_left",
                             "mask": "pleural_thickening_left"})
    if state.pleural_thickening_right:
        _capture(img, raw_masks, "pleural_thickening_right",
                 draw_pleural_thickening, geom, "right", rng)
        annotations.append({"finding": "pleural_thickening_right",
                             "mask": "pleural_thickening_right"})

    if state.volume_loss_side:
        _capture(img, raw_masks, "volume_loss",
                 draw_volume_loss, geom, state.volume_loss_side, rng)
        annotations.append({"finding": "volume_loss",
                             "side": state.volume_loss_side,
                             "mask": "volume_loss"})

    if state.consolidation > 0:
        _capture(img, raw_masks, "consolidation",
                 draw_consolidation, geom, state.consolidation,
                 state.consolidation_side or rng.choice(["left", "right"]),
                 state.consolidation_zone,
                 state.consolidation_air_bronchogram, rng)
        annotations.append({"finding": "consolidation",
                             "severity": int(state.consolidation),
                             "zone": state.consolidation_zone,
                             "air_bronchogram": bool(state.consolidation_air_bronchogram),
                             "mask": "consolidation"})

    if state.reticular_pattern > 0:
        _capture(img, raw_masks, "reticular_pattern",
                 draw_reticular_pattern, geom, state.reticular_pattern,
                 state.reticular_zone, rng)
        annotations.append({"finding": "reticular_pattern",
                             "severity": int(state.reticular_pattern),
                             "zone": state.reticular_zone,
                             "mask": "reticular_pattern"})

    if state.honeycombing > 0:
        n_cysts = _capture(img, raw_masks, "honeycombing",
                           draw_honeycombing, geom, state.honeycombing,
                           state.honeycombing_side or rng.choice(["left", "right"]),
                           rng)
        annotations.append({"finding": "honeycombing",
                             "severity": int(state.honeycombing),
                             "n_cysts": int(n_cysts or 0),
                             "mask": "honeycombing"})

    if state.kerley_b_lines:
        _capture(img, raw_masks, "kerley_b_lines",
                 draw_kerley_b_lines, geom, rng)
        annotations.append({"finding": "kerley_b_lines",
                             "mask": "kerley_b_lines"})

    # Nodules — accumulate all into a single "nodules" mask
    for nd in state.nodules:
        _capture(img, raw_masks, "nodules", draw_nodule, nd, rng)
        annotations.append({
            "finding": "nodule",
            "size_px": float(nd.size_px),
            "cavitary": bool(nd.cavitary),
            "spiculated": bool(nd.spiculated),
            "halo": bool(nd.has_halo),
            "mask": "nodules",
        })

    if state.micronodule_count > 0:
        n_inst = _capture(img, raw_masks, "micronodules",
                          draw_micronodules, geom, state.micronodule_count,
                          state.micronodule_zone, rng)
        annotations.append({
            "finding": "micronodules",
            "count": int(state.micronodule_count),
            "n_instances": int(n_inst or 0),
            "zone": state.micronodule_zone,
            "mask": "micronodules",
        })

    img = np.clip(img, 0, 255).astype(np.uint8)

    masks = {label: (raw / max(raw.max(), 1e-6)).astype(np.float32)
             for label, raw in raw_masks.items()}
    return img, annotations, masks
