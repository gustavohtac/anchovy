"""Higher-fidelity drawing primitives for realistic radiograph synthesis.

Differences vs the v1 primitives:
  - Sub-pixel positions handled via Gaussian splatting (anti-aliased)
  - Anti-aliased lines (Wu-style, with optional thickness gradient)
  - Bezier and branching tree generators (vasculature, bronchi)
  - Irregular blobs via radial perlin-ish noise (lobulated nodules)
  - Spiculated rendering with variable spike lengths and angles
  - Pixel-accurate stippling for tiny calcifications
"""
from __future__ import annotations

import math

import numpy as np


def _gaussian_kernel(sigma: float) -> tuple[np.ndarray, int]:
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    return k, radius


def aa_dot(img: np.ndarray, x: float, y: float, intensity: float, sigma: float = 0.6) -> None:
    """Sub-pixel-accurate splatting of a tiny bright/dark dot.

    Models a point source at (x, y) integrated over a Gaussian PSF — same
    way real X-ray photon-counting detectors smear a sub-pixel bright spot
    into adjacent pixels. Preserves HF detail without aliasing."""
    h, w = img.shape
    radius = max(1, int(np.ceil(3 * sigma)))
    x0 = int(np.floor(x)) - radius
    x1 = int(np.floor(x)) + radius + 1
    y0 = int(np.floor(y)) - radius
    y1 = int(np.floor(y)) + radius + 1
    x0 = max(x0, 0); y0 = max(y0, 0)
    x1 = min(x1, w); y1 = min(y1, h)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    img[y0:y1, x0:x1] += intensity * g


def aa_line(
    img: np.ndarray, x0: float, y0: float, x1: float, y1: float,
    intensity: float, thickness: float = 1.0, taper: bool = False,
) -> None:
    """Anti-aliased line via point splatting along the segment.

    With taper=True the intensity fades at the endpoints (good for vessel
    branches). thickness controls Gaussian PSF sigma — sub-pixel widths OK."""
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(length * 3))
    ts = np.linspace(0.0, 1.0, n)
    sigma = max(0.4, thickness * 0.5)
    for t in ts:
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        if taper:
            fade = math.sin(math.pi * t) ** 0.4  # sharp falloff at ends
        else:
            fade = 1.0
        aa_dot(img, x, y, intensity * fade, sigma=sigma)


def bezier_curve(
    img: np.ndarray, p0: tuple[float, float], p1: tuple[float, float],
    p2: tuple[float, float], intensity: float, thickness: float = 1.0,
    taper: bool = True,
) -> None:
    """Quadratic Bezier — used for vessels and curved edges."""
    n = 60
    ts = np.linspace(0, 1, n)
    xs = (1 - ts) ** 2 * p0[0] + 2 * (1 - ts) * ts * p1[0] + ts ** 2 * p2[0]
    ys = (1 - ts) ** 2 * p0[1] + 2 * (1 - ts) * ts * p1[1] + ts ** 2 * p2[1]
    sigma = max(0.4, thickness * 0.5)
    for i in range(n):
        if taper:
            fade = math.sin(math.pi * (i / (n - 1))) ** 0.5
        else:
            fade = 1.0
        aa_dot(img, xs[i], ys[i], intensity * fade, sigma=sigma)


def vascular_tree(
    img: np.ndarray, x: float, y: float, angle: float, length: float,
    thickness: float, intensity: float, depth: int, rng: np.random.Generator,
    branch_prob: float = 0.7, length_decay: float = 0.7,
) -> None:
    """Recursive branching vessel-like structure. Each segment is a Bezier
    arc with slightly random control point; thickness decays with depth."""
    if depth <= 0 or thickness < 0.4 or length < 4:
        return
    # End point
    ex = x + length * math.cos(angle)
    ey = y + length * math.sin(angle)
    # Control point with some curvature
    perp = angle + math.pi / 2
    curve = rng.uniform(-length * 0.15, length * 0.15)
    cx = (x + ex) / 2 + curve * math.cos(perp)
    cy = (y + ey) / 2 + curve * math.sin(perp)
    bezier_curve(img, (x, y), (cx, cy), (ex, ey), intensity, thickness=thickness, taper=False)

    # Branch: split into 1-3 children
    n_children = int(rng.integers(1, 4)) if rng.random() < branch_prob else 1
    new_thickness = thickness * (0.7 + rng.uniform(-0.05, 0.05))
    new_length = length * (length_decay + rng.uniform(-0.1, 0.1))
    for _ in range(n_children):
        d_angle = rng.uniform(-0.6, 0.6)
        vascular_tree(
            img, ex, ey, angle + d_angle, new_length, new_thickness,
            intensity * (0.92 + rng.uniform(-0.05, 0.05)),
            depth - 1, rng, branch_prob, length_decay,
        )


def soft_ellipse(
    img: np.ndarray, cx: float, cy: float, rx: float, ry: float,
    intensity: float, edge_softness: float = 0.15, rotation: float = 0.0,
) -> None:
    """Filled ellipse with Gaussian edge falloff and optional rotation."""
    h, w = img.shape
    pad = int(max(rx, ry) + 4)
    y0 = max(0, int(cy - pad)); y1 = min(h, int(cy + pad) + 1)
    x0 = max(0, int(cx - pad)); x1 = min(w, int(cx + pad) + 1)
    if y0 >= y1 or x0 >= x1: return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    dx = xx - cx
    dy = yy - cy
    rx_ = (dx * cos_r + dy * sin_r) / max(rx, 1e-6)
    ry_ = (-dx * sin_r + dy * cos_r) / max(ry, 1e-6)
    r2 = rx_ * rx_ + ry_ * ry_
    sigma = edge_softness
    weight = np.exp(-np.maximum(0.0, r2 - 1.0) / (2.0 * sigma * sigma))
    weight = np.where(r2 <= 1.0, 1.0, weight)
    img[y0:y1, x0:x1] += intensity * weight


def lobulated_blob(
    img: np.ndarray, cx: float, cy: float, mean_radius: float,
    intensity: float, n_lobes: int, lobe_amp: float, rng: np.random.Generator,
    edge_softness: float = 0.12, rotation: float = 0.0,
    inner_heterogeneity: float = 0.0,
) -> None:
    """Filled blob with an irregular boundary (sum of low-frequency
    sinusoidal perturbations of the radius). Optional internal density
    heterogeneity (additive multi-octave noise inside)."""
    h, w = img.shape
    max_r = mean_radius * (1 + lobe_amp + 0.2)
    pad = int(max_r + 4)
    y0 = max(0, int(cy - pad)); y1 = min(h, int(cy + pad) + 1)
    x0 = max(0, int(cx - pad)); x1 = min(w, int(cx + pad) + 1)
    if y0 >= y1 or x0 >= x1: return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    theta = np.arctan2(dy, dx)
    r = np.sqrt(dx * dx + dy * dy)
    # Boundary perturbation: sum of sinusoids with random phases
    pert = np.zeros_like(theta)
    for k in range(1, n_lobes + 1):
        phase = rng.uniform(0, 2 * np.pi)
        amp = lobe_amp * (1 + 0.3 * rng.standard_normal()) / k
        pert += amp * np.sin(k * theta + phase + rotation)
    boundary = mean_radius * (1.0 + pert)
    rn = r / np.maximum(boundary, 1e-6)
    sigma = edge_softness
    weight = np.exp(-np.maximum(0.0, rn ** 2 - 1.0) / (2.0 * sigma * sigma))
    weight = np.where(rn <= 1.0, 1.0, weight)

    if inner_heterogeneity > 0:
        # Multi-octave noise inside the blob
        noise = np.zeros_like(weight)
        for octave in range(3):
            scale = 2 ** octave
            ny = int((y1 - y0) // scale + 1)
            nx = int((x1 - x0) // scale + 1)
            n_layer = rng.standard_normal(size=(ny, nx)).astype(np.float32)
            n_up = np.repeat(np.repeat(n_layer, scale, axis=0), scale, axis=1)
            n_up = n_up[: y1 - y0, : x1 - x0]
            noise += n_up / (2 ** octave)
        noise = (noise - noise.mean()) / (noise.std() + 1e-6)
        weight_with_noise = weight * (1.0 + inner_heterogeneity * noise)
        img[y0:y1, x0:x1] += intensity * weight_with_noise
    else:
        img[y0:y1, x0:x1] += intensity * weight


def spiculated_lesion(
    img: np.ndarray, cx: float, cy: float, core_radius: float,
    n_spikes: int, mean_spike_length: float, spike_thickness: float,
    intensity: float, rng: np.random.Generator,
) -> None:
    """Irregular core with thin spiculations radiating outward."""
    # Irregular core
    lobulated_blob(img, cx, cy, core_radius, intensity * 0.9,
                   n_lobes=3, lobe_amp=0.25, rng=rng, edge_softness=0.18)
    # Spikes
    angle_offset = rng.uniform(0, 2 * math.pi)
    for k in range(n_spikes):
        base_angle = angle_offset + k * (2 * math.pi / n_spikes)
        angle = base_angle + rng.uniform(-0.25, 0.25)
        length = mean_spike_length * (0.5 + rng.exponential(0.5))
        length = min(length, mean_spike_length * 2.5)
        x_end = cx + (core_radius + length) * math.cos(angle)
        y_end = cy + (core_radius + length) * math.sin(angle)
        x_start = cx + core_radius * 0.7 * math.cos(angle)
        y_start = cy + core_radius * 0.7 * math.sin(angle)
        intensity_factor = 0.6 + 0.3 * rng.random()
        aa_line(img, x_start, y_start, x_end, y_end,
                intensity * intensity_factor,
                thickness=spike_thickness * (0.7 + 0.5 * rng.random()),
                taper=True)


def reticular_network(
    img: np.ndarray, region_mask: np.ndarray, intensity: float, density: float,
    line_length_range: tuple[float, float], rng: np.random.Generator,
) -> None:
    """Generate a connecting reticular pattern in the masked region: short
    line segments with shared endpoints (so they connect into a network),
    not isolated random lines."""
    ys, xs = np.where(region_mask)
    if len(ys) == 0: return
    n_nodes = max(8, int(density * len(ys) / 50))
    idx = rng.choice(len(ys), size=min(n_nodes, len(ys)), replace=False)
    nodes_x = xs[idx].astype(np.float32) + rng.uniform(-0.5, 0.5, size=len(idx))
    nodes_y = ys[idx].astype(np.float32) + rng.uniform(-0.5, 0.5, size=len(idx))
    # For each node, connect to 2-4 nearest other nodes
    for i in range(len(nodes_x)):
        d = np.hypot(nodes_x - nodes_x[i], nodes_y - nodes_y[i])
        d[i] = np.inf
        n_connect = int(rng.integers(2, 5))
        nearest = np.argsort(d)[:n_connect]
        for j in nearest:
            seg_len = d[j]
            if seg_len < line_length_range[0] or seg_len > line_length_range[1]:
                continue
            thickness = rng.uniform(0.5, 0.9)
            inten = intensity * rng.uniform(0.7, 1.0)
            aa_line(img, nodes_x[i], nodes_y[i], nodes_x[j], nodes_y[j],
                    inten, thickness=thickness, taper=False)


def air_bronchogram(
    img: np.ndarray, cx: float, cy: float, opacity_radius: float,
    n_branches: int, intensity: float, rng: np.random.Generator,
) -> None:
    """Add dark linear/branching structures (air-filled bronchi visible
    against bright consolidation). intensity should be NEGATIVE."""
    angle_offset = rng.uniform(0, 2 * math.pi)
    for k in range(n_branches):
        base_angle = angle_offset + k * (2 * math.pi / n_branches) + rng.uniform(-0.4, 0.4)
        x = cx + opacity_radius * 0.1 * math.cos(base_angle)
        y = cy + opacity_radius * 0.1 * math.sin(base_angle)
        # Recursive bronchial tree (dark)
        _bronch_branch(img, x, y, base_angle, opacity_radius * 0.7,
                       thickness=1.5, intensity=intensity, depth=3, rng=rng)


def _bronch_branch(
    img: np.ndarray, x: float, y: float, angle: float, length: float,
    thickness: float, intensity: float, depth: int, rng: np.random.Generator,
) -> None:
    if depth <= 0 or thickness < 0.4 or length < 3: return
    ex = x + length * math.cos(angle)
    ey = y + length * math.sin(angle)
    aa_line(img, x, y, ex, ey, intensity, thickness=thickness, taper=False)
    new_t = thickness * 0.75
    new_l = length * 0.65
    for _ in range(int(rng.integers(1, 3))):
        d_a = rng.uniform(-0.5, 0.5)
        _bronch_branch(img, ex, ey, angle + d_a, new_l, new_t,
                       intensity * 0.9, depth - 1, rng)


def cyst_cluster(
    img: np.ndarray, cx: float, cy: float, region_radius: float,
    n_cysts: int, cyst_radius_range: tuple[float, float],
    wall_intensity: float, rng: np.random.Generator,
) -> None:
    """Cluster of small thick-walled cysts (honeycombing). Cysts have a
    bright wall and slightly darker interior than surrounding lung."""
    placed: list[tuple[float, float, float]] = []
    attempts = 0
    while len(placed) < n_cysts and attempts < n_cysts * 6:
        attempts += 1
        rho = rng.uniform(0, region_radius)
        theta = rng.uniform(0, 2 * math.pi)
        cx_c = cx + rho * math.cos(theta)
        cy_c = cy + rho * math.sin(theta)
        r = rng.uniform(*cyst_radius_range)
        # Avoid heavy overlap
        if any(math.hypot(cx_c - x, cy_c - y) < (r + rr) * 0.7 for x, y, rr in placed):
            continue
        placed.append((cx_c, cy_c, r))
    for cx_c, cy_c, r in placed:
        # Slightly darker interior
        soft_ellipse(img, cx_c, cy_c, r, r, intensity=-wall_intensity * 0.3,
                     edge_softness=0.2)
        # Bright thin wall (annulus): bigger ellipse minus smaller
        soft_ellipse(img, cx_c, cy_c, r * 1.15, r * 1.15,
                     intensity=wall_intensity * 0.7, edge_softness=0.18)
        soft_ellipse(img, cx_c, cy_c, r * 0.85, r * 0.85,
                     intensity=-wall_intensity * 0.7, edge_softness=0.15)


def irregular_calcific(
    img: np.ndarray, cx: float, cy: float, size_px: float,
    intensity: float, rng: np.random.Generator,
) -> None:
    """A single calcific particle with irregular shape and bright HF
    detail. Tiny but high contrast — designed to live at the edge of patch
    tokenization."""
    if size_px <= 1.5:
        # Smallest: 1-2 pixel high-contrast aa_dot with sub-pixel jitter
        for _ in range(int(rng.integers(1, 3))):
            jx = cx + rng.uniform(-0.4, 0.4)
            jy = cy + rng.uniform(-0.4, 0.4)
            aa_dot(img, jx, jy, intensity * rng.uniform(0.7, 1.0), sigma=0.55)
    else:
        # Multi-particle stippled cluster
        n_pix = int(np.clip(size_px * 1.2, 2, 7))
        for _ in range(n_pix):
            theta = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0, size_px * 0.5)
            jx = cx + r * math.cos(theta)
            jy = cy + r * math.sin(theta)
            aa_dot(img, jx, jy, intensity * rng.uniform(0.6, 1.0),
                   sigma=rng.uniform(0.5, 0.8))


def random_point_in_ellipse(
    cx: float, cy: float, rx: float, ry: float, rng: np.random.Generator,
) -> tuple[float, float]:
    while True:
        u, v = rng.uniform(-1.0, 1.0, size=2)
        if u * u + v * v <= 1.0:
            return cx + u * rx, cy + v * ry


def add_photon_noise(
    img: np.ndarray, baseline: float = 80.0, noise_scale: float = 0.6,
    rng: np.random.Generator | None = None,
) -> None:
    """Photon-counting noise model in place: stddev grows like sqrt(intensity).
    Applied AFTER all anatomy/findings are composited."""
    rng = rng or np.random.default_rng()
    intensity = np.maximum(img, 1.0)
    noise = rng.standard_normal(size=img.shape).astype(np.float32) * np.sqrt(intensity) * noise_scale
    img += noise


def add_low_freq_inhomogeneity(
    img: np.ndarray, amplitude: float, scale: int, rng: np.random.Generator,
) -> None:
    """Add a smooth low-frequency intensity bias (simulates uneven beam
    intensity / phantom inhomogeneity)."""
    h, w = img.shape
    nh, nw = max(2, h // scale), max(2, w // scale)
    base = rng.standard_normal((nh, nw)).astype(np.float32)
    # Bilinear upsample
    ys = np.linspace(0, nh - 1, h, dtype=np.float32)
    xs = np.linspace(0, nw - 1, w, dtype=np.float32)
    yi = ys.astype(np.int32); xi = xs.astype(np.int32)
    yf = ys - yi; xf = xs - xi
    yi1 = np.minimum(yi + 1, nh - 1); xi1 = np.minimum(xi + 1, nw - 1)
    a = base[yi[:, None], xi[None, :]]
    b = base[yi[:, None], xi1[None, :]]
    c = base[yi1[:, None], xi[None, :]]
    d = base[yi1[:, None], xi1[None, :]]
    interp = (a * (1 - xf)[None, :] + b * xf[None, :]) * (1 - yf)[:, None] + \
             (c * (1 - xf)[None, :] + d * xf[None, :]) * yf[:, None]
    img += amplitude * interp
