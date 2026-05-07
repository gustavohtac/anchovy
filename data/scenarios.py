"""Scenario-based sampler for the v3 dataset.

We mix three regimes:

  NORMAL (~35%): no findings or one minor incidental finding.
  RANDOM (~35%): 1–3 independently sampled atomic findings.
  SYNDROME (~30%): a clinically-coherent constellation. The syndrome
    *forces* the correlated findings to co-occur (CHF needs cardiomegaly +
    bilat effusion + kerley_b/cephalization; UIP needs honeycombing +
    reticular + lower-zone bias; etc.).

The mix of regimes gives the dataset both:
  - clean independent positives (so per-finding probes have a clean signal),
  - clean syndromic positives (so multi-finding syndrome probes have signal),
  - and a non-trivial confounding structure (syndromes naturally correlate
    findings, so probes that don't actually *see* the small finding can
    still get partial credit via the correlated large one — exactly the
    radiology shortcut problem we want to expose).
"""
from __future__ import annotations

import math

import numpy as np

from data.primitives.background import IMG_SIZE
from data.findings import FindingState, NoduleSpec


# Lung field bounding boxes (must match realistic_background)
_LUNG_LEFT = (IMG_SIZE * 0.32, IMG_SIZE * 0.46, 57.0, 88.0)
_LUNG_RIGHT = (IMG_SIZE * 0.68, IMG_SIZE * 0.46, 57.0, 88.0)


def _rand_lung_point(rng, side: str | None = None,
                      zone: str = "any", margin: float = 0.85) -> tuple[float, float]:
    """Sample a uniform point inside the chosen lung field. zone biases y."""
    if side is None:
        side = rng.choice(["left", "right"])
    cx, cy, rx, ry = _LUNG_LEFT if side == "left" else _LUNG_RIGHT
    while True:
        u = rng.uniform(-1, 1)
        if zone == "upper":
            v = rng.uniform(-1, 0.0)
        elif zone == "lower":
            v = rng.uniform(0.0, 1)
        elif zone == "apical":
            v = rng.uniform(-1, -0.5)
        else:
            v = rng.uniform(-1, 1)
        if u * u + v * v <= 1.0:
            return cx + u * rx * margin, cy + v * ry * margin


def _rand_nodule(rng, size_range, *, side=None, zone="any",
                  cavitary=False, spiculated=False, has_halo=False) -> NoduleSpec:
    x, y = _rand_lung_point(rng, side=side, zone=zone)
    sz = float(rng.uniform(*size_range))
    return NoduleSpec(x=x, y=y, size_px=sz,
                      cavitary=cavitary, spiculated=spiculated, has_halo=has_halo)


# ---------- Normal / minor ----------


def _scenario_normal(rng) -> FindingState:
    """A clean image, possibly with one minor incidental finding."""
    s = FindingState()
    if rng.random() < 0.35:
        which = rng.choice(["minor_calcific", "single_nodule", "minor_reticular"])
        if which == "minor_calcific":
            s.micronodule_count = int(rng.integers(2, 8))
            s.micronodule_zone = "diffuse"
        elif which == "single_nodule":
            s.nodules = [_rand_nodule(rng, size_range=(3, 6))]
        else:
            s.reticular_pattern = 1
            s.reticular_zone = "diffuse"
    return s


# ---------- Random independent findings ----------


def _scenario_random(rng) -> FindingState:
    s = FindingState()
    n = int(rng.integers(1, 4))
    options = [
        "cardiomegaly", "effusion", "pneumothorax", "consolidation",
        "reticular", "honeycombing", "nodules", "micronodules",
        "pleural_thickening",
    ]
    chosen = list(rng.choice(options, size=min(n, len(options)), replace=False))
    for opt in chosen:
        if opt == "cardiomegaly":
            s.cardiomegaly = int(rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
        elif opt == "effusion":
            sev = int(rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
            side = rng.choice(["left", "right", "both"], p=[0.40, 0.40, 0.20])
            if side in ("left", "both"): s.effusion_left = sev
            if side in ("right", "both"): s.effusion_right = sev
        elif opt == "pneumothorax":
            sev = int(rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
            side = rng.choice(["left", "right"])
            if side == "left": s.pneumothorax_left = sev
            else: s.pneumothorax_right = sev
        elif opt == "consolidation":
            s.consolidation = int(rng.choice([1, 2, 3], p=[0.40, 0.40, 0.20]))
            s.consolidation_side = rng.choice(["left", "right"])
            s.consolidation_zone = rng.choice(["upper", "middle", "lower"])
            s.consolidation_air_bronchogram = bool(rng.random() < 0.55)
        elif opt == "reticular":
            s.reticular_pattern = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
            s.reticular_zone = rng.choice(["upper", "diffuse", "lower"], p=[0.25, 0.45, 0.30])
        elif opt == "honeycombing":
            s.honeycombing = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
            s.honeycombing_side = rng.choice(["left", "right"])
        elif opt == "nodules":
            count = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.30, 0.20, 0.15, 0.12, 0.13, 0.10]))
            for _ in range(count):
                size = float(rng.uniform(4, 16))
                s.nodules.append(_rand_nodule(
                    rng, size_range=(size, size + 0.5),
                    cavitary=bool(rng.random() < 0.05),
                    spiculated=bool(rng.random() < 0.15),
                    has_halo=bool(rng.random() < 0.15),
                ))
        elif opt == "micronodules":
            s.micronodule_count = int(rng.choice([
                rng.integers(10, 30), rng.integers(30, 80),
                rng.integers(80, 200),
            ]))
            s.micronodule_zone = rng.choice(["diffuse", "lower", "upper"], p=[0.6, 0.25, 0.15])
        elif opt == "pleural_thickening":
            side = rng.choice(["left", "right"])
            if side == "left": s.pleural_thickening_left = True
            else: s.pleural_thickening_right = True
    return s


# ---------- Syndromes ----------


def _scenario_chf(rng) -> FindingState:
    """Congestive heart failure: cardiomegaly + bilateral effusion +
    kerley_b/cephalization. Real-world: textbook cardiac decompensation."""
    s = FindingState()
    s.cardiomegaly = int(rng.choice([2, 3], p=[0.55, 0.45]))
    s.effusion_left = int(rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25]))
    s.effusion_right = int(rng.choice([1, 2, 3], p=[0.30, 0.45, 0.25]))
    if rng.random() < 0.75:
        s.kerley_b_lines = True
    if rng.random() < 0.65:
        s.pulm_venous_redistribution = True
    return s


def _scenario_uip(rng) -> FindingState:
    """UIP/IPF pattern: honeycombing + reticular pattern + LOWER zone bias.
    Diagnostic by ATS/ERS criteria."""
    s = FindingState()
    s.honeycombing = int(rng.choice([2, 3], p=[0.55, 0.45]))
    s.honeycombing_side = rng.choice(["left", "right"])
    s.reticular_pattern = int(rng.choice([2, 3], p=[0.50, 0.50]))
    s.reticular_zone = "lower"
    return s


def _scenario_sarcoidosis(rng) -> FindingState:
    """Sarcoidosis: bilateral hilar adenopathy + reticulonodular pattern +
    UPPER zone predominance + multiple small nodules."""
    s = FindingState()
    s.hilar_adenopathy = True
    s.reticular_pattern = int(rng.choice([1, 2], p=[0.6, 0.4]))
    s.reticular_zone = "upper"
    n = int(rng.integers(4, 9))
    for _ in range(n):
        s.nodules.append(_rand_nodule(rng, size_range=(3, 5),
                                       zone="upper" if rng.random() < 0.7 else "any"))
    return s


def _scenario_tb(rng) -> FindingState:
    """Post-primary TB: ≥1 cavitary nodule + apical predominance + often
    consolidation in upper lobe."""
    s = FindingState()
    n_cav = int(rng.integers(1, 4))
    for _ in range(n_cav):
        s.nodules.append(_rand_nodule(rng, size_range=(8, 16),
                                       zone="apical", cavitary=True))
    n_extra = int(rng.integers(0, 4))
    for _ in range(n_extra):
        s.nodules.append(_rand_nodule(rng, size_range=(4, 8), zone="upper"))
    if rng.random() < 0.5:
        s.consolidation = int(rng.choice([1, 2]))
        s.consolidation_side = rng.choice(["left", "right"])
        s.consolidation_zone = "upper"
        s.consolidation_air_bronchogram = bool(rng.random() < 0.4)
    return s


def _scenario_miliary(rng) -> FindingState:
    """Miliary pattern: innumerable diffuse micronodules."""
    s = FindingState()
    s.micronodule_count = int(rng.integers(150, 450))
    s.micronodule_zone = "diffuse"
    return s


def _scenario_metastatic(rng) -> FindingState:
    """Metastatic disease: multiple nodules of varying sizes (cannonball
    pattern)."""
    s = FindingState()
    n = int(rng.integers(4, 12))
    for _ in range(n):
        s.nodules.append(_rand_nodule(rng, size_range=(4, 18),
                                       has_halo=bool(rng.random() < 0.10)))
    return s


def _scenario_solitary_nodule(rng) -> FindingState:
    """Solitary pulmonary nodule (4–30 px ≈ 6 mm – 4.5 cm). Critical
    Fleischner-style management decision."""
    s = FindingState()
    sz = float(rng.uniform(4, 30))
    s.nodules = [_rand_nodule(rng, size_range=(sz, sz + 0.1),
                               spiculated=bool(rng.random() < 0.40),
                               has_halo=bool(rng.random() < 0.10))]
    return s


def _scenario_lobar_pneumonia(rng) -> FindingState:
    """Lobar pneumonia: dense consolidation in single lobe + air
    bronchograms (defining feature)."""
    s = FindingState()
    s.consolidation = int(rng.choice([2, 3], p=[0.6, 0.4]))
    s.consolidation_side = rng.choice(["left", "right"])
    s.consolidation_zone = rng.choice(["upper", "middle", "lower"])
    s.consolidation_air_bronchogram = True
    return s


def _scenario_mesothelioma(rng) -> FindingState:
    """Mesothelioma: unilateral effusion + circumferential pleural
    thickening + ipsilateral volume loss (textbook triad)."""
    s = FindingState()
    side = rng.choice(["left", "right"])
    if side == "left":
        s.effusion_left = int(rng.choice([2, 3]))
        s.pleural_thickening_left = True
        s.volume_loss_side = "left"
    else:
        s.effusion_right = int(rng.choice([2, 3]))
        s.pleural_thickening_right = True
        s.volume_loss_side = "right"
    return s


def _scenario_tension_pneumothorax(rng) -> FindingState:
    """Tension pneumothorax: large pneumothorax with contralateral
    mediastinal shift (volume loss on opposite side)."""
    s = FindingState()
    side = rng.choice(["left", "right"])
    if side == "left":
        s.pneumothorax_left = 3
        s.volume_loss_side = "right"
    else:
        s.pneumothorax_right = 3
        s.volume_loss_side = "left"
    return s


SYNDROMES = [
    ("chf", _scenario_chf, 0.16),
    ("uip", _scenario_uip, 0.10),
    ("sarcoidosis", _scenario_sarcoidosis, 0.10),
    ("tb", _scenario_tb, 0.10),
    ("miliary", _scenario_miliary, 0.10),
    ("metastatic", _scenario_metastatic, 0.10),
    ("solitary_nodule", _scenario_solitary_nodule, 0.10),
    ("lobar_pneumonia", _scenario_lobar_pneumonia, 0.10),
    ("mesothelioma", _scenario_mesothelioma, 0.06),
    ("tension_pneumothorax", _scenario_tension_pneumothorax, 0.08),
]


def _sample_syndrome(rng) -> tuple[str, FindingState]:
    names = [s[0] for s in SYNDROMES]
    weights = np.array([s[2] for s in SYNDROMES], dtype=np.float64)
    weights /= weights.sum()
    idx = int(rng.choice(len(names), p=weights))
    return names[idx], SYNDROMES[idx][1](rng)


# ---------- Top-level scenario sampler ----------


def sample_scenario(rng,
                     p_normal: float = 0.30,
                     p_random: float = 0.35,
                     p_syndrome: float = 0.35) -> tuple[str, FindingState]:
    total = p_normal + p_random + p_syndrome
    p_normal /= total; p_random /= total; p_syndrome /= total
    r = rng.random()
    if r < p_normal:
        return "normal", _scenario_normal(rng)
    elif r < p_normal + p_random:
        return "random", _scenario_random(rng)
    else:
        return _sample_syndrome(rng)
