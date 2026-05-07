"""Clinical taxonomy for the v3 dataset.

Three label tiers:

  ATOMIC FINDINGS — what the renderer actually draws on the canvas.
    Examples: cardiomegaly (severity), effusion_left, kerley_b, hilar
    adenopathy, individual nodules with per-instance size/cavitary/spiculated
    attributes.

  COMPUTED LABELS — derived from atomic findings (counts, sizes,
    bilateralities). Examples: bilateral_effusion, any_mass (any nodule
    >15 px), nodule_count_solitary, micronodule_count_high.

  SYNDROME LABELS — derived from a combination of atomic findings.
    Examples: chf_syndrome (cardiomegaly + bilat effusion + kerley_b /
    pulmonary venous redistribution), uip_pattern (honeycombing + reticular
    + lower zone), sarcoidosis_pattern, tb_pattern, mesothelioma_triad,
    metastatic_pattern, miliary_pattern, solitary_pulmonary_nodule.

These tiers correspond to real radiology decision-making — see
SOURCES_REAL_MEDICAL.md for the published criteria each syndrome encodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Side = Literal["left", "right", "both", ""]
Zone = Literal["upper", "middle", "lower", "diffuse", "lower_subpleural"]


@dataclass
class NoduleSpec:
    x: float
    y: float
    size_px: float            # ≈ diameter in 320×320 image space (1 px ≈ 1.5 mm)
    cavitary: bool = False    # central lucency (TB-style)
    spiculated: bool = False  # malignancy hint
    has_halo: bool = False    # ground-glass halo (invasive aspergillosis-style)


@dataclass
class FindingState:
    # Cardiac / mediastinum
    cardiomegaly: int = 0                # 0=absent, 1=mild, 2=moderate, 3=severe
    pulm_venous_redistribution: bool = False  # CHF marker
    hilar_adenopathy: bool = False       # bilateral hilar enlargement (sarcoid)

    # Pleura
    effusion_left: int = 0
    effusion_right: int = 0
    pneumothorax_left: int = 0
    pneumothorax_right: int = 0
    pleural_thickening_left: bool = False
    pleural_thickening_right: bool = False
    volume_loss_side: Side = ""

    # Airspace
    consolidation: int = 0               # 0=none, 1=patchy, 2=lobar, 3=multilobar
    consolidation_side: Side = ""
    consolidation_zone: Zone = "middle"
    consolidation_air_bronchogram: bool = False

    # Interstitium
    reticular_pattern: int = 0
    reticular_zone: Zone = "diffuse"
    honeycombing: int = 0
    honeycombing_side: Side = ""

    # Edema
    kerley_b_lines: bool = False         # peripheral horizontal short lines

    # Nodules / micronodules
    nodules: list[NoduleSpec] = field(default_factory=list)
    micronodule_count: int = 0
    micronodule_zone: Zone = "diffuse"


# ---------- Computed labels ----------


def _nodule_size_bin(sz: float) -> str:
    """Map nodule size to clinical category (rough Fleischner-ish thresholds
    in our 320×320 pixel space, where 1 px ≈ 1.5 mm)."""
    if sz < 4:    return "subcentimeter_small"   # <6 mm
    if sz < 8:    return "subcentimeter_medium"  # 6-12 mm
    if sz < 15:   return "centimeter_small"      # 1.5-2.2 cm
    return "mass"                                # ≥2.2 cm


def derive_labels(s: FindingState) -> dict:
    """Compute atomic, count/size, and syndrome labels from a FindingState.

    Returns a flat dict of {label_name: int(0/1) | int(severity/count)}.
    The dataset writer uses these as columns in labels.csv."""
    L: dict = {}

    # ----- Atomic presence -----
    L["cardiomegaly"] = int(s.cardiomegaly > 0)
    L["cardiomegaly_severity"] = int(s.cardiomegaly)
    L["pulm_venous_redistribution"] = int(s.pulm_venous_redistribution)
    L["hilar_adenopathy"] = int(s.hilar_adenopathy)
    L["effusion_left"] = int(s.effusion_left > 0)
    L["effusion_right"] = int(s.effusion_right > 0)
    L["pneumothorax_left"] = int(s.pneumothorax_left > 0)
    L["pneumothorax_right"] = int(s.pneumothorax_right > 0)
    L["pleural_thickening_left"] = int(s.pleural_thickening_left)
    L["pleural_thickening_right"] = int(s.pleural_thickening_right)
    L["volume_loss"] = int(bool(s.volume_loss_side))
    L["consolidation"] = int(s.consolidation > 0)
    L["consolidation_severity"] = int(s.consolidation)
    L["consolidation_air_bronchogram"] = int(s.consolidation_air_bronchogram)
    L["reticular_pattern"] = int(s.reticular_pattern > 0)
    L["reticular_severity"] = int(s.reticular_pattern)
    L["honeycombing"] = int(s.honeycombing > 0)
    L["honeycombing_severity"] = int(s.honeycombing)
    L["kerley_b_lines"] = int(s.kerley_b_lines)

    # ----- Count + size derived (nodules) -----
    n_nodules = len(s.nodules)
    L["nodule_count"] = n_nodules
    L["nodule_solitary"] = int(n_nodules == 1)
    L["nodule_few"] = int(2 <= n_nodules <= 5)
    L["nodule_multiple"] = int(n_nodules > 5)

    if n_nodules > 0:
        sizes = [nd.size_px for nd in s.nodules]
        L["nodule_max_size_px"] = int(max(sizes))
        L["nodule_min_size_px"] = int(min(sizes))
        L["nodule_size_variability"] = int(max(sizes) - min(sizes))
        L["any_mass"] = int(any(sz >= 15 for sz in sizes))
        L["any_cavitary_nodule"] = int(any(nd.cavitary for nd in s.nodules))
        L["any_spiculated_nodule"] = int(any(nd.spiculated for nd in s.nodules))
        L["any_halo_nodule"] = int(any(nd.has_halo for nd in s.nodules))
    else:
        L["nodule_max_size_px"] = 0
        L["nodule_min_size_px"] = 0
        L["nodule_size_variability"] = 0
        L["any_mass"] = 0
        L["any_cavitary_nodule"] = 0
        L["any_spiculated_nodule"] = 0
        L["any_halo_nodule"] = 0

    # Micronodules
    L["micronodule_count"] = int(s.micronodule_count)
    L["micronodule_few"] = int(0 < s.micronodule_count <= 30)
    L["micronodule_many"] = int(s.micronodule_count > 30)

    # ----- Bilateralities -----
    L["bilateral_effusion"] = int(L["effusion_left"] and L["effusion_right"])
    L["bilateral_pneumothorax"] = int(L["pneumothorax_left"] and L["pneumothorax_right"])
    any_pneumo = L["pneumothorax_left"] or L["pneumothorax_right"]
    L["any_pneumothorax"] = int(any_pneumo)
    L["any_effusion"] = int(L["effusion_left"] or L["effusion_right"])
    L["any_pleural_thickening"] = int(L["pleural_thickening_left"] or L["pleural_thickening_right"])

    # ----- Syndrome labels -----
    # CHF: cardiomegaly + bilateral effusion + (kerley_b OR pulm venous redist)
    L["chf_syndrome"] = int(
        s.cardiomegaly >= 2 and
        L["bilateral_effusion"] and
        (s.kerley_b_lines or s.pulm_venous_redistribution)
    )

    # UIP/IPF: honeycombing + reticular + LOWER zone bias
    L["uip_pattern"] = int(
        s.honeycombing >= 1 and s.reticular_pattern >= 1 and
        s.reticular_zone in ("lower", "diffuse")
    )

    # Sarcoidosis: bilat hilar adenopathy + reticulonodular + UPPER zone + ≥3 nodules
    L["sarcoidosis_pattern"] = int(
        s.hilar_adenopathy and
        s.reticular_pattern >= 1 and s.reticular_zone == "upper" and
        n_nodules >= 3
    )

    # Post-primary TB: cavitary nodule + apical (upper zone) + ≥2 nodules total
    apical_consolidation = (s.consolidation > 0 and s.consolidation_zone == "upper")
    n_cavitary = sum(1 for nd in s.nodules if nd.cavitary)
    L["tb_pattern"] = int(
        n_cavitary >= 1 and
        (n_nodules >= 2 or apical_consolidation)
    )

    # Mesothelioma triad: effusion + pleural thickening (same side) + volume loss
    meso_left = L["effusion_left"] and L["pleural_thickening_left"] and (s.volume_loss_side == "left")
    meso_right = L["effusion_right"] and L["pleural_thickening_right"] and (s.volume_loss_side == "right")
    L["mesothelioma_pattern"] = int(meso_left or meso_right)

    # Tension pneumothorax: pneumothorax + contralateral volume loss / mediastinal shift
    L["tension_pneumothorax"] = int(
        (L["pneumothorax_left"] and s.volume_loss_side == "right") or
        (L["pneumothorax_right"] and s.volume_loss_side == "left")
    )

    # Metastatic disease: ≥3 nodules with varied sizes
    L["metastatic_pattern"] = int(
        n_nodules >= 3 and L["nodule_size_variability"] >= 4
    )

    # Miliary: many micronodules diffusely
    L["miliary_pattern"] = int(s.micronodule_count >= 80 and s.micronodule_zone == "diffuse")

    # Solitary pulmonary nodule (Fleischner-relevant): exactly 1 nodule between 4–30 px
    L["solitary_pulmonary_nodule"] = int(
        n_nodules == 1 and 4 <= s.nodules[0].size_px <= 30
    )

    # Lobar pneumonia: consolidation (≥lobar) + air bronchogram
    L["lobar_pneumonia_pattern"] = int(
        s.consolidation >= 2 and s.consolidation_air_bronchogram
    )

    return L


# Helper: list ordered groups for labels.csv columns
LABEL_GROUPS = {
    "atomic": [
        "cardiomegaly", "cardiomegaly_severity",
        "pulm_venous_redistribution", "hilar_adenopathy",
        "effusion_left", "effusion_right",
        "pneumothorax_left", "pneumothorax_right",
        "pleural_thickening_left", "pleural_thickening_right",
        "volume_loss",
        "consolidation", "consolidation_severity", "consolidation_air_bronchogram",
        "reticular_pattern", "reticular_severity",
        "honeycombing", "honeycombing_severity",
        "kerley_b_lines",
    ],
    "computed": [
        "nodule_count", "nodule_solitary", "nodule_few", "nodule_multiple",
        "nodule_max_size_px", "nodule_min_size_px", "nodule_size_variability",
        "any_mass", "any_cavitary_nodule", "any_spiculated_nodule", "any_halo_nodule",
        "micronodule_count", "micronodule_few", "micronodule_many",
        "bilateral_effusion", "bilateral_pneumothorax",
        "any_pneumothorax", "any_effusion", "any_pleural_thickening",
    ],
    "syndromes": [
        "chf_syndrome", "uip_pattern", "sarcoidosis_pattern", "tb_pattern",
        "mesothelioma_pattern", "tension_pneumothorax",
        "metastatic_pattern", "miliary_pattern",
        "solitary_pulmonary_nodule", "lobar_pneumonia_pattern",
    ],
}


ALL_LABELS = LABEL_GROUPS["atomic"] + LABEL_GROUPS["computed"] + LABEL_GROUPS["syndromes"]
