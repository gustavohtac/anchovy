"""Free-text radiology reports for v3 findings.

Designed to imitate the language and structure of real chest-radiograph
reports as a training target. Each report is composed from three
randomized layers:

  1. Structure  — sectioned (TECHNIQUE / INDICATION / COMPARISON / FINDINGS /
     IMPRESSION), anatomical sub-headers within FINDINGS, or pure prose.
  2. Phrasing   — every finding draws from a bank of 3-6 alternative
     sentences with severity-adjective slots, hedging, and combined forms
     for related findings (bilateral effusions, pneumothoraces, multiple
     nodules, etc.).
  3. Indication — clinical-history line aligned with the dominant pattern
     (CHF, TB, pneumonia, pneumothorax, metastases, IPF, sarcoidosis).

Output of :func:`compose_report`::

    {
        "text":        full free-text report (training target),
        "findings":    just the FINDINGS body (legacy, plain),
        "impression":  just the IMPRESSION body (legacy, plain),
    }
"""
from __future__ import annotations

import numpy as np

from data.findings import FindingState, derive_labels


# ---------- Adjective / qualifier banks ----------

SEVERITY_ADJ = {
    1: ["mild", "subtle", "minimal", "trace", "minor", "faint", "small", "early"],
    2: ["moderate", "notable", "appreciable", "definite", "evident"],
    3: ["marked", "severe", "pronounced", "significant", "extensive",
        "prominent", "advanced", "striking", "florid"],
}

# Adverbial forms used in templates like "{adv} enlarged". Curated separately
# because not every adjective above forms a clean adverb (e.g. "trace"/"minor"/"small").
SEVERITY_ADV = {
    1: ["mildly", "subtly", "minimally", "faintly", "slightly"],
    2: ["moderately", "appreciably", "definitely", "evidently", "notably"],
    3: ["markedly", "severely", "prominently", "significantly", "extensively",
        "strikingly"],
}


# ---------- Report-level scaffolding ----------

TECHNIQUE_LINES = [
    "Single AP frontal chest radiograph.",
    "PA and lateral chest radiographs were obtained.",
    "Frontal chest radiograph performed at the bedside.",
    "Portable AP chest radiograph.",
    "Single PA view of the chest.",
    "Upright frontal chest radiograph.",
    "Chest radiograph, supine AP technique.",
    "Single frontal view of the chest, performed portably.",
]


INDICATION_LINES = {
    "chf": [
        "Acute dyspnea, evaluate for cardiac decompensation.",
        "Worsening shortness of breath.",
        "Suspected congestive heart failure.",
        "Lower extremity edema and dyspnea on exertion.",
    ],
    "tb": [
        "Chronic productive cough.",
        "Cough, weight loss, and night sweats.",
        "Evaluate for tuberculosis; positive PPD.",
        "Hemoptysis.",
    ],
    "pna": [
        "Fever and productive cough.",
        "Suspected pneumonia.",
        "Acute febrile illness with leukocytosis.",
        "Cough and pleuritic chest pain.",
    ],
    "pneumothorax": [
        "Sudden-onset pleuritic chest pain.",
        "Acute dyspnea, post-procedural.",
        "Status post line placement.",
        "Status post thoracentesis.",
    ],
    "metastatic": [
        "Known malignancy, surveillance imaging.",
        "Staging of biopsy-proven cancer.",
        "Follow-up of pulmonary nodules.",
        "History of breast cancer, surveillance.",
    ],
    "ipf": [
        "Progressive exertional dyspnea.",
        "Chronic non-productive cough.",
        "Suspected interstitial lung disease.",
        "Chronic dyspnea, evaluate for fibrosis.",
    ],
    "sarcoid": [
        "Suspected sarcoidosis.",
        "Bilateral hilar fullness on prior imaging.",
        "Dyspnea with skin findings.",
    ],
    "default": [
        "Routine chest imaging.",
        "Pre-operative evaluation.",
        "Evaluate cardiopulmonary status.",
        "Annual screening.",
        "Clinical assessment.",
        "Asymptomatic, screening.",
    ],
}


COMPARISON_LINES = [
    "No prior imaging available for comparison.",
    "No comparison available.",
    "Compared with prior radiograph; findings are new.",
    "When compared to the prior study, findings have progressed.",
    "Stable when compared to prior.",
    "No prior.",
]


# ---------- Per-finding phrase banks ----------

PHRASES = {
    "cardiomegaly": [
        "There is {sev} cardiomegaly.",
        "The cardiac silhouette is {adv} enlarged.",
        "{Sev_cap} enlargement of the cardiac silhouette.",
        "Cardiomediastinal contour is {adv} widened, compatible with cardiomegaly.",
        "Heart size is {adv} increased.",
        "The cardiac silhouette appears {adv} enlarged on this projection.",
    ],
    "pulm_venous_redistribution": [
        "Upper-zone pulmonary vascular redistribution is present.",
        "Cephalization of the pulmonary vasculature is noted.",
        "Pulmonary vasculature is redistributed cranially.",
        "Engorgement of the upper-zone pulmonary vessels.",
        "Vascular cephalization compatible with elevated left atrial pressure.",
    ],
    "kerley_b": [
        "Kerley B lines are visible at the lung bases.",
        "Short peripheral horizontal opacities consistent with Kerley B lines.",
        "Septal lines are seen at the costophrenic angles bilaterally.",
        "Interstitial septal thickening at the bases (Kerley B).",
        "Bilateral basal Kerley B lines.",
    ],
    "hilar_adenopathy": [
        "Bilateral hilar adenopathy is appreciated.",
        "There is bilateral hilar fullness suggestive of lymphadenopathy.",
        "Bilateral lobulated hilar enlargement.",
        "Symmetric prominence of the hila, compatible with adenopathy.",
        "Bilateral hilar prominence with lobulated contour.",
    ],
    "effusion_unilateral": [
        "{Sev_cap} {side} pleural effusion.",
        "There is a {sev} effusion in the {side} pleural space.",
        "Layering fluid in the {side} hemithorax, {sev} in volume.",
        "Blunting of the {side} costophrenic angle compatible with {sev} effusion.",
        "{Sev_cap} fluid layering in the {side} pleural cavity.",
        "{Side_cap}-sided pleural effusion, {sev} in extent.",
    ],
    "effusion_bilateral": [
        "Bilateral {sev} pleural effusions.",
        "{Sev_cap} pleural effusions are present bilaterally.",
        "Layering fluid in both pleural spaces, {sev} in volume.",
        "Bilateral blunting of the costophrenic angles, compatible with {sev} effusions.",
        "There are {sev} bilateral pleural effusions.",
    ],
    "pneumothorax_unilateral": [
        "{Sev_cap} {side} apical pneumothorax.",
        "There is a {sev} pneumothorax at the {side} apex.",
        "{Sev_cap} {side}-sided pleural air collection.",
        "{Side_cap} pneumothorax, {sev} in size.",
        "Visualized visceral pleural line at the {side} apex compatible with {sev} pneumothorax.",
    ],
    "pneumothorax_bilateral": [
        "Bilateral pneumothoraces.",
        "Pneumothoraces are present on both sides.",
        "Bilateral apical pleural air collections.",
    ],
    "pleural_thickening": [
        "Pleural thickening along the {sides} pleural surface.",
        "{Sides_cap} pleural thickening.",
        "Irregular pleural thickening on the {sides}.",
        "Smooth pleural thickening, {sides}-sided.",
    ],
    "volume_loss": [
        "Volume loss in the {side} hemithorax with mediastinal shift toward the affected side.",
        "Decreased {side} lung volume with ipsilateral mediastinal deviation.",
        "{Side_cap} hemithoracic volume loss.",
        "Crowding of the {side} lung markings with ipsilateral mediastinal shift.",
        "Mediastinal shift to the {side} compatible with volume loss.",
    ],
    "consolidation": [
        "{Sev_cap} airspace consolidation in the {side} {zone} zone{ab}.",
        "Patchy {sev} opacity in the {side} {zone} lung{ab}.",
        "{Sev_cap} consolidation involving the {side} {zone} lung{ab}.",
        "{Sev_cap} opacification in the {zone} {side} hemithorax{ab}.",
        "Focal {sev} airspace disease in the {side} {zone} lung{ab}.",
    ],
    "reticular": [
        "{Sev_cap} reticular interstitial pattern, {zone}-zone predominant.",
        "Reticulation in a {zone}-zone distribution, {sev} in extent.",
        "{Sev_cap} {zone}-predominant interlobular septal thickening.",
        "Fine reticular markings, {zone}-predominant, {sev} in degree.",
        "Diffuse fine reticular opacities with {zone}-zone bias, {sev} in severity.",
    ],
    "honeycombing": [
        "{Sev_cap} subpleural honeycombing.",
        "Subpleural cystic clustering compatible with honeycombing.",
        "Honeycomb-like cystic change, {sev} in extent, in a peripheral distribution.",
        "Peripheral cystic airspaces consistent with honeycombing, {sev} in degree.",
    ],
    "nodule_solitary": [
        "Solitary {qualifier}pulmonary nodule, approximately {sz} mm in diameter.",
        "There is a single {qualifier}pulmonary nodule measuring approximately {sz} mm.",
        "{Qualifier_cap}pulmonary nodule, ~{sz} mm.",
        "A solitary {qualifier}pulmonary nodule is identified, ~{sz} mm.",
    ],
    "nodule_multiple": [
        "{n} pulmonary nodules{qual_str}, sizes ranging from {smin} to {smax} mm.",
        "Multiple ({n}) pulmonary nodules{qual_str}, {smin}–{smax} mm.",
        "{n} discrete pulmonary nodules{qual_str}; sizes {smin}–{smax} mm.",
        "Several ({n}) nodules are identified{qual_str}, between {smin} and {smax} mm.",
    ],
    "micronodules_innumerable": [
        "Innumerable micronodules in a {zone} distribution.",
        "Diffuse {zone} miliary micronodular opacities.",
        "Countless small nodular densities, {zone} predominant.",
        "Innumerable punctate opacities throughout both lungs ({zone} predominant).",
    ],
    "micronodules_few": [
        "{n} scattered micronodular calcific densities ({zone} distribution).",
        "Several ({n}) small calcific micronodules in a {zone} pattern.",
        "{n} punctate calcific opacities, {zone} predominant.",
        "Scattered ({n}) calcified micronodules, {zone}-predominant.",
    ],
}


IMPRESSION = {
    "chf_syndrome": [
        "Findings consistent with congestive heart failure.",
        "Constellation compatible with cardiogenic pulmonary edema.",
        "Cardiomegaly with bilateral effusions and septal lines, in keeping with CHF.",
        "Decompensated heart failure suspected based on cardiomegaly, effusions, and vascular redistribution.",
    ],
    "uip_pattern": [
        "Pattern suggestive of usual interstitial pneumonia (UIP/IPF).",
        "Subpleural reticulation and honeycombing consistent with UIP.",
        "Findings concerning for idiopathic pulmonary fibrosis (UIP pattern).",
    ],
    "sarcoidosis_pattern": [
        "Bilateral hilar adenopathy with upper-zone reticulonodular changes; consider sarcoidosis.",
        "Findings consistent with pulmonary sarcoidosis.",
        "Pattern compatible with stage II sarcoidosis (adenopathy plus parenchymal involvement).",
    ],
    "tb_pattern": [
        "Cavitary apical opacities concerning for post-primary tuberculosis.",
        "Apical cavitation raises concern for active pulmonary tuberculosis; clinical correlation and isolation advised.",
        "Cavitary upper-zone disease suspicious for post-primary TB.",
    ],
    "mesothelioma_pattern": [
        "Unilateral pleural effusion with thickening and ipsilateral volume loss; mesothelioma should be considered.",
        "Triad of unilateral effusion, pleural thickening, and volume loss; consider malignant mesothelioma.",
        "Unilateral pleural disease with volume loss is concerning for mesothelioma; CT recommended.",
    ],
    "tension_pneumothorax": [
        "Large pneumothorax with contralateral mediastinal shift; tension physiology suspected — urgent decompression.",
        "Tension pneumothorax suspected based on contralateral shift; prompt clinical action required.",
    ],
    "metastatic_pattern": [
        "Multiple pulmonary nodules of varying sizes, concerning for metastatic disease.",
        "Cannonball-like nodules of variable size, suspicious for hematogenous metastases.",
        "Multifocal nodular disease; metastatic neoplasm should be considered.",
    ],
    "miliary_pattern": [
        "Diffuse miliary nodular pattern; differential includes miliary tuberculosis and hematogenous metastases.",
        "Miliary micronodular pattern.",
        "Innumerable micronodules in a miliary distribution; consider tuberculosis or fungal infection.",
    ],
    "lobar_pneumonia_pattern": [
        "Lobar consolidation with air bronchograms, consistent with bacterial pneumonia.",
        "Lobar opacification compatible with community-acquired pneumonia.",
    ],
    "spn_small": [
        "Subcentimeter pulmonary nodule (~{sz} mm); follow Fleischner low-risk guidance.",
        "Small pulmonary nodule (~{sz} mm); routine follow-up advised.",
    ],
    "spn_medium": [
        "Indeterminate pulmonary nodule (~{sz} mm); CT follow-up recommended.",
        "Pulmonary nodule (~{sz} mm), indeterminate; non-contrast CT chest recommended.",
    ],
    "spn_large": [
        "Pulmonary nodule (~{sz} mm); further characterization with PET-CT or biopsy advised.",
        "Pulmonary mass (~{sz} mm); tissue sampling should be considered.",
    ],
}


NORMAL_BODY = [
    "The lungs are clear bilaterally without focal consolidation, effusion, or pneumothorax. The cardiomediastinal silhouette is within normal limits. No acute osseous abnormality.",
    "Lungs are well-expanded and clear. Heart size is normal. No pleural effusion or pneumothorax.",
    "No focal airspace opacity, pleural effusion, or pneumothorax. Cardiomediastinal contour is normal. Bones are intact.",
    "Clear lung fields bilaterally. Normal cardiomediastinal silhouette. No acute findings.",
    "The lungs are clear. The cardiac silhouette and mediastinum are unremarkable. No acute cardiopulmonary abnormality is identified.",
]


NORMAL_IMPRESSION = [
    "No acute cardiopulmonary findings.",
    "No acute intrathoracic abnormality.",
    "Normal chest radiograph.",
    "Unremarkable chest radiograph.",
    "No acute cardiopulmonary process.",
]


# ---------- Helpers ----------


def _pick(rng: np.random.Generator, options):
    return options[int(rng.integers(0, len(options)))]


def _adj(sev: int, rng) -> str:
    return _pick(rng, SEVERITY_ADJ[max(1, min(3, sev))])


def _adv(sev: int, rng) -> str:
    return _pick(rng, SEVERITY_ADV[max(1, min(3, sev))])


def _fill(template: str, **kwargs) -> str:
    """Format a phrase template, also filling `_cap` capitalized variants on
    the fly. e.g. {Sev_cap} from kwargs['sev']='moderate' -> 'Moderate'."""
    extras = {}
    for k, v in list(kwargs.items()):
        if isinstance(v, str):
            extras[k.capitalize() + "_cap"] = v.capitalize() if v else v
            extras[k[:1].upper() + k[1:] + "_cap"] = v.capitalize() if v else v
    return template.format(**{**kwargs, **extras})


def _dominant_indication(labels: dict, s: FindingState) -> str:
    if labels["chf_syndrome"]: return "chf"
    if labels["tb_pattern"] or labels["miliary_pattern"]: return "tb"
    if labels["lobar_pneumonia_pattern"] or s.consolidation > 0: return "pna"
    if labels["any_pneumothorax"]: return "pneumothorax"
    if labels["metastatic_pattern"] or labels["solitary_pulmonary_nodule"]: return "metastatic"
    if labels["uip_pattern"]: return "ipf"
    if labels["sarcoidosis_pattern"]: return "sarcoid"
    return "default"


# ---------- Per-finding sentence generation ----------


def _sentences_for(s: FindingState, rng) -> list[str]:
    """Generate one or more sentences per active finding, drawing each
    from the matching phrase bank."""
    out: list[str] = []

    if s.cardiomegaly > 0:
        out.append(_fill(_pick(rng, PHRASES["cardiomegaly"]),
                         sev=_adj(s.cardiomegaly, rng),
                         adv=_adv(s.cardiomegaly, rng)))

    if s.pulm_venous_redistribution:
        out.append(_pick(rng, PHRASES["pulm_venous_redistribution"]))

    if s.kerley_b_lines:
        out.append(_pick(rng, PHRASES["kerley_b"]))

    if s.hilar_adenopathy:
        out.append(_pick(rng, PHRASES["hilar_adenopathy"]))

    # Effusion: bilateral combined / unilateral separately
    if s.effusion_left and s.effusion_right:
        sev = max(s.effusion_left, s.effusion_right)
        out.append(_fill(_pick(rng, PHRASES["effusion_bilateral"]),
                         sev=_adj(sev, rng)))
    elif s.effusion_left:
        out.append(_fill(_pick(rng, PHRASES["effusion_unilateral"]),
                         sev=_adj(s.effusion_left, rng), side="left"))
    elif s.effusion_right:
        out.append(_fill(_pick(rng, PHRASES["effusion_unilateral"]),
                         sev=_adj(s.effusion_right, rng), side="right"))

    if s.pneumothorax_left and s.pneumothorax_right:
        out.append(_pick(rng, PHRASES["pneumothorax_bilateral"]))
    elif s.pneumothorax_left:
        out.append(_fill(_pick(rng, PHRASES["pneumothorax_unilateral"]),
                         sev=_adj(s.pneumothorax_left, rng), side="left"))
    elif s.pneumothorax_right:
        out.append(_fill(_pick(rng, PHRASES["pneumothorax_unilateral"]),
                         sev=_adj(s.pneumothorax_right, rng), side="right"))

    if s.pleural_thickening_left or s.pleural_thickening_right:
        sides_list = []
        if s.pleural_thickening_left: sides_list.append("left")
        if s.pleural_thickening_right: sides_list.append("right")
        sides = " and ".join(sides_list) if len(sides_list) > 1 else sides_list[0]
        out.append(_fill(_pick(rng, PHRASES["pleural_thickening"]),
                         sides=sides))

    if s.volume_loss_side:
        out.append(_fill(_pick(rng, PHRASES["volume_loss"]),
                         side=s.volume_loss_side))

    if s.consolidation > 0:
        side = s.consolidation_side or "unilateral"
        ab = " with air bronchograms" if s.consolidation_air_bronchogram else ""
        out.append(_fill(_pick(rng, PHRASES["consolidation"]),
                         sev=_adj(s.consolidation, rng),
                         side=side, zone=s.consolidation_zone, ab=ab))

    if s.reticular_pattern > 0:
        out.append(_fill(_pick(rng, PHRASES["reticular"]),
                         sev=_adj(s.reticular_pattern, rng),
                         zone=s.reticular_zone))

    if s.honeycombing > 0:
        out.append(_fill(_pick(rng, PHRASES["honeycombing"]),
                         sev=_adj(s.honeycombing, rng)))

    n = len(s.nodules)
    if n == 1:
        nd = s.nodules[0]
        qualifiers = []
        if nd.spiculated: qualifiers.append("spiculated")
        if nd.cavitary: qualifiers.append("cavitary")
        if nd.has_halo: qualifiers.append("halo-bearing")
        qualifier = (", ".join(qualifiers) + " ") if qualifiers else ""
        out.append(_fill(_pick(rng, PHRASES["nodule_solitary"]),
                         qualifier=qualifier, sz=int(nd.size_px * 1.5)))
    elif n > 1:
        sizes = [nd.size_px for nd in s.nodules]
        n_cav = sum(1 for nd in s.nodules if nd.cavitary)
        n_spic = sum(1 for nd in s.nodules if nd.spiculated)
        n_halo = sum(1 for nd in s.nodules if nd.has_halo)
        descr = []
        if n_cav: descr.append(f"{n_cav} cavitary")
        if n_spic: descr.append(f"{n_spic} spiculated")
        if n_halo: descr.append(f"{n_halo} with halo")
        qual_str = " (" + ", ".join(descr) + ")" if descr else ""
        out.append(_fill(_pick(rng, PHRASES["nodule_multiple"]),
                         n=n, qual_str=qual_str,
                         smin=int(min(sizes) * 1.5),
                         smax=int(max(sizes) * 1.5)))

    if s.micronodule_count > 0:
        if s.micronodule_count >= 80:
            out.append(_fill(_pick(rng, PHRASES["micronodules_innumerable"]),
                             zone=s.micronodule_zone))
        else:
            out.append(_fill(_pick(rng, PHRASES["micronodules_few"]),
                             n=int(s.micronodule_count),
                             zone=s.micronodule_zone))

    return out


# ---------- Anatomical sub-header bucketing ----------


def _bucket_sentences(s: FindingState, sentences: list[str]) -> dict[str, list[str]]:
    """Roughly group findings sentences into anatomical buckets for the
    sectioned-by-system report style. Best-effort substring matching."""
    buckets: dict[str, list[str]] = {
        "Heart and mediastinum": [],
        "Pleura": [],
        "Lungs": [],
        "Other": [],
    }
    for sent in sentences:
        low = sent.lower()
        if any(k in low for k in ("cardio", "heart", "cardiac", "mediastin", "vascular redistr", "cephaliz", "hilar")):
            buckets["Heart and mediastinum"].append(sent)
        elif any(k in low for k in ("effusion", "pneumothorax", "pleural")):
            buckets["Pleura"].append(sent)
        elif any(k in low for k in (
            "consolidation", "reticular", "honeycomb", "kerley", "nodule",
            "micronod", "opacit", "airspace", "bronchogram", "fibrosi", "interstit",
        )):
            buckets["Lungs"].append(sent)
        else:
            buckets["Other"].append(sent)
    return buckets


# ---------- Impression composition ----------


def _impression_lines(labels: dict, s: FindingState, rng) -> list[str]:
    out: list[str] = []
    if labels["chf_syndrome"]:
        out.append(_pick(rng, IMPRESSION["chf_syndrome"]))
    if labels["uip_pattern"]:
        out.append(_pick(rng, IMPRESSION["uip_pattern"]))
    if labels["sarcoidosis_pattern"]:
        out.append(_pick(rng, IMPRESSION["sarcoidosis_pattern"]))
    if labels["tb_pattern"]:
        out.append(_pick(rng, IMPRESSION["tb_pattern"]))
    if labels["mesothelioma_pattern"]:
        out.append(_pick(rng, IMPRESSION["mesothelioma_pattern"]))
    if labels["tension_pneumothorax"]:
        out.append(_pick(rng, IMPRESSION["tension_pneumothorax"]))
    if labels["metastatic_pattern"]:
        out.append(_pick(rng, IMPRESSION["metastatic_pattern"]))
    if labels["miliary_pattern"]:
        out.append(_pick(rng, IMPRESSION["miliary_pattern"]))
    if labels["lobar_pneumonia_pattern"]:
        out.append(_pick(rng, IMPRESSION["lobar_pneumonia_pattern"]))

    if labels["solitary_pulmonary_nodule"]:
        sz = int(s.nodules[0].size_px * 1.5)
        if sz < 6:
            out.append(_fill(_pick(rng, IMPRESSION["spn_small"]), sz=sz))
        elif sz < 12:
            out.append(_fill(_pick(rng, IMPRESSION["spn_medium"]), sz=sz))
        else:
            out.append(_fill(_pick(rng, IMPRESSION["spn_large"]), sz=sz))

    if not out:
        # Generic descriptive fallbacks for findings that don't form a named syndrome
        if s.cardiomegaly > 0:
            out.append(f"{_adj(s.cardiomegaly, rng).capitalize()} cardiomegaly.")
        if labels["any_effusion"]:
            out.append("Pleural effusion, etiology to be determined clinically.")
        if s.consolidation > 0:
            out.append("Pulmonary consolidation; correlate clinically for infection vs other etiology.")
        if s.honeycombing > 0:
            out.append("Honeycomb interstitial change.")
        if s.reticular_pattern > 0:
            out.append("Reticular interstitial markings.")
        if labels["any_pneumothorax"]:
            out.append("Pneumothorax.")
        if labels["any_pleural_thickening"]:
            out.append("Pleural thickening; consider asbestos exposure or prior inflammation.")
        if s.hilar_adenopathy:
            out.append("Hilar fullness compatible with adenopathy; correlate clinically.")
        if s.volume_loss_side:
            out.append(f"Volume loss in the {s.volume_loss_side} hemithorax.")
        if s.pulm_venous_redistribution or s.kerley_b_lines:
            out.append("Findings compatible with interstitial pulmonary edema.")
        if len(s.nodules) >= 2 and not labels["metastatic_pattern"]:
            out.append(f"{len(s.nodules)} pulmonary nodules; further characterization with CT recommended.")
        if s.micronodule_count > 0 and not labels["miliary_pattern"]:
            out.append(f"Scattered micronodular opacities; differential includes prior granulomatous disease.")
    if not out:
        out.append("Findings as described above.")
    return out


def _format_impression(lines: list[str], rng) -> str:
    if not lines:
        return _pick(rng, NORMAL_IMPRESSION)
    if len(lines) == 1:
        return lines[0]
    # Numbered list ~70%, prose join ~30%
    if rng.random() < 0.70:
        return "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
    return " ".join(lines)


# ---------- Top-level composition ----------


def _compose_findings_body(sentences: list[str], rng) -> tuple[str, str]:
    """Return (prose_findings, structured_findings_with_subheaders)."""
    rng.shuffle(sentences)
    prose = " ".join(sentences)
    return prose, prose  # caller picks which to keep


def compose_report(s: FindingState, rng) -> dict[str, str]:
    """Compose a free-text radiology report imitating real-world phrasing.

    Returns ``{"text", "findings", "impression"}``.
    """
    labels = derive_labels(s)
    sentences = _sentences_for(s, rng)

    technique = _pick(rng, TECHNIQUE_LINES)
    indication = _pick(rng, INDICATION_LINES[_dominant_indication(labels, s)])
    has_indication = rng.random() < 0.75
    has_comparison = rng.random() < 0.55
    comparison = _pick(rng, COMPARISON_LINES) if has_comparison else None

    if not sentences:
        body_plain = _pick(rng, NORMAL_BODY)
        impression_plain = _pick(rng, NORMAL_IMPRESSION)
    else:
        rng.shuffle(sentences)
        body_plain = " ".join(sentences)
        impression_plain = _format_impression(_impression_lines(labels, s, rng), rng)

    # Pick a top-level structural style
    style_roll = rng.random()

    if style_roll < 0.30 and sentences:
        # Style A: sectioned with anatomical sub-headers in FINDINGS
        buckets = _bucket_sentences(s, sentences)
        findings_block = []
        for name, sents in buckets.items():
            if not sents: continue
            findings_block.append(f"{name}: " + " ".join(sents))
        findings_text = "\n".join(findings_block)
        parts = [f"TECHNIQUE: {technique}"]
        if has_indication:
            parts.append(f"CLINICAL INDICATION: {indication}")
        if comparison:
            parts.append(f"COMPARISON: {comparison}")
        parts.append("FINDINGS:\n" + findings_text)
        parts.append("IMPRESSION:\n" + impression_plain)
        text = "\n\n".join(parts)

    elif style_roll < 0.80:
        # Style B: sectioned, prose FINDINGS body
        parts = [f"TECHNIQUE: {technique}"]
        if has_indication:
            parts.append(f"CLINICAL INDICATION: {indication}")
        if comparison:
            parts.append(f"COMPARISON: {comparison}")
        parts.append(f"FINDINGS: {body_plain}")
        parts.append(f"IMPRESSION: {impression_plain}" if "\n" not in impression_plain
                     else "IMPRESSION:\n" + impression_plain)
        text = "\n\n".join(parts)

    else:
        # Style C: pure prose, no headers
        intro = technique
        if has_indication:
            intro = f"{technique} Indication: {indication.lower()}"
        text = f"{intro} {body_plain} Impression: {impression_plain.replace(chr(10), ' ')}"

    return {
        "text": text,
        "findings": body_plain,
        "impression": impression_plain,
    }
