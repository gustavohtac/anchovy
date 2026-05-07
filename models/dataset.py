"""PyTorch Dataset for the v3 synthetic radiograph data.

Loads images for one split (train/val/test), applies the standard DINOv2
input pipeline (224×224 RGB, ImageNet-normalized), and stacks both label
groups (atomic findings and syndromes) up-front so feature extractors can
read them as plain numpy arrays.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


# Binary label columns we probe on. *_severity columns (multi-class integers)
# are intentionally excluded — linear probe targets are binary multi-label.
ATOMIC_COLS = [
    "cardiomegaly", "pulm_venous_redistribution", "hilar_adenopathy",
    "effusion_left", "effusion_right",
    "pneumothorax_left", "pneumothorax_right",
    "pleural_thickening_left", "pleural_thickening_right",
    "volume_loss",
    "consolidation", "consolidation_air_bronchogram",
    "reticular_pattern", "honeycombing", "kerley_b_lines",
]

SYNDROME_COLS = [
    "chf_syndrome", "uip_pattern", "sarcoidosis_pattern", "tb_pattern",
    "mesothelioma_pattern", "tension_pneumothorax",
    "metastatic_pattern", "miliary_pattern",
    "solitary_pulmonary_nodule", "lobar_pneumonia_pattern",
]

# Soft segmentation channels emitted by data.render. Order matters: it
# defines the (n_findings)-axis layout of the mask tensor used by
# ``mode='cls+mask'`` in models.tuna_mini.
MASK_LABELS = [
    "cardiomegaly", "pulm_venous_redistribution", "hilar_adenopathy",
    "effusion_left", "effusion_right",
    "pneumothorax_left", "pneumothorax_right",
    "pleural_thickening_left", "pleural_thickening_right",
    "volume_loss", "consolidation",
    "reticular_pattern", "honeycombing", "kerley_b_lines",
    "nodules", "micronodules",
]


def make_transform(image_size: int = 224):
    """DINOv2-compatible input pipeline: convert grayscale PNG to 3-channel
    RGB, resize to a square divisible by patch_size=14, normalize with
    ImageNet stats (DINOv2 was pre-trained that way)."""
    return transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class RadiographDataset(torch.utils.data.Dataset):
    """Yields (image_tensor, atomic_label_vec, syndrome_label_vec) per item."""

    def __init__(self, root: str | Path, split: str, image_size: int = 224):
        self.root = Path(root)
        with (self.root / "splits" / f"{split}.txt").open() as f:
            self.ids: list[str] = [ln.strip() for ln in f if ln.strip()]
        df = pd.read_csv(self.root / "labels.csv").set_index("image_id")
        df = df.loc[self.ids]
        self.atomic = df[ATOMIC_COLS].values.astype(np.float32)
        self.syndromes = df[SYNDROME_COLS].values.astype(np.float32)
        self.transform = make_transform(image_size)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = Image.open(self.root / "images" / f"{self.ids[i]}.png")
        return self.transform(img), self.atomic[i], self.syndromes[i]
