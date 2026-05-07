"""Plot convergence of generated masks over training epochs.

Reads the per-epoch NPZ snapshots written by ``train_tuna.py`` when
``--sample-mask-every > 0`` was passed (``cls+mask`` mode). Produces a
single grid PNG: rows are sample images, columns are training epochs,
with the input image in the leftmost column and the GT mask overlay
just to its right for reference.

Usage:
    python scripts/plot_mask_convergence.py \\
        --snaps  experiments/results/n0010000_pixel_cls_mask_s0_masks \\
        --out    docs/images/mask_convergence_pixel_n10k.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.dataset import MASK_LABELS


# Pick a few salient channels to visualize (any present finding stands out)
SHOW_CHANNELS = [
    ("cardiomegaly",      (1.0, 0.30, 0.30)),
    ("effusion_left",     (0.30, 0.50, 1.0)),
    ("effusion_right",    (0.30, 0.85, 0.85)),
    ("consolidation",     (0.95, 0.20, 0.40)),
    ("nodules",           (0.30, 0.85, 0.30)),
    ("kerley_b_lines",    (1.00, 0.92, 0.40)),
    ("reticular_pattern", (0.60, 0.60, 1.0)),
    ("honeycombing",      (1.0,  0.50, 0.85)),
]


def composite(img_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """img_gray: (H, W) in [0, 1]; mask: (n_findings, h_m, w_m) at any
    resolution (GT is 320×320, predicted is the token-grid side, e.g.
    16×16). Returns (H, W, 3) overlay of all SHOW_CHANNELS that have
    non-trivial intensity, blended by mask amplitude."""
    from PIL import Image as PILImage
    H, W = img_gray.shape
    rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    name_to_idx = {n: i for i, n in enumerate(MASK_LABELS)}
    for name, color in SHOW_CHANNELS:
        idx = name_to_idx[name]
        ch = np.clip(mask[idx].astype(np.float32), 0.0, 1.0)
        if ch.max() < 0.05:
            continue
        # Resize to image size with nearest (preserves the soft mask
        # falloff but doesn't introduce new intermediate values from the
        # token grid)
        if ch.shape != (H, W):
            up = np.array(
                PILImage.fromarray((ch * 255.0).astype(np.uint8))
                .resize((W, H), PILImage.NEAREST)
            ).astype(np.float32) / 255.0
        else:
            up = ch
        col = np.array(color, dtype=np.float32)[None, None, :]
        rgb = rgb * (1.0 - 0.55 * up[..., None]) + col * 0.55 * up[..., None]
    return np.clip(rgb, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snaps", required=True,
                        help="directory with _inputs.npz + epoch_*.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=6,
                        help="how many sample images to show as rows")
    args = parser.parse_args()

    snap_dir = Path(args.snaps)
    inputs = np.load(snap_dir / "_inputs.npz")
    images = inputs["images"]   # [B, 3, H, W] in [0, 1]
    gts = inputs["gt_masks"]    # [B, n_findings, 320, 320]
    ids = inputs["ids"]
    rows = min(args.rows, images.shape[0])

    epochs = sorted(int(p.stem.split("_")[1]) for p in snap_dir.glob("epoch_*.npz"))
    if not epochs:
        print(f"no epoch_*.npz under {snap_dir}"); return
    print(f"epochs found: {epochs}")

    # Columns = [image, GT, e1, e2, ...]
    cols = 2 + len(epochs)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows),
                              squeeze=False)
    for r in range(rows):
        # Greyscale base image  (channel-0 of the 3-channel grayscale clone)
        gray = images[r, 0]
        gt = gts[r]
        gt_img = composite(gray, gt)
        axes[r, 0].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("input" if r == 0 else "", fontsize=10)
        axes[r, 1].imshow(gt_img)
        axes[r, 1].set_title("GT mask" if r == 0 else "", fontsize=10)
        for c, ep in enumerate(epochs):
            data = np.load(snap_dir / f"epoch_{ep:03d}.npz")
            pred = data["pred"][r]
            axes[r, c + 2].imshow(composite(gray, pred))
            if r == 0:
                axes[r, c + 2].set_title(f"epoch {ep}", fontsize=10)
        # row label = id
        axes[r, 0].set_ylabel(str(ids[r]), fontsize=8, rotation=0,
                               ha="right", va="center", labelpad=10)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Mask-prediction convergence — sampled via rectified-flow integration",
                  fontsize=12, y=0.995)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
