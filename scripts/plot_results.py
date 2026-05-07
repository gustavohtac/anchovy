"""Render the figures used in the README from the on-disk results.

Outputs (under ``docs/images/``):

    scaling_curves.png  — atomic / syndrome AUC and AP versus training-set
                          size for each (variant, mode), one panel per metric;
    sample_radiograph.png  — one synthetic chest-radiograph image;
    sample_overlay.png     — same image with all per-finding soft masks
                              colour-blended on top.

Usage:
    python scripts/plot_results.py \\
        --results docs/results/results.csv \\
        --dataset datasets/n10000      # any size: only used for sample images
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- finding -> colour for mask overlays ----
PALETTE = {
    "cardiomegaly":              (235,  64,  52),
    "pulm_venous_redistribution":(255, 140,   0),
    "hilar_adenopathy":          (255,  80, 200),
    "effusion_left":             ( 30, 144, 255),
    "effusion_right":            ( 64, 224, 208),
    "pneumothorax_left":         (255, 215,   0),
    "pneumothorax_right":        (218, 165,  32),
    "pleural_thickening_left":   (138,  43, 226),
    "pleural_thickening_right":  (186,  85, 211),
    "volume_loss":               (160,  82,  45),
    "consolidation":             (220,  20,  60),
    "reticular_pattern":         (100, 149, 237),
    "honeycombing":              (255, 105, 180),
    "kerley_b_lines":            (240, 230, 140),
    "nodules":                   ( 50, 205,  50),
    "micronodules":              (152, 251, 152),
}


VARIANT_STYLE = {
    "pixel":   ("#0072B2", "o"),
    "encoder": ("#D55E00", "s"),
    "vae":     ("#009E73", "^"),
}

MODE_STYLE = {"cls": "-", "cls+diff": "--"}


# ---------------- scaling curves ----------------


def plot_scaling(results_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(results_csv)
    df = df[df["status"] == "done"].copy()
    if df.empty:
        print(f"[skip] no completed runs in {results_csv}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    metrics = [
        ("atomic_auc",   "Atomic findings — macro AUC"),
        ("atomic_ap",    "Atomic findings — macro AP"),
        ("syndrome_auc", "Clinical syndromes — macro AUC"),
        ("syndrome_ap",  "Clinical syndromes — macro AP"),
    ]
    flat_axes = axes.flat
    for ax, (col, title) in zip(flat_axes, metrics):
        for variant in ("pixel", "encoder", "vae"):
            color, marker = VARIANT_STYLE[variant]
            for mode in ("cls", "cls+diff"):
                style = MODE_STYLE[mode]
                sub = df[(df.variant == variant) & (df["mode"] == mode)]
                if sub.empty:
                    continue
                sub = sub.sort_values("data_size")
                label = f"{variant} · {mode}"
                ax.plot(sub["data_size"], sub[col], style, color=color,
                        marker=marker, markersize=8, linewidth=1.8,
                        label=label, alpha=0.9)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=13)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
        ax.set_ylim(bottom=max(0.0, df[col].min() - 0.05))
        ax.tick_params(labelsize=10)
    # x-label on the bottom row only
    for ax in axes[-1, :]:
        ax.set_xlabel("training samples (log)", fontsize=11)
    # y-label on the left column only
    for ax in axes[:, 0]:
        ax.set_ylabel("metric on test set", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=11)
    fig.suptitle("TUNA-mini scaling sweep — variant × mode × data regime",
                  fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------- sample image + mask overlay ----------------


def _pick_sample(dataset_dir: Path) -> str | None:
    """Pick the test-set image with the most distinct finding masks."""
    masks_dir = dataset_dir / "masks"
    if not masks_dir.exists():
        return None
    best_id, best_n = None, -1
    candidates = sorted(masks_dir.glob("*.npz"))[:200]
    for p in candidates:
        try:
            data = np.load(p)
            n = len(data.files)
        except Exception:
            continue
        if n > best_n:
            best_n, best_id = n, p.stem
    return best_id


def plot_sample(dataset_dir: Path, out_image: Path, out_overlay: Path) -> None:
    img_id = _pick_sample(dataset_dir)
    if img_id is None:
        print(f"[skip] no masks under {dataset_dir}")
        return
    img = np.array(Image.open(dataset_dir / "images" / f"{img_id}.png"))
    masks = np.load(dataset_dir / "masks" / f"{img_id}.npz")

    out_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_image)
    print(f"wrote {out_image}  (id={img_id}, masks={list(masks.files)})")

    SCALE = 3
    img_big = np.array(Image.fromarray(img).resize(
        (img.shape[1] * SCALE, img.shape[0] * SCALE), Image.BILINEAR))
    rgb = np.stack([img_big, img_big, img_big], axis=-1).astype(np.float32)
    legend: list[tuple[str, tuple[int, int, int]]] = []
    for label in masks.files:
        col = np.array(PALETTE.get(label, (200, 200, 200)), dtype=np.float32)
        m = masks[label].astype(np.float32) / 255.0
        m_big = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
            (img.shape[1] * SCALE, img.shape[0] * SCALE), Image.BILINEAR)
        ).astype(np.float32) / 255.0
        rgb = rgb * (1.0 - 0.55 * m_big[..., None]) + col * 0.55 * m_big[..., None]
        legend.append((label, tuple(int(c) for c in col)))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    canvas = Image.new("RGB", (w + 360, h), (20, 20, 22))
    canvas.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        font_label = ImageFont.truetype("DejaVuSans.ttf", 18)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    draw.text((w + 24, 24), img_id, font=font_title, fill=(245, 245, 245))
    draw.text((w + 24, 56), "soft masks", font=font_label, fill=(180, 180, 180))
    for i, (label, color) in enumerate(legend):
        ly = 100 + i * 32
        draw.rectangle([w + 24, ly, w + 50, ly + 22], fill=color)
        draw.text((w + 60, ly), label, font=font_label, fill=(230, 230, 230))
    canvas.save(out_overlay)
    print(f"wrote {out_overlay}")


# ---------------- entrypoint ----------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="docs/results/results.csv",
                        help="CSV produced by experiments.aggregate")
    parser.add_argument("--dataset", default="datasets/n10000",
                        help="any generated dataset to sample one image from")
    parser.add_argument("--out", default="docs/images")
    args = parser.parse_args()

    out = Path(args.out)
    plot_scaling(Path(args.results), out / "scaling_curves.png")
    plot_sample(
        Path(args.dataset),
        out / "sample_radiograph.png",
        out / "sample_overlay.png",
    )


if __name__ == "__main__":
    main()
