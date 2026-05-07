"""Plot per-epoch training curves from a long-run result JSON.

The result JSONs written by ``models.train_tuna`` contain a ``history``
field with one entry per epoch — train losses, val metrics, and (when
``--track-test > 0`` was passed) test metrics. This script renders those
into a 4-panel figure showing how each metric evolved during training,
which is the standard way to read off convergence and overfitting from a
single run.

Usage:
    python scripts/plot_training_curves.py \\
        --runs path/to/run_a.json path/to/run_b.json \\
        --out  docs/images/training_curves.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]


def _series(history: list[dict], path: str) -> tuple[list[int], list[float]]:
    """Pull (epoch, value) from history under a dotted path like
    'val_atomic.macro_auc'. Skips entries where the value is None."""
    keys = path.split(".")
    xs, ys = [], []
    for h in history:
        v: object = h
        for k in keys:
            if not isinstance(v, dict) or k not in v:
                v = None; break
            v = v[k]
        if v is None: continue
        xs.append(h["epoch"]); ys.append(float(v))
    return xs, ys


def _label_for(meta: dict) -> str:
    return f"{meta.get('variant','?')} · {meta.get('mode','?')}  (n={meta.get('data_size','?')})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="paths to result JSONs produced by train_tuna")
    parser.add_argument("--out", default="docs/images/training_curves.png")
    args = parser.parse_args()

    runs = []
    for p in args.runs:
        d = json.loads(Path(p).read_text())
        # Some result files don't carry data_size — accept missing meta
        runs.append(d)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    panels = [
        ("cls loss (train)",        "cls_loss",                   axes[0, 0], False),
        ("atomic AUC (val + test)", "{set}_atomic.macro_auc",      axes[0, 1], True),
        ("atomic AP (val + test)",  "{set}_atomic.macro_ap",       axes[1, 0], True),
        ("syndrome AUC (val + test)", "{set}_syndromes.macro_auc", axes[1, 1], True),
    ]

    for title, key, ax, has_split in panels:
        for i, run in enumerate(runs):
            colour = COLOURS[i % len(COLOURS)]
            label = _label_for(run)
            if has_split:
                xv, yv = _series(run["history"], key.format(set="val"))
                xt, yt = _series(run["history"], key.format(set="test"))
                if xv:
                    ax.plot(xv, yv, "-", color=colour, label=label + " · val",
                            linewidth=1.6, alpha=0.9)
                if xt:
                    ax.plot(xt, yt, "--", color=colour, marker="o",
                            markersize=4, label=label + " · test",
                            linewidth=1.4, alpha=0.9)
            else:
                xs, ys = _series(run["history"], key)
                if xs:
                    ax.plot(xs, ys, "-", color=colour, label=label,
                            linewidth=1.6, alpha=0.9)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("epoch", fontsize=11)
    axes[0, 0].set_ylabel("loss", fontsize=11)
    axes[0, 1].set_ylabel("metric", fontsize=11)
    axes[1, 0].set_ylabel("metric", fontsize=11)
    axes[1, 1].set_ylabel("metric", fontsize=11)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.suptitle("Long-run training curves — per-epoch train / val / test",
                  fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
