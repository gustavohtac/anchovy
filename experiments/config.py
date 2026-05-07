"""Declarative experiment matrix.

Defines the cross-product of

    data regime ×  variant  ×  mode  ×  seed

into a deterministic list of :class:`Experiment` records consumed by the
queue runner.  Editing this file is the *only* way to expand the sweep —
ids are derived from the configuration so no two distinct configs collide.

Scaling hypothesis
------------------
We want to test whether pixel-patch front-ends (TUNA-2-style) overtake
encoder- and VAE-based front-ends as the training set grows.  The minimal
controlled comparison is

    front-end ∈ {pixel, encoder, vae}
    mode      ∈ {cls, cls+diff}
    data size ∈ {1k, 10k, 100k, 600k}

with the same backbone, the same val/test split, and the same optimizer
hyperparameters across all cells.  Multiple seeds can be added by
extending :data:`SEEDS`.

Per-regime epoch counts are scaled sublinearly with the dataset size so
that the total training compute (samples × epochs) stays within an order
of magnitude across regimes — see :data:`EPOCHS_BY_REGIME`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


# ---------------- sweep axes ----------------

DATA_REGIMES: Sequence[int] = (1_000, 10_000, 100_000, 600_000)
VARIANTS:     Sequence[str] = ("pixel", "encoder", "vae")
MODES:        Sequence[str] = ("cls", "cls+diff")
SEEDS:        Sequence[int] = (0,)  # extend e.g. to (0, 1, 2) for 3-seed reps

# Total samples seen across regimes (epochs × data_size):
#   1k    × 50 =    50,000
#   10k   × 30 =   300,000
#   100k  × 10 = 1,000,000
#   600k  × 4  = 2,400,000
EPOCHS_BY_REGIME: dict[int, int] = {
    1_000:     50,
    10_000:    30,
    100_000:   10,
    600_000:    4,
}


@dataclass(frozen=True)
class Experiment:
    """A single reproducible training run.

    The :attr:`id` encodes every dimension of the sweep — anyone reading
    a result file or log can recover the exact configuration without
    consulting any external table.
    """
    data_size: int
    variant: str            # "pixel" | "encoder" | "vae"
    mode: str               # "cls"   | "cls+diff"
    seed: int
    epochs: int
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    embed_dim: int = 768
    depth: int = 12
    n_heads: int = 12
    image_size: int = 224
    patch_size: int = 14
    mask_ratio: float = 0.5
    diff_weight: float = 1.0

    @property
    def id(self) -> str:
        mode_tag = self.mode.replace("+", "_")
        return f"n{self.data_size:>07d}_{self.variant}_{mode_tag}_s{self.seed}"


def all_experiments() -> list[Experiment]:
    """Materialize the full cross-product as a deterministic list.

    Order is (data_size, variant, mode, seed) so that smaller experiments
    surface first when the queue is consumed top-to-bottom.
    """
    out: list[Experiment] = []
    for n in DATA_REGIMES:
        for v in VARIANTS:
            for m in MODES:
                for s in SEEDS:
                    out.append(Experiment(
                        data_size=n, variant=v, mode=m, seed=s,
                        epochs=EPOCHS_BY_REGIME[n],
                    ))
    return out
